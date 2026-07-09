"""Keltner Channel 策略 — EMA + ATR 通道突破/回调。

Chester Keltner (1960, 经典技术分析) → Linda Raschke 改进版 (1990s):
  Middle = EMA(close, period)
  Upper = Middle + multiplier * ATR
  Lower = Middle - multiplier * ATR

与布林带的区别：
  - 布林带用标准差（对极端值敏感）
  - Keltner 用 ATR（更稳定，反映真实波动范围）
  - 更适合趋势市场，布林带更适合震荡市场

信号构建：
  1. 价格突破上通道 → 趋势做多
  2. 价格突破下通道 → 趋势做空
  3. 回到中线附近 → 平仓
  4. Squeeze 检测：当布林带在 Keltner 通道内 → 即将爆发
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..indicators import calc_atr
from ..mixins import EMASlopeRegimeMixin, SignalBalanceMixin
from ..registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS: dict[str, Any] = {
    "ema_period": 20,
    "atr_period": 14,
    "multiplier": 2.0,
    "exit_at_middle": True,
    "squeeze_detect": True,
    "bb_period": 20,
    "bb_mult": 2.0,
    "max_hold_bars": 40,
    "cooldown_bars": 3,
}


def _ema(values: list[float], period: int) -> float:
    if len(values) < period:
        return np.mean(values)
    alpha = 2.0 / (period + 1)
    ema_val = values[-period]
    for v in values[-period + 1:]:
        ema_val = alpha * v + (1 - alpha) * ema_val
    return ema_val


@auto_register("keltner_channel")
class KeltnerChannelStrategy(SignalBalanceMixin, EMASlopeRegimeMixin, BaseStrategy):
    """Keltner Channel 趋势突破策略。"""

    def __init__(self, config: StrategyConfig) -> None:
        config = config.model_copy(
            update={"params": {**DEFAULT_PARAMS, **config.params}}
        )
        super().__init__(config)
        self._closes: deque[float] = deque(maxlen=200)
        self._highs: deque[float] = deque(maxlen=200)
        self._lows: deque[float] = deque(maxlen=200)
        self._bar_count = 0
        self._position_side: str | None = None
        self._entry_price = 0.0
        self._hold_bars = 0
        self._cd = 0
        self._in_squeeze = False

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        h = float(bar.get("high", 0))
        l = float(bar.get("low", 0))
        c = float(bar.get("close", 0))

        self._highs.append(h)
        self._lows.append(l)
        self._closes.append(c)
        self._bar_count += 1
        self._regime_update(c)

        ema_period = self.get_param("ema_period", 20)
        if self._bar_count < ema_period + 5:
            return []

        atr = calc_atr(list(self._highs), list(self._lows), list(self._closes),
                       self.get_param("atr_period", 14))
        if atr is None or atr < 1e-10:
            return []

        close_list = list(self._closes)
        ema_val = _ema(close_list, ema_period)
        mult = self.get_param("multiplier", 2.0)

        upper = ema_val + mult * atr
        lower = ema_val - mult * atr

        if self.get_param("squeeze_detect", True):
            bb_period = self.get_param("bb_period", 20)
            bb_mult = self.get_param("bb_mult", 2.0)
            if len(close_list) >= bb_period:
                bb_mid = np.mean(close_list[-bb_period:])
                bb_std = np.std(close_list[-bb_period:])
                bb_upper = bb_mid + bb_mult * bb_std
                bb_lower = bb_mid - bb_mult * bb_std
                self._in_squeeze = bb_upper < upper and bb_lower > lower

        signals = []

        if self._position_side:
            self._hold_bars += 1
            max_hold = self.get_param("max_hold_bars", 40)
            exit_at_middle = self.get_param("exit_at_middle", True)

            at_middle = abs(c - ema_val) < 0.3 * atr if exit_at_middle else False

            if self._hold_bars >= max_hold or at_middle:
                exit_type = SignalType.LONG_EXIT if self._position_side == "long" else SignalType.SHORT_EXIT
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_type, strength=0.8, price=c,
                    reason=f"kc_exit: hold={self._hold_bars} at_middle={at_middle}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = None
                self._hold_bars = 0
                self._cd = self.get_param("cooldown_bars", 3)
                return signals

        if self._cd > 0:
            self._cd -= 1

        if not self._position_side and self._cd <= 0:
            squeeze_bonus = " squeeze!" if self._in_squeeze else ""

            if c > upper and self._regime_allow(SignalType.LONG_ENTRY) and self._sb_allow(SignalType.LONG_ENTRY):
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=0.9 if self._in_squeeze else 0.7, price=c,
                    reason=f"kc_break_up: c={c:.1f} upper={upper:.1f} ema={ema_val:.1f}{squeeze_bonus} regime={self._regime()}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._sb_record(SignalType.LONG_ENTRY)
                self._position_side = "long"
                self._entry_price = c
                self._hold_bars = 0

            elif c < lower and self._regime_allow(SignalType.SHORT_ENTRY) and self._sb_allow(SignalType.SHORT_ENTRY):
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=0.9 if self._in_squeeze else 0.7, price=c,
                    reason=f"kc_break_down: c={c:.1f} lower={lower:.1f} ema={ema_val:.1f}{squeeze_bonus} regime={self._regime()}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._sb_record(SignalType.SHORT_ENTRY)
                self._position_side = "short"
                self._entry_price = c
                self._hold_bars = 0

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        return []
