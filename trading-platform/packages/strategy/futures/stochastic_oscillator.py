"""随机指标策略 — %K/%D 交叉 + 超买超卖区域信号。

George Lane (1950s, 经典):
  %K = (C - Low_N) / (High_N - Low_N) * 100
  %D = SMA(%K, 3)

信号：
  1. %K 在超卖区（<20）上穿 %D → 做多
  2. %K 在超买区（>80）下穿 %D → 做空
  3. 价格背离检测（与 RSI 背离类似）
  4. 中间区域不交易（避免噪声）

Method: Stochastic Oscillator (Lane 1950s, 经典动量指标)
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..indicators import calc_atr
from ..registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS: dict[str, Any] = {
    "k_period": 14,
    "d_period": 3,
    "overbought": 80,
    "oversold": 20,
    "atr_period": 14,
    "tp_atr_mult": 2.0,
    "sl_atr_mult": 1.0,
    "max_hold_bars": 30,
    "cooldown_bars": 3,
}


@auto_register("stochastic_oscillator")
class StochasticOscillatorStrategy(BaseStrategy):
    """随机指标策略 — %K/%D 交叉信号。"""

    def __init__(self, config: StrategyConfig) -> None:
        config = config.model_copy(
            update={"params": {**DEFAULT_PARAMS, **config.params}}
        )
        super().__init__(config)
        self._closes: deque[float] = deque(maxlen=200)
        self._highs: deque[float] = deque(maxlen=200)
        self._lows: deque[float] = deque(maxlen=200)
        self._k_values: deque[float] = deque(maxlen=50)
        self._bar_count = 0
        self._position_side: str | None = None
        self._entry_price = 0.0
        self._hold_bars = 0
        self._cd = 0

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        h = float(bar.get("high", 0))
        l = float(bar.get("low", 0))
        c = float(bar.get("close", 0))

        self._highs.append(h)
        self._lows.append(l)
        self._closes.append(c)
        self._bar_count += 1

        k_period = self.get_param("k_period", 14)
        d_period = self.get_param("d_period", 3)

        if self._bar_count < k_period + d_period + 3:
            return []

        high_list = list(self._highs)
        low_list = list(self._lows)

        highest = max(high_list[-k_period:])
        lowest = min(low_list[-k_period:])
        hl_range = highest - lowest

        k = ((c - lowest) / hl_range * 100) if hl_range > 1e-10 else 50.0
        self._k_values.append(k)

        if len(self._k_values) < d_period + 1:
            return []

        k_list = list(self._k_values)
        d = np.mean(k_list[-d_period:])
        d_prev = np.mean(k_list[-d_period - 1:-1])
        k_prev = k_list[-2]

        atr = calc_atr(list(self._highs), list(self._lows), list(self._closes),
                       self.get_param("atr_period", 14))
        if atr is None or atr < 1e-10:
            return []

        signals = []
        oversold = self.get_param("oversold", 20)
        overbought = self.get_param("overbought", 80)

        k_cross_up = k_prev <= d_prev and k > d
        k_cross_down = k_prev >= d_prev and k < d

        if self._position_side:
            self._hold_bars += 1
            max_hold = self.get_param("max_hold_bars", 30)
            tp_mult = self.get_param("tp_atr_mult", 2.0)
            sl_mult = self.get_param("sl_atr_mult", 1.0)

            pnl = (c - self._entry_price) / self._entry_price if self._position_side == "long" else (self._entry_price - c) / self._entry_price
            if pnl >= tp_mult * atr / self._entry_price or pnl <= -sl_mult * atr / self._entry_price or self._hold_bars >= max_hold:
                exit_type = SignalType.LONG_EXIT if self._position_side == "long" else SignalType.SHORT_EXIT
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_type, strength=0.8, price=c,
                    reason=f"stoch_exit: K={k:.0f} D={d:.0f} hold={self._hold_bars}",
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
            if k_cross_up and k < oversold + 10 and d < oversold + 10:
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=0.8, price=c,
                    reason=f"stoch_buy: K={k:.0f} crosses D={d:.0f} in oversold zone",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "long"
                self._entry_price = c
                self._hold_bars = 0

            elif k_cross_down and k > overbought - 10 and d > overbought - 10:
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=0.8, price=c,
                    reason=f"stoch_sell: K={k:.0f} crosses D={d:.0f} in overbought zone",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "short"
                self._entry_price = c
                self._hold_bars = 0

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        return []
