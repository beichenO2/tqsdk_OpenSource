"""Supertrend 策略 — ATR 自适应趋势跟踪通道。

Supertrend (Olivier Seban, 2000s, 经典趋势指标):
  Upper Band = (H + L) / 2 + multiplier * ATR
  Lower Band = (H + L) / 2 - multiplier * ATR
  趋势 = 上升时用 Lower Band, 下降时用 Upper Band

优势：
  - 自动适应波动率（ATR 随市场调整）
  - 产生明确的多空翻转信号
  - 避免过早出场（趋势跟踪特性）

日内适配：
  - 较短的 ATR 周期（10-14）+ 较小的乘数（2-3）
  - 配合 IntradayGuard 强制收盘平仓

Method: Supertrend indicator (Seban 2000s, 广泛使用的经典趋势工具)
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..indicators import calc_atr
from ..mixins import EMASlopeRegimeMixin, SignalBalanceMixin
from ..registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS: dict[str, Any] = {
    "atr_period": 10,
    "multiplier": 2.5,
    "confirm_bars": 2,
    "max_hold_bars": 50,
    "cooldown_bars": 2,
}


@auto_register("supertrend")
class SupertrendStrategy(SignalBalanceMixin, EMASlopeRegimeMixin, BaseStrategy):
    """Supertrend 日内趋势策略 — ATR 通道翻转信号。"""

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

        self._upper_band = 0.0
        self._lower_band = 0.0
        self._supertrend = 0.0
        self._trend = 0  # 1=up, -1=down
        self._prev_trend = 0
        self._trend_change_count = 0

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        h = float(bar.get("high", 0))
        l = float(bar.get("low", 0))
        c = float(bar.get("close", 0))

        self._highs.append(h)
        self._lows.append(l)
        self._closes.append(c)
        self._bar_count += 1
        self._regime_update(c)

        atr_period = self.get_param("atr_period", 10)
        if self._bar_count < atr_period + 3:
            return []

        atr = calc_atr(list(self._highs), list(self._lows), list(self._closes), atr_period)
        if atr is None or atr < 1e-10:
            return []

        mult = self.get_param("multiplier", 2.5)
        mid = (h + l) / 2
        new_upper = mid + mult * atr
        new_lower = mid - mult * atr

        if self._upper_band > 0:
            if new_upper < self._upper_band or list(self._closes)[-2] > self._upper_band:
                self._upper_band = new_upper
            if new_lower > self._lower_band or list(self._closes)[-2] < self._lower_band:
                self._lower_band = new_lower
        else:
            self._upper_band = new_upper
            self._lower_band = new_lower

        self._prev_trend = self._trend
        if c > self._upper_band:
            self._trend = 1
            self._supertrend = self._lower_band
        elif c < self._lower_band:
            self._trend = -1
            self._supertrend = self._upper_band
        else:
            if self._trend == 0:
                self._trend = 1 if c > mid else -1
            self._supertrend = self._lower_band if self._trend == 1 else self._upper_band

        trend_changed = self._trend != self._prev_trend and self._prev_trend != 0
        if trend_changed:
            self._trend_change_count += 1
        else:
            self._trend_change_count = max(0, self._trend_change_count)

        signals = []

        if self._position_side:
            self._hold_bars += 1
            max_hold = self.get_param("max_hold_bars", 50)

            reversed_trend = (self._position_side == "long" and self._trend == -1) or \
                            (self._position_side == "short" and self._trend == 1)

            if self._hold_bars >= max_hold or reversed_trend:
                exit_type = SignalType.LONG_EXIT if self._position_side == "long" else SignalType.SHORT_EXIT
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_type, strength=0.8, price=c,
                    reason=f"st_exit: trend={self._trend} hold={self._hold_bars}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = None
                self._hold_bars = 0
                self._cd = self.get_param("cooldown_bars", 2)
                return signals

        if self._cd > 0:
            self._cd -= 1

        confirm = self.get_param("confirm_bars", 2)

        if not self._position_side and self._cd <= 0 and trend_changed:
            if self._trend == 1 and self._regime_allow(SignalType.LONG_ENTRY) and self._sb_allow(SignalType.LONG_ENTRY):
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=0.8, price=c,
                    reason=f"st_buy: trend_flip_up c={c:.1f} st={self._supertrend:.1f} regime={self._regime()}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._sb_record(SignalType.LONG_ENTRY)
                self._position_side = "long"
                self._entry_price = c
                self._hold_bars = 0

            elif self._trend == -1 and self._regime_allow(SignalType.SHORT_ENTRY) and self._sb_allow(SignalType.SHORT_ENTRY):
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=0.8, price=c,
                    reason=f"st_sell: trend_flip_down c={c:.1f} st={self._supertrend:.1f} regime={self._regime()}",
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
