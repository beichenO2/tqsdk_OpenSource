"""Williams %R 策略 — 超买超卖极端区域反转信号。

Larry Williams (1973, 经典):
  %R = (Highest_N - Close) / (Highest_N - Lowest_N) * -100

范围 [-100, 0]:
  -100 到 -80 = 超卖区 → 做多信号
  -20 到 0 = 超买区 → 做空信号

与随机指标的区别：Williams %R 更灵敏（不做平滑），适合短线日内交易。

Method: Williams %R (Larry Williams 1973, 经典短线指标)
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..indicators import calc_atr
from ..registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS: dict[str, Any] = {
    "period": 14,
    "overbought": -20,
    "oversold": -80,
    "confirm_bars": 2,
    "atr_period": 14,
    "tp_atr_mult": 1.5,
    "sl_atr_mult": 0.8,
    "max_hold_bars": 20,
    "cooldown_bars": 2,
}


@auto_register("williams_r")
class WilliamsRStrategy(BaseStrategy):
    """Williams %R 策略 — 极端区域反转。"""

    def __init__(self, config: StrategyConfig) -> None:
        config = config.model_copy(update={"params": {**DEFAULT_PARAMS, **config.params}})
        super().__init__(config)
        self._closes: deque[float] = deque(maxlen=200)
        self._highs: deque[float] = deque(maxlen=200)
        self._lows: deque[float] = deque(maxlen=200)
        self._wr_values: deque[float] = deque(maxlen=50)
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

        period = self.get_param("period", 14)
        if self._bar_count < period + 5:
            return []

        high_list = list(self._highs)
        low_list = list(self._lows)
        highest = max(high_list[-period:])
        lowest = min(low_list[-period:])
        hl_range = highest - lowest

        wr = ((highest - c) / hl_range * -100) if hl_range > 1e-10 else -50.0
        self._wr_values.append(wr)

        atr = calc_atr(list(self._highs), list(self._lows), list(self._closes),
                       self.get_param("atr_period", 14))
        if atr is None or atr < 1e-10:
            return []

        signals = []
        oversold = self.get_param("oversold", -80)
        overbought = self.get_param("overbought", -20)

        wr_list = list(self._wr_values)
        confirm = self.get_param("confirm_bars", 2)
        was_oversold = len(wr_list) >= confirm + 1 and all(w < oversold for w in wr_list[-confirm - 1:-1])
        leaving_oversold = was_oversold and wr > oversold

        was_overbought = len(wr_list) >= confirm + 1 and all(w > overbought for w in wr_list[-confirm - 1:-1])
        leaving_overbought = was_overbought and wr < overbought

        if self._position_side:
            self._hold_bars += 1
            max_hold = self.get_param("max_hold_bars", 20)
            tp_mult = self.get_param("tp_atr_mult", 1.5)
            sl_mult = self.get_param("sl_atr_mult", 0.8)

            pnl = (c - self._entry_price) / self._entry_price if self._position_side == "long" else (self._entry_price - c) / self._entry_price
            if pnl >= tp_mult * atr / self._entry_price or pnl <= -sl_mult * atr / self._entry_price or self._hold_bars >= max_hold:
                exit_type = SignalType.LONG_EXIT if self._position_side == "long" else SignalType.SHORT_EXIT
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_type, strength=0.8, price=c,
                    reason=f"wr_exit: %R={wr:.0f} hold={self._hold_bars}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = None
                self._hold_bars = 0
                self._cd = self.get_param("cooldown_bars", 2)
                return signals

        if self._cd > 0:
            self._cd -= 1

        if not self._position_side and self._cd <= 0:
            if leaving_oversold:
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=0.8, price=c,
                    reason=f"wr_buy: %R={wr:.0f} leaving oversold zone",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "long"
                self._entry_price = c
                self._hold_bars = 0

            elif leaving_overbought:
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=0.8, price=c,
                    reason=f"wr_sell: %R={wr:.0f} leaving overbought zone",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "short"
                self._entry_price = c
                self._hold_bars = 0

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        return []
