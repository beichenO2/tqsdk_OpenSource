"""MACD 柱状图策略 — 经典动量指标的增强版。

Gerald Appel (1979, 经典):
  MACD = EMA(12) - EMA(26)
  Signal = EMA(MACD, 9)
  Histogram = MACD - Signal

增强信号：
  1. 直方图从负变正（零轴交叉）→ 做多
  2. 直方图缩短（动量衰减）→ 平仓
  3. 直方图峰值下降但价格创新高 → 背离
  4. 结合 ATR 过滤低波动期间的虚假信号

Method: MACD (Appel 1979, 经典动量指标)
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
    "fast_period": 12,
    "slow_period": 26,
    "signal_period": 9,
    "hist_threshold": 0.0,
    "atr_period": 14,
    "min_atr_mult": 0.5,
    "tp_atr_mult": 2.0,
    "sl_atr_mult": 1.0,
    "max_hold_bars": 40,
    "cooldown_bars": 3,
}


def _ema_series(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    alpha = 2.0 / (period + 1)
    result = [values[0]]
    for v in values[1:]:
        result.append(alpha * v + (1 - alpha) * result[-1])
    return result


@auto_register("macd_histogram")
class MACDHistogramStrategy(BaseStrategy):
    """MACD 柱状图策略 — 零轴交叉 + 动量衰减。"""

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

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        h = float(bar.get("high", 0))
        l = float(bar.get("low", 0))
        c = float(bar.get("close", 0))

        self._highs.append(h)
        self._lows.append(l)
        self._closes.append(c)
        self._bar_count += 1

        slow = self.get_param("slow_period", 26)
        if self._bar_count < slow + 15:
            return []

        atr = calc_atr(list(self._highs), list(self._lows), list(self._closes),
                       self.get_param("atr_period", 14))
        if atr is None or atr < 1e-10:
            return []

        close_list = list(self._closes)
        fast_ema = _ema_series(close_list, self.get_param("fast_period", 12))
        slow_ema = _ema_series(close_list, slow)
        macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]
        signal_line = _ema_series(macd_line, self.get_param("signal_period", 9))
        histogram = [m - s for m, s in zip(macd_line, signal_line)]

        if len(histogram) < 3:
            return []

        signals = []
        hist_now = histogram[-1]
        hist_prev = histogram[-2]
        hist_prev2 = histogram[-3]

        zero_cross_up = hist_prev <= 0 < hist_now
        zero_cross_down = hist_prev >= 0 > hist_now
        momentum_decay = (hist_now > 0 and hist_now < hist_prev < hist_prev2) or \
                         (hist_now < 0 and hist_now > hist_prev > hist_prev2)

        if self._position_side:
            self._hold_bars += 1
            max_hold = self.get_param("max_hold_bars", 40)
            tp_mult = self.get_param("tp_atr_mult", 2.0)
            sl_mult = self.get_param("sl_atr_mult", 1.0)

            pnl = (c - self._entry_price) / self._entry_price if self._position_side == "long" else (self._entry_price - c) / self._entry_price
            tp_hit = pnl >= tp_mult * atr / self._entry_price
            sl_hit = pnl <= -sl_mult * atr / self._entry_price

            should_exit = sl_hit or tp_hit or self._hold_bars >= max_hold
            if self._position_side == "long" and zero_cross_down:
                should_exit = True
            elif self._position_side == "short" and zero_cross_up:
                should_exit = True
            if momentum_decay and self._hold_bars >= 5:
                should_exit = True

            if should_exit:
                exit_type = SignalType.LONG_EXIT if self._position_side == "long" else SignalType.SHORT_EXIT
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_type, strength=0.8, price=c,
                    reason=f"macd_exit: hist={hist_now:.4f} hold={self._hold_bars}",
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
            if zero_cross_up:
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=min(abs(hist_now) / atr * 10, 1.0), price=c,
                    reason=f"macd_buy: hist_cross_zero={hist_now:.4f} macd={macd_line[-1]:.4f}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "long"
                self._entry_price = c
                self._hold_bars = 0

            elif zero_cross_down:
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=min(abs(hist_now) / atr * 10, 1.0), price=c,
                    reason=f"macd_sell: hist_cross_zero={hist_now:.4f} macd={macd_line[-1]:.4f}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "short"
                self._entry_price = c
                self._hold_bars = 0

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        return []
