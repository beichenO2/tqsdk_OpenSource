"""经典枢轴点策略 — 场内交易员的日内支撑/阻力系统。

理论：Pivot Point (经典场内交易, 1930s+ 芝加哥交易所)
  P = (H + L + C) / 3   (前一交易日的高/低/收)
  R1 = 2*P - L,  S1 = 2*P - H
  R2 = P + (H - L), S2 = P - (H - L)
  R3 = H + 2*(P - L), S3 = L - 2*(H - P)

日内交易逻辑：
  1. 开盘在 P 上方 → 多头偏向
  2. 价格突破 R1 → 趋势做多，目标 R2
  3. 价格跌破 S1 → 趋势做空，目标 S2
  4. 回踩 P → 均值回归信号

Method: 经典场内交易方法 (Floor Trader Pivot), 数十年实战验证
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
    "atr_period": 14,
    "breakout_confirm_atr": 0.2,
    "tp_atr_mult": 2.0,
    "sl_atr_mult": 1.0,
    "max_hold_bars": 30,
    "cooldown_bars": 3,
}


@auto_register("pivot_point")
class PivotPointStrategy(BaseStrategy):
    """经典枢轴点日内策略。"""

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

        self._prev_high: float = 0.0
        self._prev_low: float = 0.0
        self._prev_close: float = 0.0
        self._pivot: float = 0.0
        self._r1: float = 0.0
        self._r2: float = 0.0
        self._s1: float = 0.0
        self._s2: float = 0.0
        self._day_high: float = 0.0
        self._day_low: float = float("inf")
        self._last_session_bar: int = -1
        self._pivot_valid = False

    def _compute_pivots(self) -> None:
        if self._prev_high <= 0 or self._prev_low <= 0:
            return
        p = (self._prev_high + self._prev_low + self._prev_close) / 3
        self._pivot = p
        self._r1 = 2 * p - self._prev_low
        self._r2 = p + (self._prev_high - self._prev_low)
        self._s1 = 2 * p - self._prev_high
        self._s2 = p - (self._prev_high - self._prev_low)
        self._pivot_valid = True

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        h = float(bar.get("high", 0))
        l = float(bar.get("low", 0))
        c = float(bar.get("close", 0))

        self._highs.append(h)
        self._lows.append(l)
        self._closes.append(c)
        self._bar_count += 1

        self._day_high = max(self._day_high, h)
        self._day_low = min(self._day_low, l)

        dt = bar.get("datetime")
        if dt is not None:
            try:
                import pandas as pd
                ts = pd.Timestamp(dt)
                if ts.hour == 9 and ts.minute < 10 and self._bar_count - self._last_session_bar > 5:
                    self._prev_high = self._day_high
                    self._prev_low = self._day_low
                    self._prev_close = list(self._closes)[-2] if len(self._closes) >= 2 else c
                    self._compute_pivots()
                    self._day_high = h
                    self._day_low = l
                    self._last_session_bar = self._bar_count
            except Exception:
                pass

        if not self._pivot_valid or self._bar_count < 20:
            return []

        atr = calc_atr(list(self._highs), list(self._lows), list(self._closes),
                       self.get_param("atr_period", 14))
        if atr is None or atr < 1e-10:
            return []

        signals = []
        confirm_dist = self.get_param("breakout_confirm_atr", 0.2) * atr

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
                    reason=f"pivot_exit: hold={self._hold_bars}",
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
            if c > self._r1 + confirm_dist:
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=0.8, price=c,
                    reason=f"pivot_breakout_R1: c={c:.1f} R1={self._r1:.1f} P={self._pivot:.1f}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "long"
                self._entry_price = c
                self._hold_bars = 0

            elif c < self._s1 - confirm_dist:
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=0.8, price=c,
                    reason=f"pivot_breakout_S1: c={c:.1f} S1={self._s1:.1f} P={self._pivot:.1f}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "short"
                self._entry_price = c
                self._hold_bars = 0

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        return []
