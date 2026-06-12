"""Supertrend Strategy — ATR-based dynamic support/resistance.

Supertrend is one of the cleanest trend indicators:
  UpperBand = (high + low) / 2 + mult * ATR
  LowerBand = (high + low) / 2 - mult * ATR

When close > Supertrend → uptrend (Supertrend = LowerBand)
When close < Supertrend → downtrend (Supertrend = UpperBand)

Signal: enter on Supertrend flip. The beauty is in its simplicity —
it's essentially a volatility-adjusted trend line.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig
from strategy.indicators import calc_atr
from strategy.registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "atr_period": 10,
    "atr_multiplier": 3.0,
    "confirm_bars": 2,
    "trail_atr_mult": 2.0,
    "max_hold_bars": 120,
    "cooldown_bars": 4,
    "max_risk_per_trade": 0.02,
    "position_fraction": 0.4,
}


@auto_register("supertrend")
class SupertrendStrategy(BaseStrategy):
    """Enter on Supertrend direction flip."""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._c: dict[str, deque[float]] = {}
        self._h: dict[str, deque[float]] = {}
        self._l: dict[str, deque[float]] = {}
        self._st_value: dict[str, float] = {}
        self._st_direction: dict[str, int] = {}
        self._prev_upper: dict[str, float] = {}
        self._prev_lower: dict[str, float] = {}
        self._flip_count: dict[str, int] = {}
        self._hold: dict[str, int] = {}
        self._cd: dict[str, int] = {}
        self._entry: dict[str, float] = {}
        self._peak: dict[str, float] = {}
        self._buf = 200

    def _init(self, s: str) -> None:
        if s not in self._c:
            self._c[s] = deque(maxlen=self._buf)
            self._h[s] = deque(maxlen=self._buf)
            self._l[s] = deque(maxlen=self._buf)
            self._st_direction[s] = 0
            self._flip_count[s] = 0

    def _update_supertrend(self, s: str, h: float, l: float, c: float, atr: float) -> tuple[float, int, bool]:
        """Update Supertrend and return (value, direction, flipped)."""
        mult = self.get_param("atr_multiplier")
        mid = (h + l) / 2
        basic_upper = mid + mult * atr
        basic_lower = mid - mult * atr

        prev_upper = self._prev_upper.get(s, basic_upper)
        prev_lower = self._prev_lower.get(s, basic_lower)
        prev_dir = self._st_direction.get(s, 0)

        upper = min(basic_upper, prev_upper) if c > prev_upper else basic_upper
        lower = max(basic_lower, prev_lower) if c < prev_lower else basic_lower

        self._prev_upper[s] = upper
        self._prev_lower[s] = lower

        if prev_dir <= 0 and c > upper:
            direction = 1
        elif prev_dir >= 0 and c < lower:
            direction = -1
        else:
            direction = prev_dir

        flipped = direction != prev_dir and prev_dir != 0
        self._st_direction[s] = direction

        value = lower if direction == 1 else upper
        self._st_value[s] = value

        return value, direction, flipped

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._init(symbol)
        c, h, l = bar["close"], bar["high"], bar["low"]
        self._c[symbol].append(c)
        self._h[symbol].append(h)
        self._l[symbol].append(l)

        atr = calc_atr(self._h[symbol], self._l[symbol], self._c[symbol], self.get_param("atr_period"))
        if atr is None or atr <= 0:
            return []

        st_val, direction, flipped = self._update_supertrend(symbol, h, l, c, atr)

        if flipped:
            self._flip_count[symbol] = self.get_param("confirm_bars")

        flip_pending = self._flip_count.get(symbol, 0)
        if flip_pending > 0:
            self._flip_count[symbol] = flip_pending - 1

        signals: list[Signal] = []
        pos = self.get_position(symbol)
        self._cd[symbol] = max(self._cd.get(symbol, 0) - 1, 0)

        risk = self.get_param("max_risk_per_trade")
        sl_dist = atr * self.get_param("trail_atr_mult")
        pos_size = min(risk / (sl_dist / c) if sl_dist > 0 and c > 0 else 0.4, self.get_param("position_fraction"))

        if pos is None:
            if self._cd.get(symbol, 0) > 0:
                return signals

            if flip_pending == 1 and direction == 1:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY, strength=0.85, price=c,
                    reason=f"SUPERTREND flip LONG st={st_val:.0f}",
                    metadata={"supertrend": st_val, "direction": direction, "position_fraction": pos_size},
                ))
                self._hold[symbol] = 0
                self._entry[symbol] = c
                self._peak[symbol] = h

            elif flip_pending == 1 and direction == -1:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY, strength=0.85, price=c,
                    reason=f"SUPERTREND flip SHORT st={st_val:.0f}",
                    metadata={"supertrend": st_val, "direction": direction, "position_fraction": pos_size},
                ))
                self._hold[symbol] = 0
                self._entry[symbol] = c
                self._peak[symbol] = l

        else:
            self._hold[symbol] = self._hold.get(symbol, 0) + 1
            trail = atr * self.get_param("trail_atr_mult")
            ex = False
            reason = ""

            if self._hold[symbol] >= self.get_param("max_hold_bars"):
                ex, reason = True, "timeout"
            elif pos.side.value == "buy":
                self._peak[symbol] = max(self._peak.get(symbol, c), h)
                if l <= self._peak[symbol] - trail:
                    ex, reason = True, "trail SL"
                elif direction == -1:
                    ex, reason = True, "ST flip down"
            elif pos.side.value == "sell":
                self._peak[symbol] = min(self._peak.get(symbol, c), l)
                if h >= self._peak[symbol] + trail:
                    ex, reason = True, "trail SL"
                elif direction == 1:
                    ex, reason = True, "ST flip up"

            if ex:
                et = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=et, strength=0.9, price=c,
                    reason=f"SUPERTREND: {reason}",
                ))
                self._cd[symbol] = self.get_param("cooldown_bars")

        return signals

    async def generate_signals(self, md: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        for s in self.config.symbols:
            if s in md:
                out.extend(await self.on_bar(s, md[s]))
        return out
