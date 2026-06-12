"""Fibonacci Retracement Pullback Strategy.

Automatically detects swing highs/lows and computes Fibonacci levels.
Enters at key retracement levels (38.2%, 50%, 61.8%) during pullbacks
in trending markets.

The "Golden Pocket" (61.8%-65%) is the highest probability zone for
trend continuation entries.

Logic:
1. Detect swing high and swing low
2. Compute Fib levels between them
3. Wait for price to pull back to a Fib level
4. Confirm with trend direction (EMA filter)
5. Enter at Fib level, target at Fib extension
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig
from strategy.indicators import calc_atr, ema_update
from strategy.registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "swing_lookback": 5,
    "swing_min_range_atr": 3.0,
    "fib_entry_levels": [0.618, 0.5, 0.382],
    "fib_tolerance": 0.005,
    "ema_trend_period": 50,
    "atr_period": 14,
    "sl_atr_mult": 1.5,
    "tp_fib_extension": 1.618,
    "max_hold_bars": 60,
    "cooldown_bars": 6,
    "max_risk_per_trade": 0.02,
    "position_fraction": 0.4,
}


def _find_last_swing(highs: list[float], lows: list[float], lookback: int) -> tuple[float, float, str] | None:
    """Find last significant swing high and low. Return (swing_high, swing_low, direction)."""
    if len(highs) < lookback * 3:
        return None

    high_idx = None
    low_idx = None

    for i in range(lookback, len(highs) - lookback):
        if all(highs[i] >= highs[i-j] for j in range(1, lookback+1)) and \
           all(highs[i] >= highs[i+j] for j in range(1, min(lookback+1, len(highs)-i))):
            high_idx = i

    for i in range(lookback, len(lows) - lookback):
        if all(lows[i] <= lows[i-j] for j in range(1, lookback+1)) and \
           all(lows[i] <= lows[i+j] for j in range(1, min(lookback+1, len(lows)-i))):
            low_idx = i

    if high_idx is None or low_idx is None:
        return None

    sh = highs[high_idx]
    sl = lows[low_idx]

    if high_idx > low_idx:
        return sh, sl, "upswing"
    else:
        return sh, sl, "downswing"


def _fib_level(sh: float, sl: float, ratio: float, direction: str) -> float:
    """Compute Fibonacci retracement level."""
    rng = sh - sl
    if direction == "upswing":
        return sh - rng * ratio
    else:
        return sl + rng * ratio


@auto_register("fibonacci_pullback")
class FibonacciPullbackStrategy(BaseStrategy):
    """Enter at Fibonacci retracement levels during pullbacks."""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._c: dict[str, deque[float]] = {}
        self._h: dict[str, deque[float]] = {}
        self._l: dict[str, deque[float]] = {}
        self._ema: dict[str, float | None] = {}
        self._hold: dict[str, int] = {}
        self._cd: dict[str, int] = {}
        self._entry: dict[str, float] = {}
        self._target: dict[str, float] = {}
        self._buf = 200

    def _init(self, s: str) -> None:
        if s not in self._c:
            self._c[s] = deque(maxlen=self._buf)
            self._h[s] = deque(maxlen=self._buf)
            self._l[s] = deque(maxlen=self._buf)

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._init(symbol)
        c, h, l = bar["close"], bar["high"], bar["low"]
        self._c[symbol].append(c)
        self._h[symbol].append(h)
        self._l[symbol].append(l)
        self._ema[symbol] = ema_update(self._ema.get(symbol), c, self.get_param("ema_trend_period"))

        atr = calc_atr(self._h[symbol], self._l[symbol], self._c[symbol], self.get_param("atr_period"))
        if atr is None or atr <= 0:
            return []

        highs = list(self._h[symbol])
        lows = list(self._l[symbol])
        swing = _find_last_swing(highs, lows, self.get_param("swing_lookback"))
        if swing is None:
            return []

        sh, sl, direction = swing
        swing_range = sh - sl
        if swing_range < atr * self.get_param("swing_min_range_atr"):
            return []

        ema_val = self._ema[symbol]
        if ema_val is None:
            return []

        signals: list[Signal] = []
        pos = self.get_position(symbol)
        self._cd[symbol] = max(self._cd.get(symbol, 0) - 1, 0)
        tol = self.get_param("fib_tolerance")

        risk = self.get_param("max_risk_per_trade")
        sl_dist = atr * self.get_param("sl_atr_mult")
        pos_size = min(risk / (sl_dist / c) if sl_dist > 0 and c > 0 else 0.4, self.get_param("position_fraction"))

        if pos is None:
            if self._cd.get(symbol, 0) > 0:
                return signals

            for fib_ratio in self.get_param("fib_entry_levels"):
                fib_price = _fib_level(sh, sl, fib_ratio, direction)

                if direction == "upswing" and c > ema_val:
                    if abs(c - fib_price) / c < tol:
                        ext = _fib_level(sh, sl, -self.get_param("tp_fib_extension") + 1, direction)
                        signals.append(Signal(
                            strategy_id=self.strategy_id, symbol=symbol,
                            signal_type=SignalType.LONG_ENTRY, strength=0.8, price=c,
                            reason=f"FIB LONG at {fib_ratio:.1%} retrace={fib_price:.0f}",
                            metadata={"fib_ratio": fib_ratio, "swing_high": sh, "swing_low": sl, "position_fraction": pos_size},
                        ))
                        self._hold[symbol] = 0
                        self._entry[symbol] = c
                        self._target[symbol] = ext
                        break

                elif direction == "downswing" and c < ema_val:
                    if abs(c - fib_price) / c < tol:
                        ext = _fib_level(sh, sl, -self.get_param("tp_fib_extension") + 1, direction)
                        signals.append(Signal(
                            strategy_id=self.strategy_id, symbol=symbol,
                            signal_type=SignalType.SHORT_ENTRY, strength=0.8, price=c,
                            reason=f"FIB SHORT at {fib_ratio:.1%} retrace={fib_price:.0f}",
                            metadata={"fib_ratio": fib_ratio, "swing_high": sh, "swing_low": sl, "position_fraction": pos_size},
                        ))
                        self._hold[symbol] = 0
                        self._entry[symbol] = c
                        self._target[symbol] = ext
                        break

        else:
            self._hold[symbol] = self._hold.get(symbol, 0) + 1
            entry = self._entry.get(symbol, c)
            target = self._target.get(symbol, c)
            ex = False
            reason = ""

            if self._hold[symbol] >= self.get_param("max_hold_bars"):
                ex, reason = True, "timeout"
            elif pos.side.value == "buy":
                if c >= target:
                    ex, reason = True, f"TP at Fib ext {target:.0f}"
                elif c <= entry - atr * self.get_param("sl_atr_mult"):
                    ex, reason = True, "SL"
            elif pos.side.value == "sell":
                if c <= target:
                    ex, reason = True, f"TP at Fib ext {target:.0f}"
                elif c >= entry + atr * self.get_param("sl_atr_mult"):
                    ex, reason = True, "SL"

            if ex:
                et = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=et, strength=0.9, price=c,
                    reason=f"FIB: {reason}",
                ))
                self._cd[symbol] = self.get_param("cooldown_bars")

        return signals

    async def generate_signals(self, md: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        for s in self.config.symbols:
            if s in md:
                out.extend(await self.on_bar(s, md[s]))
        return out
