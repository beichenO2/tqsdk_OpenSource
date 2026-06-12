"""Fair Value Gap (FVG) Strategy — Smart Money Concepts.

Fair Value Gaps are imbalances in price caused by aggressive institutional
moves. They appear as gaps between candle wicks — areas where one side
(buyers or sellers) dominated completely with no opposing orders filled.

Detection:
  Bullish FVG: bar[i].high < bar[i+2].low (gap between wick of bar i and bar i+2)
  Bearish FVG: bar[i].low > bar[i+2].high

Strategy logic:
1. Detect FVGs as they form
2. Wait for price to retrace INTO the gap (fill attempt)
3. Enter when price touches the FVG zone
4. Target: opposite end of the FVG
5. Stop: beyond the FVG zone

This is a pullback entry strategy — similar to how institutional
traders enter on retracements to "fair value" zones.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import math

from strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig
from strategy.indicators import calc_atr
from strategy.registry import auto_register
from .regime_detector import MarketRegimeDetector

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "min_fvg_atr_ratio": 0.5,
    "max_fvg_age_bars": 48,
    "max_active_fvgs": 5,
    "atr_period": 14,
    "sl_atr_mult": 1.5,
    "max_hold_bars": 36,
    "cooldown_bars": 4,
    "max_risk_per_trade": 0.02,
    "position_fraction": 0.4,
    "use_regime_filter": True,
    "blocked_regimes": ["high_volatility"],
}


class FVGZone:
    """Represents a Fair Value Gap."""
    __slots__ = ("upper", "lower", "direction", "bar_idx", "filled")

    def __init__(self, upper: float, lower: float, direction: str, bar_idx: int) -> None:
        self.upper = upper
        self.lower = lower
        self.direction = direction
        self.bar_idx = bar_idx
        self.filled = False

    @property
    def midpoint(self) -> float:
        return (self.upper + self.lower) / 2

    @property
    def size(self) -> float:
        return self.upper - self.lower


@auto_register("smart_money_fvg")
class SmartMoneyFVGStrategy(BaseStrategy):
    """Trade retracements into Fair Value Gaps."""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._c: dict[str, deque[float]] = {}
        self._h: dict[str, deque[float]] = {}
        self._l: dict[str, deque[float]] = {}
        self._fvgs: dict[str, list[FVGZone]] = {}
        self._bar_idx: dict[str, int] = {}
        self._hold: dict[str, int] = {}
        self._cd: dict[str, int] = {}
        self._entry: dict[str, float] = {}
        self._target: dict[str, float] = {}
        self._regime: dict[str, MarketRegimeDetector] = {}
        self._returns: dict[str, deque[float]] = {}
        self._buf = 200

    def _init(self, s: str) -> None:
        if s not in self._c:
            self._c[s] = deque(maxlen=self._buf)
            self._h[s] = deque(maxlen=self._buf)
            self._l[s] = deque(maxlen=self._buf)
            self._fvgs[s] = []
            self._bar_idx[s] = 0
            self._regime[s] = MarketRegimeDetector()
            self._returns[s] = deque(maxlen=100)

    def _detect_fvg(self, s: str, atr: float) -> None:
        """Check if a new FVG formed (need at least 3 bars)."""
        highs = list(self._h[s])
        lows = list(self._l[s])
        if len(highs) < 3:
            return

        bar_i = highs[-3], lows[-3]
        bar_k = highs[-1], lows[-1]
        idx = self._bar_idx[s]
        min_size = atr * self.get_param("min_fvg_atr_ratio")

        if bar_i[0] < bar_k[1]:
            gap_size = bar_k[1] - bar_i[0]
            if gap_size >= min_size:
                fvg = FVGZone(bar_k[1], bar_i[0], "bullish", idx)
                self._fvgs[s].append(fvg)

        if bar_i[1] > bar_k[0]:
            gap_size = bar_i[1] - bar_k[0]
            if gap_size >= min_size:
                fvg = FVGZone(bar_i[1], bar_k[0], "bearish", idx)
                self._fvgs[s].append(fvg)

        max_active = self.get_param("max_active_fvgs")
        max_age = self.get_param("max_fvg_age_bars")
        self._fvgs[s] = [
            f for f in self._fvgs[s]
            if not f.filled and (idx - f.bar_idx) <= max_age
        ][-max_active:]

    def _compute_position_size(self, symbol: str, atr: float, price: float) -> float:
        risk = self.get_param("max_risk_per_trade")
        sl_dist = atr * self.get_param("sl_atr_mult")
        if sl_dist <= 0 or price <= 0:
            return self.get_param("position_fraction")
        size = risk / (sl_dist / price)
        return min(size, self.get_param("position_fraction"))

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._init(symbol)
        c, h, l = bar["close"], bar["high"], bar["low"]

        if self._c[symbol]:
            prev = self._c[symbol][-1]
            if prev > 0:
                self._returns[symbol].append((c - prev) / prev)

        self._c[symbol].append(c)
        self._h[symbol].append(h)
        self._l[symbol].append(l)
        self._bar_idx[symbol] += 1
        self._regime[symbol].update(h, l, c)

        atr = calc_atr(self._h[symbol], self._l[symbol], self._c[symbol], self.get_param("atr_period"))
        if atr is None or atr <= 0:
            return []

        self._detect_fvg(symbol, atr)

        signals: list[Signal] = []
        pos = self.get_position(symbol)
        self._cd[symbol] = max(self._cd.get(symbol, 0) - 1, 0)

        if pos is None:
            if self._cd.get(symbol, 0) > 0:
                return signals

            if self.get_param("use_regime_filter"):
                regime = self._regime[symbol].current_regime
                if regime.value in self.get_param("blocked_regimes"):
                    return signals

            pos_size = self._compute_position_size(symbol, atr, c)

            for fvg in self._fvgs.get(symbol, []):
                if fvg.filled:
                    continue

                if fvg.direction == "bullish" and l <= fvg.upper and c > fvg.lower:
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.LONG_ENTRY, strength=0.75, price=c,
                        reason=f"FVG LONG retrace into gap [{fvg.lower:.0f}-{fvg.upper:.0f}]",
                        metadata={"fvg_lower": fvg.lower, "fvg_upper": fvg.upper, "fvg_dir": "bullish", "position_fraction": pos_size},
                    ))
                    self._hold[symbol] = 0
                    self._entry[symbol] = c
                    self._target[symbol] = fvg.upper + fvg.size
                    fvg.filled = True
                    break

                elif fvg.direction == "bearish" and h >= fvg.lower and c < fvg.upper:
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.SHORT_ENTRY, strength=0.75, price=c,
                        reason=f"FVG SHORT retrace into gap [{fvg.lower:.0f}-{fvg.upper:.0f}]",
                        metadata={"fvg_lower": fvg.lower, "fvg_upper": fvg.upper, "fvg_dir": "bearish", "position_fraction": pos_size},
                    ))
                    self._hold[symbol] = 0
                    self._entry[symbol] = c
                    self._target[symbol] = fvg.lower - fvg.size
                    fvg.filled = True
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
                    ex, reason = True, "TP (FVG extension)"
                elif c <= entry - atr * self.get_param("sl_atr_mult"):
                    ex, reason = True, "SL"
            elif pos.side.value == "sell":
                if c <= target:
                    ex, reason = True, "TP (FVG extension)"
                elif c >= entry + atr * self.get_param("sl_atr_mult"):
                    ex, reason = True, "SL"

            if ex:
                et = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=et, strength=0.9, price=c,
                    reason=f"FVG: {reason}",
                ))
                self._cd[symbol] = self.get_param("cooldown_bars")

        return signals

    async def generate_signals(self, md: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        for s in self.config.symbols:
            if s in md:
                out.extend(await self.on_bar(s, md[s]))
        return out
