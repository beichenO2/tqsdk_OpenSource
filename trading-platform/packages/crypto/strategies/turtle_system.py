"""Turtle Trading System — Original Rules Adapted for Crypto.

The Turtle System (Richard Dennis, 1983) uses two systems:
System 1: 20-day breakout entry, 10-day exit (short-term)
System 2: 55-day breakout entry, 20-day exit (long-term)

Position sizing: risk 1% per trade, size = 1% / (ATR * multiplier)
Add to winners: up to 4 units, each 0.5 ATR apart

Adapted for crypto:
- 20/10 bar breakout (instead of days, since crypto is 24/7)
- ATR-based position sizing
- No pyramiding in simple version
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
    "entry_period": 20,
    "exit_period": 10,
    "atr_period": 20,
    "risk_per_trade": 0.01,
    "max_units": 1,
    "unit_spacing_atr": 0.5,
    "max_hold_bars": 120,
    "cooldown_bars": 3,
    "position_fraction": 0.5,
}


@auto_register("turtle_system")
class TurtleSystemStrategy(BaseStrategy):
    """Classic Turtle Trading System for crypto."""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._c: dict[str, deque[float]] = {}
        self._h: dict[str, deque[float]] = {}
        self._l: dict[str, deque[float]] = {}
        self._hold: dict[str, int] = {}
        self._cd: dict[str, int] = {}
        self._entry: dict[str, float] = {}
        self._buf = 200

    def _init(self, s: str) -> None:
        if s not in self._c:
            self._c[s] = deque(maxlen=self._buf)
            self._h[s] = deque(maxlen=self._buf)
            self._l[s] = deque(maxlen=self._buf)

    def _channel(self, s: str, period: int) -> tuple[float, float] | None:
        highs = list(self._h[s])
        lows = list(self._l[s])
        if len(highs) < period + 1:
            return None
        return max(highs[-(period+1):-1]), min(lows[-(period+1):-1])

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._init(symbol)
        c, h, l = bar["close"], bar["high"], bar["low"]
        self._c[symbol].append(c)
        self._h[symbol].append(h)
        self._l[symbol].append(l)

        atr = calc_atr(self._h[symbol], self._l[symbol], self._c[symbol], self.get_param("atr_period"))
        if atr is None or atr <= 0:
            return []

        entry_chan = self._channel(symbol, self.get_param("entry_period"))
        exit_chan = self._channel(symbol, self.get_param("exit_period"))
        if entry_chan is None or exit_chan is None:
            return []

        entry_high, entry_low = entry_chan
        exit_high, exit_low = exit_chan

        risk = self.get_param("risk_per_trade")
        pos_size = min(risk / (atr / c) if atr > 0 and c > 0 else 0.5, self.get_param("position_fraction"))

        signals: list[Signal] = []
        pos = self.get_position(symbol)
        self._cd[symbol] = max(self._cd.get(symbol, 0) - 1, 0)

        if pos is None:
            if self._cd.get(symbol, 0) > 0:
                return signals

            if c > entry_high:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY, strength=0.85, price=c,
                    reason=f"TURTLE LONG breakout above {entry_high:.0f}",
                    metadata={"entry_high": entry_high, "atr": atr, "position_fraction": pos_size},
                ))
                self._hold[symbol] = 0
                self._entry[symbol] = c

            elif c < entry_low:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY, strength=0.85, price=c,
                    reason=f"TURTLE SHORT breakdown below {entry_low:.0f}",
                    metadata={"entry_low": entry_low, "atr": atr, "position_fraction": pos_size},
                ))
                self._hold[symbol] = 0
                self._entry[symbol] = c

        else:
            self._hold[symbol] = self._hold.get(symbol, 0) + 1
            ex = False
            reason = ""

            if self._hold[symbol] >= self.get_param("max_hold_bars"):
                ex, reason = True, "timeout"
            elif pos.side.value == "buy":
                if c < exit_low:
                    ex, reason = True, f"exit below {exit_low:.0f}"
            elif pos.side.value == "sell":
                if c > exit_high:
                    ex, reason = True, f"exit above {exit_high:.0f}"

            if ex:
                et = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=et, strength=0.9, price=c,
                    reason=f"TURTLE: {reason}",
                ))
                self._cd[symbol] = self.get_param("cooldown_bars")

        return signals

    async def generate_signals(self, md: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        for s in self.config.symbols:
            if s in md:
                out.extend(await self.on_bar(s, md[s]))
        return out
