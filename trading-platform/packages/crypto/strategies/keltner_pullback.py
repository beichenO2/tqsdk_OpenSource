"""Keltner Channel Pullback Strategy.

Uses Keltner Channels (EMA + ATR envelope) to identify pullback
entries in trending markets. Enter on pullback to middle band
(EMA) when trend is established (price above upper band recently).

Logic:
1. Trend confirmed when price traded above upper KC in last N bars
2. Pullback entry when price touches middle band (EMA)
3. Trail stop using lower KC band
4. Exit when price breaks below lower KC band

This is a "buy the dip in uptrend" / "sell the rally in downtrend"
strategy with dynamic volatility-based levels.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any

from strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig
from strategy.indicators import calc_atr, ema_update
from strategy.registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "kc_ema_period": 20,
    "kc_atr_mult": 2.0,
    "atr_period": 14,
    "trend_lookback": 10,
    "sl_atr_mult": 1.5,
    "max_hold_bars": 60,
    "cooldown_bars": 5,
    "max_risk_per_trade": 0.02,
    "position_fraction": 0.4,
}


@auto_register("keltner_pullback")
class KeltnerPullbackStrategy(BaseStrategy):
    """Pullback entries to Keltner Channel EMA in trending markets."""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._c: dict[str, deque[float]] = {}
        self._h: dict[str, deque[float]] = {}
        self._l: dict[str, deque[float]] = {}
        self._ema: dict[str, float | None] = {}
        self._above_upper: dict[str, deque[bool]] = {}
        self._below_lower: dict[str, deque[bool]] = {}
        self._hold: dict[str, int] = {}
        self._cd: dict[str, int] = {}
        self._entry: dict[str, float] = {}
        self._buf = 200

    def _init(self, s: str) -> None:
        if s not in self._c:
            self._c[s] = deque(maxlen=self._buf)
            self._h[s] = deque(maxlen=self._buf)
            self._l[s] = deque(maxlen=self._buf)
            self._above_upper[s] = deque(maxlen=50)
            self._below_lower[s] = deque(maxlen=50)

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._init(symbol)
        c, h, l = bar["close"], bar["high"], bar["low"]
        self._c[symbol].append(c)
        self._h[symbol].append(h)
        self._l[symbol].append(l)

        ema_p = self.get_param("kc_ema_period")
        self._ema[symbol] = ema_update(self._ema.get(symbol), c, ema_p)
        ema_val = self._ema[symbol]

        atr = calc_atr(self._h[symbol], self._l[symbol], self._c[symbol], self.get_param("atr_period"))
        if atr is None or atr <= 0 or ema_val is None:
            return []

        mult = self.get_param("kc_atr_mult")
        upper = ema_val + mult * atr
        lower = ema_val - mult * atr

        self._above_upper[symbol].append(h > upper)
        self._below_lower[symbol].append(l < lower)

        lookback = self.get_param("trend_lookback")
        recent_above = list(self._above_upper[symbol])
        recent_below = list(self._below_lower[symbol])

        was_above_recently = any(recent_above[-lookback:]) if len(recent_above) >= lookback else False
        was_below_recently = any(recent_below[-lookback:]) if len(recent_below) >= lookback else False

        at_ema = abs(c - ema_val) < atr * 0.3

        signals: list[Signal] = []
        pos = self.get_position(symbol)
        self._cd[symbol] = max(self._cd.get(symbol, 0) - 1, 0)

        risk = self.get_param("max_risk_per_trade")
        sl_dist = atr * self.get_param("sl_atr_mult")
        pos_size = min(risk / (sl_dist / c) if sl_dist > 0 and c > 0 else 0.4, self.get_param("position_fraction"))

        if pos is None:
            if self._cd.get(symbol, 0) > 0:
                return signals

            if was_above_recently and at_ema and c > lower:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY, strength=0.8, price=c,
                    reason=f"KC PULLBACK LONG to EMA={ema_val:.0f}",
                    metadata={"kc_upper": upper, "kc_lower": lower, "ema": ema_val, "position_fraction": pos_size},
                ))
                self._hold[symbol] = 0
                self._entry[symbol] = c

            elif was_below_recently and at_ema and c < upper:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY, strength=0.8, price=c,
                    reason=f"KC PULLBACK SHORT to EMA={ema_val:.0f}",
                    metadata={"kc_upper": upper, "kc_lower": lower, "ema": ema_val, "position_fraction": pos_size},
                ))
                self._hold[symbol] = 0
                self._entry[symbol] = c

        else:
            self._hold[symbol] = self._hold.get(symbol, 0) + 1
            entry = self._entry.get(symbol, c)
            ex = False
            reason = ""

            if self._hold[symbol] >= self.get_param("max_hold_bars"):
                ex, reason = True, "timeout"
            elif pos.side.value == "buy":
                if c < lower:
                    ex, reason = True, "below KC lower"
                elif c <= entry - atr * self.get_param("sl_atr_mult"):
                    ex, reason = True, "SL"
            elif pos.side.value == "sell":
                if c > upper:
                    ex, reason = True, "above KC upper"
                elif c >= entry + atr * self.get_param("sl_atr_mult"):
                    ex, reason = True, "SL"

            if ex:
                et = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=et, strength=0.9, price=c,
                    reason=f"KC: {reason}",
                ))
                self._cd[symbol] = self.get_param("cooldown_bars")

        return signals

    async def generate_signals(self, md: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        for s in self.config.symbols:
            if s in md:
                out.extend(await self.on_bar(s, md[s]))
        return out
