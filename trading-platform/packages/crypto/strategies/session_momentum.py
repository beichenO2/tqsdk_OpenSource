"""Session Momentum Strategy — Time-of-Day Patterns.

Crypto markets have known session-based patterns:
- Asian session (00:00-08:00 UTC): lower volatility, ranging
- European session (08:00-16:00 UTC): increasing activity
- US session (14:00-22:00 UTC): highest volatility

Strategy: trade the directional bias established in the first
N bars of each session, capturing momentum from session transitions.

Also captures weekly patterns: Monday tends to set the direction.
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
    "session_open_bars": 3,
    "session_bias_threshold": 0.005,
    "ema_trend_period": 50,
    "atr_period": 14,
    "sl_atr_mult": 2.0,
    "tp_atr_mult": 3.5,
    "max_hold_bars": 24,
    "cooldown_bars": 6,
    "max_risk_per_trade": 0.02,
    "position_fraction": 0.35,
    "session_hours_4h": [0, 4, 8, 12, 16, 20],
}


@auto_register("session_momentum")
class SessionMomentumStrategy(BaseStrategy):
    """Trade session open directional bias."""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._c: dict[str, deque[float]] = {}
        self._h: dict[str, deque[float]] = {}
        self._l: dict[str, deque[float]] = {}
        self._o: dict[str, deque[float]] = {}
        self._ema: dict[str, float | None] = {}
        self._session_bar: dict[str, int] = {}
        self._session_open: dict[str, float] = {}
        self._hold: dict[str, int] = {}
        self._cd: dict[str, int] = {}
        self._entry: dict[str, float] = {}
        self._buf = 200

    def _init(self, s: str) -> None:
        if s not in self._c:
            self._c[s] = deque(maxlen=self._buf)
            self._h[s] = deque(maxlen=self._buf)
            self._l[s] = deque(maxlen=self._buf)
            self._o[s] = deque(maxlen=self._buf)

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._init(symbol)
        c, o, h, l = bar["close"], bar["open"], bar["high"], bar["low"]
        self._c[symbol].append(c)
        self._o[symbol].append(o)
        self._h[symbol].append(h)
        self._l[symbol].append(l)
        self._ema[symbol] = ema_update(self._ema.get(symbol), c, self.get_param("ema_trend_period"))

        bars_mod = len(self._c[symbol]) % 6
        if bars_mod == 0:
            self._session_bar[symbol] = 0
            self._session_open[symbol] = o

        self._session_bar[symbol] = self._session_bar.get(symbol, 0) + 1

        atr = calc_atr(self._h[symbol], self._l[symbol], self._c[symbol], self.get_param("atr_period"))
        if atr is None or atr <= 0:
            return []

        signals: list[Signal] = []
        pos = self.get_position(symbol)
        self._cd[symbol] = max(self._cd.get(symbol, 0) - 1, 0)

        session_bar = self._session_bar.get(symbol, 0)
        session_open = self._session_open.get(symbol, c)
        open_bars = self.get_param("session_open_bars")
        bias_threshold = self.get_param("session_bias_threshold")
        ema_val = self._ema.get(symbol, c)

        risk = self.get_param("max_risk_per_trade")
        sl_dist = atr * self.get_param("sl_atr_mult")
        pos_size = min(risk / (sl_dist / c) if sl_dist > 0 and c > 0 else 0.35, self.get_param("position_fraction"))

        if pos is None:
            if self._cd.get(symbol, 0) > 0:
                return signals

            if session_bar == open_bars and session_open > 0:
                session_return = (c - session_open) / session_open

                if session_return > bias_threshold and c > ema_val:
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.LONG_ENTRY, strength=0.75, price=c,
                        reason=f"SESSION LONG bias={session_return:.3f}",
                        metadata={"session_return": session_return, "session_bar": session_bar, "position_fraction": pos_size},
                    ))
                    self._hold[symbol] = 0
                    self._entry[symbol] = c

                elif session_return < -bias_threshold and c < ema_val:
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.SHORT_ENTRY, strength=0.75, price=c,
                        reason=f"SESSION SHORT bias={session_return:.3f}",
                        metadata={"session_return": session_return, "session_bar": session_bar, "position_fraction": pos_size},
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
                if c <= entry - atr * self.get_param("sl_atr_mult"):
                    ex, reason = True, "SL"
                elif c >= entry + atr * self.get_param("tp_atr_mult"):
                    ex, reason = True, "TP"
            elif pos.side.value == "sell":
                if c >= entry + atr * self.get_param("sl_atr_mult"):
                    ex, reason = True, "SL"
                elif c <= entry - atr * self.get_param("tp_atr_mult"):
                    ex, reason = True, "TP"

            if ex:
                et = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=et, strength=0.9, price=c,
                    reason=f"SESSION: {reason}",
                ))
                self._cd[symbol] = self.get_param("cooldown_bars")

        return signals

    async def generate_signals(self, md: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        for s in self.config.symbols:
            if s in md:
                out.extend(await self.on_bar(s, md[s]))
        return out
