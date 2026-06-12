"""ATR Channel Breakout Strategy.

Uses ATR-based channels around EMA for dynamic breakout detection.
Similar to Keltner but with more aggressive breakout logic.

Entry: price closes beyond ATR channel (volatility expansion)
Exit: price returns inside channel or trailing stop hit

Combined with momentum filter (ROC > 0 for longs, < 0 for shorts).
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
    "ema_period": 20,
    "atr_period": 14,
    "channel_mult": 2.0,
    "roc_period": 10,
    "trail_atr_mult": 2.0,
    "max_hold_bars": 72,
    "cooldown_bars": 4,
    "max_risk_per_trade": 0.02,
    "position_fraction": 0.4,
}


@auto_register("atr_channel_breakout")
class ATRChannelBreakoutStrategy(BaseStrategy):
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
        self._peak: dict[str, float] = {}
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
        self._ema[symbol] = ema_update(self._ema.get(symbol), c, self.get_param("ema_period"))

        atr = calc_atr(self._h[symbol], self._l[symbol], self._c[symbol], self.get_param("atr_period"))
        ema_val = self._ema[symbol]
        if atr is None or atr <= 0 or ema_val is None:
            return []

        upper = ema_val + self.get_param("channel_mult") * atr
        lower = ema_val - self.get_param("channel_mult") * atr

        closes = list(self._c[symbol])
        roc_p = self.get_param("roc_period")
        roc = 0.0
        if len(closes) > roc_p:
            roc = (closes[-1] - closes[-roc_p-1]) / closes[-roc_p-1] if closes[-roc_p-1] > 0 else 0

        signals: list[Signal] = []
        pos = self.get_position(symbol)
        self._cd[symbol] = max(self._cd.get(symbol, 0) - 1, 0)
        risk = self.get_param("max_risk_per_trade")
        sl_dist = atr * self.get_param("trail_atr_mult")
        pos_size = min(risk / (sl_dist / c) if sl_dist > 0 and c > 0 else 0.4, self.get_param("position_fraction"))

        if pos is None:
            if self._cd.get(symbol, 0) > 0:
                return signals
            if c > upper and roc > 0:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY, strength=0.85, price=c,
                    reason=f"ATR CHAN LONG above {upper:.0f}",
                    metadata={"upper": upper, "lower": lower, "roc": roc, "position_fraction": pos_size},
                ))
                self._hold[symbol] = 0
                self._peak[symbol] = h
            elif c < lower and roc < 0:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY, strength=0.85, price=c,
                    reason=f"ATR CHAN SHORT below {lower:.0f}",
                    metadata={"upper": upper, "lower": lower, "roc": roc, "position_fraction": pos_size},
                ))
                self._hold[symbol] = 0
                self._peak[symbol] = l
        else:
            self._hold[symbol] = self._hold.get(symbol, 0) + 1
            trail = atr * self.get_param("trail_atr_mult")
            ex, reason = False, ""
            if self._hold[symbol] >= self.get_param("max_hold_bars"):
                ex, reason = True, "timeout"
            elif pos.side.value == "buy":
                self._peak[symbol] = max(self._peak.get(symbol, c), h)
                if l <= self._peak[symbol] - trail:
                    ex, reason = True, "trail SL"
                elif c < ema_val:
                    ex, reason = True, "below EMA"
            elif pos.side.value == "sell":
                self._peak[symbol] = min(self._peak.get(symbol, c), l)
                if h >= self._peak[symbol] + trail:
                    ex, reason = True, "trail SL"
                elif c > ema_val:
                    ex, reason = True, "above EMA"
            if ex:
                et = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=et, strength=0.9, price=c, reason=f"ATR CHAN: {reason}",
                ))
                self._cd[symbol] = self.get_param("cooldown_bars")
        return signals

    async def generate_signals(self, md: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        for s in self.config.symbols:
            if s in md:
                out.extend(await self.on_bar(s, md[s]))
        return out
