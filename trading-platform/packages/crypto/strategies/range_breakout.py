"""Range Breakout Strategy — Consolidation → Expansion.

Detects consolidation ranges (tight price action) and trades the
breakout when price escapes the range with momentum confirmation.

Range detection: N consecutive bars within ATR-defined range
Breakout: close beyond range high/low with volume confirmation

Simple but effective — captures the most fundamental pattern
in markets: consolidation followed by expansion.
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
    "range_period": 15,
    "range_atr_width": 2.0,
    "min_range_bars": 5,
    "breakout_confirm_pct": 0.003,
    "volume_mult": 1.5,
    "ema_trend_period": 50,
    "atr_period": 14,
    "trail_atr_mult": 2.5,
    "max_hold_bars": 72,
    "cooldown_bars": 5,
    "max_risk_per_trade": 0.02,
    "position_fraction": 0.4,
}


@auto_register("range_breakout")
class RangeBreakoutStrategy(BaseStrategy):
    """Consolidation detection + breakout entry."""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._c: dict[str, deque[float]] = {}
        self._h: dict[str, deque[float]] = {}
        self._l: dict[str, deque[float]] = {}
        self._v: dict[str, deque[float]] = {}
        self._ema: dict[str, float | None] = {}
        self._range_bars: dict[str, int] = {}
        self._range_high: dict[str, float] = {}
        self._range_low: dict[str, float] = {}
        self._hold: dict[str, int] = {}
        self._cd: dict[str, int] = {}
        self._peak: dict[str, float] = {}
        self._buf = 200

    def _init(self, s: str) -> None:
        if s not in self._c:
            self._c[s] = deque(maxlen=self._buf)
            self._h[s] = deque(maxlen=self._buf)
            self._l[s] = deque(maxlen=self._buf)
            self._v[s] = deque(maxlen=self._buf)

    def _update_range(self, s: str, atr: float) -> None:
        highs = list(self._h[s])
        lows = list(self._l[s])
        p = self.get_param("range_period")
        if len(highs) < p:
            return

        rng_h = max(highs[-p:])
        rng_l = min(lows[-p:])
        width = rng_h - rng_l
        max_width = atr * self.get_param("range_atr_width")

        if width <= max_width:
            self._range_bars[s] = self._range_bars.get(s, 0) + 1
            self._range_high[s] = rng_h
            self._range_low[s] = rng_l
        else:
            self._range_bars[s] = 0

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._init(symbol)
        c, h, l = bar["close"], bar["high"], bar["low"]
        vol = bar.get("volume", 0.0)

        self._c[symbol].append(c)
        self._h[symbol].append(h)
        self._l[symbol].append(l)
        self._v[symbol].append(vol)
        self._ema[symbol] = ema_update(self._ema.get(symbol), c, self.get_param("ema_trend_period"))

        atr = calc_atr(self._h[symbol], self._l[symbol], self._c[symbol], self.get_param("atr_period"))
        if atr is None or atr <= 0:
            return []

        self._update_range(symbol, atr)
        range_bars = self._range_bars.get(symbol, 0)
        min_bars = self.get_param("min_range_bars")

        signals: list[Signal] = []
        pos = self.get_position(symbol)
        self._cd[symbol] = max(self._cd.get(symbol, 0) - 1, 0)

        risk = self.get_param("max_risk_per_trade")
        sl_dist = atr * self.get_param("trail_atr_mult")
        pos_size = min(risk / (sl_dist / c) if sl_dist > 0 and c > 0 else 0.4, self.get_param("position_fraction"))

        if pos is None:
            if self._cd.get(symbol, 0) > 0:
                return signals

            rng_h = self._range_high.get(symbol, 0)
            rng_l = self._range_low.get(symbol, 0)

            if range_bars < min_bars or rng_h <= rng_l:
                return signals

            confirm = self.get_param("breakout_confirm_pct")
            vols = list(self._v[symbol])
            vol_ok = True
            if len(vols) >= 20:
                avg_vol = sum(vols[-20:]) / 20
                vol_ok = vol >= avg_vol * self.get_param("volume_mult") if avg_vol > 0 else True

            ema_val = self._ema.get(symbol, c)

            if c > rng_h * (1 + confirm) and vol_ok and c > ema_val:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY, strength=0.85, price=c,
                    reason=f"RANGE BREAKOUT UP after {range_bars} bars, range=[{rng_l:.0f}-{rng_h:.0f}]",
                    metadata={"range_bars": range_bars, "range_high": rng_h, "range_low": rng_l, "position_fraction": pos_size},
                ))
                self._hold[symbol] = 0
                self._peak[symbol] = h
                self._range_bars[symbol] = 0

            elif c < rng_l * (1 - confirm) and vol_ok and c < ema_val:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY, strength=0.85, price=c,
                    reason=f"RANGE BREAKOUT DOWN after {range_bars} bars, range=[{rng_l:.0f}-{rng_h:.0f}]",
                    metadata={"range_bars": range_bars, "range_high": rng_h, "range_low": rng_l, "position_fraction": pos_size},
                ))
                self._hold[symbol] = 0
                self._peak[symbol] = l
                self._range_bars[symbol] = 0

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
            elif pos.side.value == "sell":
                self._peak[symbol] = min(self._peak.get(symbol, c), l)
                if h >= self._peak[symbol] + trail:
                    ex, reason = True, "trail SL"

            if ex:
                et = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=et, strength=0.9, price=c,
                    reason=f"RANGE: {reason}",
                ))
                self._cd[symbol] = self.get_param("cooldown_bars")

        return signals

    async def generate_signals(self, md: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        for s in self.config.symbols:
            if s in md:
                out.extend(await self.on_bar(s, md[s]))
        return out
