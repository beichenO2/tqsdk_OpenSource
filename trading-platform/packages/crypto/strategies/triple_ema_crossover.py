"""Triple EMA Crossover Strategy.

Uses three EMAs (fast/medium/slow) in alignment for trend confirmation.
Only trades when all three align: fast > medium > slow (bull) or reverse.

The "rainbow" of EMAs provides clearer trend signals than dual EMA
because it requires stronger consensus before entry.

Enhanced with:
- Volume confirmation on crossover
- ATR trailing stop for ride-the-trend exits
- Regime-aware position sizing
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
    "ema_fast": 8,
    "ema_medium": 21,
    "ema_slow": 55,
    "volume_confirm_ratio": 1.3,
    "atr_period": 14,
    "trail_atr_mult": 2.5,
    "max_hold_bars": 96,
    "cooldown_bars": 5,
    "max_risk_per_trade": 0.02,
    "position_fraction": 0.4,
}


@auto_register("triple_ema_crossover")
class TripleEMACrossoverStrategy(BaseStrategy):
    """Triple EMA alignment for high-conviction trend entries."""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._c: dict[str, deque[float]] = {}
        self._h: dict[str, deque[float]] = {}
        self._l: dict[str, deque[float]] = {}
        self._v: dict[str, deque[float]] = {}
        self._ef: dict[str, float | None] = {}
        self._em: dict[str, float | None] = {}
        self._es: dict[str, float | None] = {}
        self._prev_aligned: dict[str, str] = {}
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
            self._prev_aligned[s] = "none"

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._init(symbol)
        c, h, l = bar["close"], bar["high"], bar["low"]
        vol = bar.get("volume", 0.0)
        self._c[symbol].append(c)
        self._h[symbol].append(h)
        self._l[symbol].append(l)
        self._v[symbol].append(vol)

        self._ef[symbol] = ema_update(self._ef.get(symbol), c, self.get_param("ema_fast"))
        self._em[symbol] = ema_update(self._em.get(symbol), c, self.get_param("ema_medium"))
        self._es[symbol] = ema_update(self._es.get(symbol), c, self.get_param("ema_slow"))

        ef = self._ef[symbol]
        em = self._em[symbol]
        es = self._es[symbol]
        if ef is None or em is None or es is None:
            return []

        atr = calc_atr(self._h[symbol], self._l[symbol], self._c[symbol], self.get_param("atr_period"))
        if atr is None or atr <= 0:
            return []

        current_align = "none"
        if ef > em > es:
            current_align = "bull"
        elif ef < em < es:
            current_align = "bear"

        prev = self._prev_aligned.get(symbol, "none")
        just_aligned = current_align != "none" and current_align != prev
        self._prev_aligned[symbol] = current_align

        vols = list(self._v[symbol])
        vol_ok = True
        if len(vols) >= 20:
            avg_vol = sum(vols[-20:]) / 20
            vol_ok = vol >= avg_vol * self.get_param("volume_confirm_ratio") if avg_vol > 0 else True

        signals: list[Signal] = []
        pos = self.get_position(symbol)
        self._cd[symbol] = max(self._cd.get(symbol, 0) - 1, 0)

        risk = self.get_param("max_risk_per_trade")
        sl_dist = atr * self.get_param("trail_atr_mult")
        pos_size = min(risk / (sl_dist / c) if sl_dist > 0 and c > 0 else 0.4, self.get_param("position_fraction"))

        if pos is None:
            if self._cd.get(symbol, 0) > 0:
                return signals

            if just_aligned and current_align == "bull" and vol_ok:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY, strength=0.85, price=c,
                    reason=f"3EMA LONG aligned f={ef:.0f}>m={em:.0f}>s={es:.0f}",
                    metadata={"ema_fast": ef, "ema_medium": em, "ema_slow": es, "position_fraction": pos_size},
                ))
                self._hold[symbol] = 0
                self._peak[symbol] = h

            elif just_aligned and current_align == "bear" and vol_ok:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY, strength=0.85, price=c,
                    reason=f"3EMA SHORT aligned f={ef:.0f}<m={em:.0f}<s={es:.0f}",
                    metadata={"ema_fast": ef, "ema_medium": em, "ema_slow": es, "position_fraction": pos_size},
                ))
                self._hold[symbol] = 0
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
                elif current_align == "bear":
                    ex, reason = True, "EMA reversed"
            elif pos.side.value == "sell":
                self._peak[symbol] = min(self._peak.get(symbol, c), l)
                if h >= self._peak[symbol] + trail:
                    ex, reason = True, "trail SL"
                elif current_align == "bull":
                    ex, reason = True, "EMA reversed"

            if ex:
                et = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=et, strength=0.9, price=c,
                    reason=f"3EMA: {reason}",
                ))
                self._cd[symbol] = self.get_param("cooldown_bars")

        return signals

    async def generate_signals(self, md: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        for s in self.config.symbols:
            if s in md:
                out.extend(await self.on_bar(s, md[s]))
        return out
