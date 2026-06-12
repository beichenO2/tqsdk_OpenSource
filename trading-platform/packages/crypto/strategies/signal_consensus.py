"""Signal Consensus Strategy — Meta-Strategy Aggregator.

Instead of trading individual strategy signals, this meta-strategy
runs multiple sub-strategies internally and only trades when
a minimum number agree on direction.

Sub-strategies polled:
1. Supertrend direction
2. Triple EMA alignment
3. RSI extreme zone
4. MACD histogram direction
5. Price vs VWAP

When N out of 5 agree → high conviction entry.
This is the "wisdom of crowds" applied to trading signals.
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
    "ema_fast": 8,
    "ema_medium": 21,
    "ema_slow": 55,
    "rsi_period": 14,
    "rsi_bull_threshold": 55,
    "rsi_bear_threshold": 45,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "vwap_period": 20,
    "supertrend_atr_mult": 3.0,
    "atr_period": 14,
    "min_consensus": 4,
    "trail_atr_mult": 2.5,
    "max_hold_bars": 72,
    "cooldown_bars": 5,
    "max_risk_per_trade": 0.02,
    "position_fraction": 0.5,
}


@auto_register("signal_consensus")
class SignalConsensusStrategy(BaseStrategy):
    """Trade only when multiple indicators agree (consensus)."""

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
        self._rsi_ag: dict[str, float] = {}
        self._rsi_al: dict[str, float] = {}
        self._rsi: dict[str, float] = {}
        self._bc: dict[str, int] = {}
        self._st_dir: dict[str, int] = {}
        self._st_upper: dict[str, float] = {}
        self._st_lower: dict[str, float] = {}
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
            self._bc[s] = 0
            self._rsi[s] = 50.0
            self._st_dir[s] = 0

    def _update_rsi(self, s: str, c: float) -> float:
        p = self.get_param("rsi_period")
        self._bc[s] = self._bc.get(s, 0) + 1
        if self._bc[s] < 2 or not self._c[s]:
            return 50.0
        prev = self._c[s][-1]
        d = c - prev
        g, lo = max(d, 0), max(-d, 0)
        if self._bc[s] <= p + 1:
            self._rsi_ag[s] = self._rsi_ag.get(s, 0) + g
            self._rsi_al[s] = self._rsi_al.get(s, 0) + lo
            if self._bc[s] == p + 1:
                self._rsi_ag[s] /= p
                self._rsi_al[s] /= p
        else:
            self._rsi_ag[s] = (self._rsi_ag.get(s, 0) * (p-1) + g) / p
            self._rsi_al[s] = (self._rsi_al.get(s, 0) * (p-1) + lo) / p
        ag = self._rsi_ag.get(s, 0)
        al = self._rsi_al.get(s, 0)
        rs = ag / al if al > 1e-10 else 100
        self._rsi[s] = 100 - 100 / (1 + rs)
        return self._rsi[s]

    def _compute_consensus(self, s: str, c: float, atr: float) -> tuple[int, int]:
        """Returns (bull_votes, bear_votes) out of 5."""
        bull = bear = 0

        ef = self._ef.get(s)
        em = self._em.get(s)
        es = self._es.get(s)
        if ef and em and es:
            if ef > em > es:
                bull += 1
            elif ef < em < es:
                bear += 1

        rsi = self._rsi.get(s, 50)
        if rsi > self.get_param("rsi_bull_threshold"):
            bull += 1
        elif rsi < self.get_param("rsi_bear_threshold"):
            bear += 1

        closes = list(self._c[s])
        if len(closes) >= 26:
            def _ema_list(data, p):
                r = [data[0]]
                a = 2.0 / (p + 1)
                for i in range(1, len(data)):
                    r.append(a * data[i] + (1-a) * r[-1])
                return r
            f_ema = _ema_list(closes, self.get_param("macd_fast"))
            s_ema = _ema_list(closes, self.get_param("macd_slow"))
            macd = f_ema[-1] - s_ema[-1]
            if macd > 0:
                bull += 1
            else:
                bear += 1

        vp = self.get_param("vwap_period")
        if len(closes) >= vp and len(self._v[s]) >= vp:
            tp_list = [(list(self._h[s])[i] + list(self._l[s])[i] + closes[i]) / 3 for i in range(-vp, 0)]
            v_list = list(self._v[s])[-vp:]
            tv = sum(v_list)
            if tv > 0:
                vwap = sum(t * v for t, v in zip(tp_list, v_list)) / tv
                if c > vwap:
                    bull += 1
                else:
                    bear += 1

        if self._st_dir.get(s, 0) == 1:
            bull += 1
        elif self._st_dir.get(s, 0) == -1:
            bear += 1

        return bull, bear

    def _update_supertrend(self, s: str, h: float, l: float, c: float, atr: float) -> None:
        mult = self.get_param("supertrend_atr_mult")
        mid = (h + l) / 2
        bu = mid + mult * atr
        bl = mid - mult * atr
        pu = self._st_upper.get(s, bu)
        pl = self._st_lower.get(s, bl)
        pd = self._st_dir.get(s, 0)
        u = min(bu, pu) if c > pu else bu
        lo = max(bl, pl) if c < pl else bl
        self._st_upper[s] = u
        self._st_lower[s] = lo
        if pd <= 0 and c > u:
            self._st_dir[s] = 1
        elif pd >= 0 and c < lo:
            self._st_dir[s] = -1

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._init(symbol)
        c, h, l = bar["close"], bar["high"], bar["low"]
        vol = bar.get("volume", 0.0)

        rsi = self._update_rsi(symbol, c)
        self._c[symbol].append(c)
        self._h[symbol].append(h)
        self._l[symbol].append(l)
        self._v[symbol].append(vol)

        self._ef[symbol] = ema_update(self._ef.get(symbol), c, self.get_param("ema_fast"))
        self._em[symbol] = ema_update(self._em.get(symbol), c, self.get_param("ema_medium"))
        self._es[symbol] = ema_update(self._es.get(symbol), c, self.get_param("ema_slow"))

        atr = calc_atr(self._h[symbol], self._l[symbol], self._c[symbol], self.get_param("atr_period"))
        if atr is None or atr <= 0:
            return []

        self._update_supertrend(symbol, h, l, c, atr)
        bull, bear = self._compute_consensus(symbol, c, atr)

        signals: list[Signal] = []
        pos = self.get_position(symbol)
        self._cd[symbol] = max(self._cd.get(symbol, 0) - 1, 0)

        min_cons = self.get_param("min_consensus")
        risk = self.get_param("max_risk_per_trade")
        sl_dist = atr * self.get_param("trail_atr_mult")
        pos_size = min(risk / (sl_dist / c) if sl_dist > 0 and c > 0 else 0.5, self.get_param("position_fraction"))

        if pos is None:
            if self._cd.get(symbol, 0) > 0:
                return signals

            if bull >= min_cons:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=min(bull / 5, 1.0), price=c,
                    reason=f"CONSENSUS LONG {bull}/5 signals agree",
                    metadata={"bull_votes": bull, "bear_votes": bear, "rsi": rsi, "position_fraction": pos_size},
                ))
                self._hold[symbol] = 0
                self._peak[symbol] = h

            elif bear >= min_cons:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=min(bear / 5, 1.0), price=c,
                    reason=f"CONSENSUS SHORT {bear}/5 signals agree",
                    metadata={"bull_votes": bull, "bear_votes": bear, "rsi": rsi, "position_fraction": pos_size},
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
                elif bear >= 3:
                    ex, reason = True, f"consensus reversed ({bear}/5 bearish)"
            elif pos.side.value == "sell":
                self._peak[symbol] = min(self._peak.get(symbol, c), l)
                if h >= self._peak[symbol] + trail:
                    ex, reason = True, "trail SL"
                elif bull >= 3:
                    ex, reason = True, f"consensus reversed ({bull}/5 bullish)"

            if ex:
                et = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=et, strength=0.9, price=c,
                    reason=f"CONSENSUS: {reason}",
                ))
                self._cd[symbol] = self.get_param("cooldown_bars")

        return signals

    async def generate_signals(self, md: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        for s in self.config.symbols:
            if s in md:
                out.extend(await self.on_bar(s, md[s]))
        return out
