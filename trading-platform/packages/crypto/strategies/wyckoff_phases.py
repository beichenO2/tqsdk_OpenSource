"""Wyckoff Spring/Upthrust Detection Strategy.

Classic institutional market structure analysis:
- Spring: price breaks below support → immediately reverses up (shakeout)
  = institutions buying cheap from panicked retail stops
- Upthrust: price breaks above resistance → immediately reverses down
  = institutions selling into retail FOMO

Detection method:
1. Identify support/resistance via recent swing lows/highs
2. Detect false breakout (wick beyond level, close back inside)
3. Confirm with volume (spring should have declining volume → low conviction)
4. Enter on reversal bar after spring/upthrust confirmed

Best on 4h/1d timeframes where patterns have time to develop.
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
    "swing_lookback": 40,
    "min_touches": 3,
    "false_break_pct": 0.01,
    "volume_decline_ratio": 0.7,
    "atr_period": 14,
    "sl_atr_mult": 1.5,
    "tp_atr_mult": 4.0,
    "max_hold_bars": 60,
    "cooldown_bars": 12,
    "confirmation_bars": 2,
}


def _find_support_resistance(
    highs: list[float], lows: list[float], closes: list[float],
    lookback: int, min_touches: int, tolerance_pct: float = 0.01,
) -> tuple[float | None, float | None]:
    """Find nearest support and resistance levels from swing points."""
    if len(closes) < lookback:
        return None, None

    window_h = highs[-lookback:]
    window_l = lows[-lookback:]
    current = closes[-1]

    resistance_candidates: list[float] = []
    support_candidates: list[float] = []

    for i in range(2, len(window_h) - 2):
        if window_h[i] >= max(window_h[i-2:i]) and window_h[i] >= max(window_h[i+1:i+3]):
            if window_h[i] > current:
                resistance_candidates.append(window_h[i])
        if window_l[i] <= min(window_l[i-2:i]) and window_l[i] <= min(window_l[i+1:i+3]):
            if window_l[i] < current:
                support_candidates.append(window_l[i])

    def _cluster_level(candidates: list[float], tol: float) -> float | None:
        if not candidates:
            return None
        best_level = None
        best_count = 0
        for c in candidates:
            count = sum(1 for x in candidates if abs(x - c) / c < tol)
            if count >= min_touches and count > best_count:
                best_count = count
                best_level = c
        if best_level is not None:
            return best_level
        return sorted(candidates, key=lambda x: abs(x - current))[0]

    resistance = _cluster_level(resistance_candidates, tolerance_pct)
    support = _cluster_level(support_candidates, tolerance_pct)
    return support, resistance


@auto_register("wyckoff_phases")
class WyckoffPhasesStrategy(BaseStrategy):
    """Detect Wyckoff Springs and Upthrusts for entry."""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._c: dict[str, deque[float]] = {}
        self._h: dict[str, deque[float]] = {}
        self._l: dict[str, deque[float]] = {}
        self._o: dict[str, deque[float]] = {}
        self._v: dict[str, deque[float]] = {}
        self._spring_pending: dict[str, int] = {}
        self._upthrust_pending: dict[str, int] = {}
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
            self._v[s] = deque(maxlen=self._buf)

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._init(symbol)
        c, h, l, o = bar["close"], bar["high"], bar["low"], bar["open"]
        vol = bar.get("volume", 0.0)

        self._c[symbol].append(c)
        self._h[symbol].append(h)
        self._l[symbol].append(l)
        self._o[symbol].append(o)
        self._v[symbol].append(vol)

        lookback = self.get_param("swing_lookback")
        if len(self._c[symbol]) < lookback + 5:
            return []

        support, resistance = _find_support_resistance(
            list(self._h[symbol]), list(self._l[symbol]), list(self._c[symbol]),
            lookback, self.get_param("min_touches"),
        )

        atr = calc_atr(self._h[symbol], self._l[symbol], self._c[symbol], self.get_param("atr_period"))
        if atr is None or atr <= 0:
            return []

        false_break = self.get_param("false_break_pct")
        vol_decline = self.get_param("volume_decline_ratio")
        vols = list(self._v[symbol])
        avg_vol = sum(vols[-20:]) / 20 if len(vols) >= 20 else vol

        if support is not None and l < support * (1 - false_break) and c > support:
            is_low_vol = vol < avg_vol * vol_decline
            if is_low_vol or (c > o):
                self._spring_pending[symbol] = self.get_param("confirmation_bars")

        if resistance is not None and h > resistance * (1 + false_break) and c < resistance:
            is_low_vol = vol < avg_vol * vol_decline
            if is_low_vol or (c < o):
                self._upthrust_pending[symbol] = self.get_param("confirmation_bars")

        signals: list[Signal] = []
        pos = self.get_position(symbol)
        self._cd[symbol] = max(self._cd.get(symbol, 0) - 1, 0)

        spring_p = self._spring_pending.get(symbol, 0)
        upthrust_p = self._upthrust_pending.get(symbol, 0)

        if spring_p > 0:
            self._spring_pending[symbol] = spring_p - 1
        if upthrust_p > 0:
            self._upthrust_pending[symbol] = upthrust_p - 1

        if pos is None:
            if self._cd.get(symbol, 0) > 0:
                return signals

            if spring_p == 1 and c > o:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY, strength=0.8, price=c,
                    reason=f"WYCKOFF SPRING confirmed, support={support:.0f}",
                    metadata={"support": support, "atr": atr},
                ))
                self._hold[symbol] = 0
                self._entry[symbol] = c
                self._spring_pending[symbol] = 0

            elif upthrust_p == 1 and c < o:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY, strength=0.8, price=c,
                    reason=f"WYCKOFF UPTHRUST confirmed, resistance={resistance:.0f}",
                    metadata={"resistance": resistance, "atr": atr},
                ))
                self._hold[symbol] = 0
                self._entry[symbol] = c
                self._upthrust_pending[symbol] = 0

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
                    reason=f"WYCKOFF: {reason}",
                ))
                self._cd[symbol] = self.get_param("cooldown_bars")

        return signals

    async def generate_signals(self, md: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        for s in self.config.symbols:
            if s in md:
                out.extend(await self.on_bar(s, md[s]))
        return out
