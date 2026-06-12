"""Williams Fractal Strategy.

Bill Williams' fractal is a 5-bar pattern:
- Bullish fractal: middle bar has lowest low (with 2 higher lows on each side)
- Bearish fractal: middle bar has highest high (with 2 lower highs on each side)

Strategy: enter in trend direction when fractal breaks.
Uses Alligator lines (3 displaced SMAs) for trend confirmation.

Classic price action approach — works on any timeframe.
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
    "jaw_period": 13,
    "jaw_shift": 8,
    "teeth_period": 8,
    "teeth_shift": 5,
    "lips_period": 5,
    "lips_shift": 3,
    "atr_period": 14,
    "trail_atr_mult": 2.5,
    "max_hold_bars": 72,
    "cooldown_bars": 5,
    "max_risk_per_trade": 0.02,
    "position_fraction": 0.4,
}


def _sma(data: list[float], period: int) -> float | None:
    if len(data) < period:
        return None
    return sum(data[-period:]) / period


@auto_register("williams_fractal")
class WilliamsFractalStrategy(BaseStrategy):
    """Williams Fractal breakout with Alligator trend confirmation."""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._c: dict[str, deque[float]] = {}
        self._h: dict[str, deque[float]] = {}
        self._l: dict[str, deque[float]] = {}
        self._bull_fractal: dict[str, float | None] = {}
        self._bear_fractal: dict[str, float | None] = {}
        self._hold: dict[str, int] = {}
        self._cd: dict[str, int] = {}
        self._peak: dict[str, float] = {}
        self._buf = 200

    def _init(self, s: str) -> None:
        if s not in self._c:
            self._c[s] = deque(maxlen=self._buf)
            self._h[s] = deque(maxlen=self._buf)
            self._l[s] = deque(maxlen=self._buf)

    def _detect_fractals(self, s: str) -> None:
        highs = list(self._h[s])
        lows = list(self._l[s])
        if len(highs) < 5:
            return

        mid = -3
        if (highs[mid] > highs[mid-1] and highs[mid] > highs[mid-2] and
            highs[mid] > highs[mid+1] and highs[mid] > highs[mid+2]):
            self._bear_fractal[s] = highs[mid]

        if (lows[mid] < lows[mid-1] and lows[mid] < lows[mid-2] and
            lows[mid] < lows[mid+1] and lows[mid] < lows[mid+2]):
            self._bull_fractal[s] = lows[mid]

    def _alligator(self, s: str) -> tuple[float | None, float | None, float | None]:
        closes = list(self._c[s])
        jaw = _sma(closes, self.get_param("jaw_period"))
        teeth = _sma(closes, self.get_param("teeth_period"))
        lips = _sma(closes, self.get_param("lips_period"))
        return jaw, teeth, lips

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._init(symbol)
        c, h, l = bar["close"], bar["high"], bar["low"]
        self._c[symbol].append(c)
        self._h[symbol].append(h)
        self._l[symbol].append(l)
        self._detect_fractals(symbol)

        atr = calc_atr(self._h[symbol], self._l[symbol], self._c[symbol], self.get_param("atr_period"))
        if atr is None or atr <= 0:
            return []

        jaw, teeth, lips = self._alligator(symbol)
        if jaw is None or teeth is None or lips is None:
            return []

        bull_frac = self._bull_fractal.get(symbol)
        bear_frac = self._bear_fractal.get(symbol)
        alligator_bull = lips > teeth > jaw
        alligator_bear = lips < teeth < jaw

        signals: list[Signal] = []
        pos = self.get_position(symbol)
        self._cd[symbol] = max(self._cd.get(symbol, 0) - 1, 0)

        risk = self.get_param("max_risk_per_trade")
        sl_dist = atr * self.get_param("trail_atr_mult")
        pos_size = min(risk / (sl_dist / c) if sl_dist > 0 and c > 0 else 0.4, self.get_param("position_fraction"))

        if pos is None:
            if self._cd.get(symbol, 0) > 0:
                return signals

            if bear_frac and c > bear_frac and alligator_bull:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY, strength=0.8, price=c,
                    reason=f"FRACTAL LONG above {bear_frac:.0f}, alligator bullish",
                    metadata={"fractal": bear_frac, "jaw": jaw, "teeth": teeth, "lips": lips, "position_fraction": pos_size},
                ))
                self._hold[symbol] = 0
                self._peak[symbol] = h

            elif bull_frac and c < bull_frac and alligator_bear:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY, strength=0.8, price=c,
                    reason=f"FRACTAL SHORT below {bull_frac:.0f}, alligator bearish",
                    metadata={"fractal": bull_frac, "jaw": jaw, "teeth": teeth, "lips": lips, "position_fraction": pos_size},
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
            elif pos.side.value == "sell":
                self._peak[symbol] = min(self._peak.get(symbol, c), l)
                if h >= self._peak[symbol] + trail:
                    ex, reason = True, "trail SL"

            if ex:
                et = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=et, strength=0.9, price=c,
                    reason=f"FRACTAL: {reason}",
                ))
                self._cd[symbol] = self.get_param("cooldown_bars")

        return signals

    async def generate_signals(self, md: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        for s in self.config.symbols:
            if s in md:
                out.extend(await self.on_bar(s, md[s]))
        return out
