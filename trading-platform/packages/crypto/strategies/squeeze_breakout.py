"""Volatility Squeeze Breakout Strategy.

Detects periods of low volatility (squeeze) and trades the breakout
when volatility expands. Combines Bollinger/Keltner squeeze detection
with momentum ignition confirmation.

Squeeze = BB inside KC (Bollinger Bands narrower than Keltner Channels)
When BB breaks outside KC → squeeze fires → enter breakout direction.

This captures the "coiled spring" pattern: extended low-vol periods
tend to resolve with explosive directional moves.
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
    "bb_period": 20,
    "bb_std": 2.0,
    "kc_period": 20,
    "kc_atr_mult": 1.5,
    "atr_period": 14,
    "min_squeeze_bars": 3,
    "roc_period": 10,
    "volume_surge_mult": 1.5,
    "trail_atr_mult": 2.5,
    "max_hold_bars": 60,
    "cooldown_bars": 6,
    "ema_trend_period": 50,
}


@auto_register("squeeze_breakout")
class SqueezeBreakoutStrategy(BaseStrategy):
    """BB/KC squeeze detection + momentum breakout."""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._c: dict[str, deque[float]] = {}
        self._h: dict[str, deque[float]] = {}
        self._l: dict[str, deque[float]] = {}
        self._v: dict[str, deque[float]] = {}
        self._ema: dict[str, float | None] = {}
        self._squeeze_bars: dict[str, int] = {}
        self._was_squeezing: dict[str, bool] = {}
        self._hold: dict[str, int] = {}
        self._cd: dict[str, int] = {}
        self._entry: dict[str, float] = {}
        self._peak: dict[str, float] = {}
        self._buf = 200

    def _init(self, s: str) -> None:
        if s not in self._c:
            self._c[s] = deque(maxlen=self._buf)
            self._h[s] = deque(maxlen=self._buf)
            self._l[s] = deque(maxlen=self._buf)
            self._v[s] = deque(maxlen=self._buf)

    def _is_squeezing(self, s: str) -> bool:
        """Check if BB is inside KC (squeeze condition)."""
        closes = list(self._c[s])
        bb_p = self.get_param("bb_period")
        kc_p = self.get_param("kc_period")

        if len(closes) < max(bb_p, kc_p):
            return False

        bb_window = closes[-bb_p:]
        bb_mid = sum(bb_window) / bb_p
        bb_std = math.sqrt(sum((x - bb_mid)**2 for x in bb_window) / bb_p)
        bb_upper = bb_mid + self.get_param("bb_std") * bb_std
        bb_lower = bb_mid - self.get_param("bb_std") * bb_std

        atr = calc_atr(self._h[s], self._l[s], self._c[s], self.get_param("atr_period"))
        if atr is None or atr <= 0:
            return False

        kc_mid = sum(closes[-kc_p:]) / kc_p
        kc_mult = self.get_param("kc_atr_mult")
        kc_upper = kc_mid + kc_mult * atr
        kc_lower = kc_mid - kc_mult * atr

        return bb_upper < kc_upper and bb_lower > kc_lower

    def _roc(self, s: str) -> float:
        """Rate of Change momentum."""
        closes = list(self._c[s])
        p = self.get_param("roc_period")
        if len(closes) < p + 1:
            return 0.0
        return (closes[-1] - closes[-p-1]) / closes[-p-1] if closes[-p-1] > 0 else 0.0

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._init(symbol)
        c, h, l = bar["close"], bar["high"], bar["low"]
        vol = bar.get("volume", 0.0)

        self._c[symbol].append(c)
        self._h[symbol].append(h)
        self._l[symbol].append(l)
        self._v[symbol].append(vol)
        self._ema[symbol] = ema_update(self._ema.get(symbol), c, self.get_param("ema_trend_period"))

        squeezing = self._is_squeezing(symbol)
        if squeezing:
            self._squeeze_bars[symbol] = self._squeeze_bars.get(symbol, 0) + 1
        else:
            was = self._was_squeezing.get(symbol, False)
            if was and self._squeeze_bars.get(symbol, 0) >= self.get_param("min_squeeze_bars"):
                pass
            self._squeeze_bars[symbol] = 0
        self._was_squeezing[symbol] = squeezing

        atr = calc_atr(self._h[symbol], self._l[symbol], self._c[symbol], self.get_param("atr_period"))
        if atr is None or atr <= 0:
            return []

        signals: list[Signal] = []
        pos = self.get_position(symbol)
        self._cd[symbol] = max(self._cd.get(symbol, 0) - 1, 0)

        if pos is None:
            if self._cd.get(symbol, 0) > 0:
                return signals

            was_squeezing = self._was_squeezing.get(symbol, False)
            squeeze_count = self._squeeze_bars.get(symbol, 0)
            just_released = not squeezing and was_squeezing and squeeze_count >= self.get_param("min_squeeze_bars")

            # Even if not just released, check if we recently released (within 2 bars)
            if not just_released:
                return signals

            roc = self._roc(symbol)
            ema_val = self._ema.get(symbol, c)

            vols = list(self._v[symbol])
            vol_ok = True
            if len(vols) >= 20:
                avg_vol = sum(vols[-20:]) / 20
                vol_ok = vol >= avg_vol * self.get_param("volume_surge_mult") if avg_vol > 0 else True

            if roc > 0 and c > ema_val and vol_ok:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY, strength=0.85, price=c,
                    reason=f"SQUEEZE LONG release after {squeeze_count} bars, ROC={roc:.3f}",
                    metadata={"squeeze_bars": squeeze_count, "roc": roc, "atr": atr},
                ))
                self._hold[symbol] = 0
                self._entry[symbol] = c
                self._peak[symbol] = h

            elif roc < 0 and c < ema_val and vol_ok:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY, strength=0.85, price=c,
                    reason=f"SQUEEZE SHORT release after {squeeze_count} bars, ROC={roc:.3f}",
                    metadata={"squeeze_bars": squeeze_count, "roc": roc, "atr": atr},
                ))
                self._hold[symbol] = 0
                self._entry[symbol] = c
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
                    ex, reason = True, f"trail SL @{self._peak[symbol] - trail:.0f}"
            elif pos.side.value == "sell":
                self._peak[symbol] = min(self._peak.get(symbol, c), l)
                if h >= self._peak[symbol] + trail:
                    ex, reason = True, f"trail SL @{self._peak[symbol] + trail:.0f}"

            if ex:
                et = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=et, strength=0.9, price=c,
                    reason=f"SQUEEZE: {reason}",
                ))
                self._cd[symbol] = self.get_param("cooldown_bars")

        return signals

    async def generate_signals(self, md: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        for s in self.config.symbols:
            if s in md:
                out.extend(await self.on_bar(s, md[s]))
        return out
