"""Extreme Reversal Strategy — Fade Overextended Moves.

When price makes an extreme move (> N standard deviations in a single bar
or over a short window), the probability of at least a partial reversion
is elevated. This strategy fades extreme moves.

Works best in conjunction with volume divergence — extreme price moves
on declining volume are more likely to revert.

This is essentially a "rubber band" strategy: the further price stretches
from its recent mean, the stronger the reversion force.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any

from strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig
from strategy.indicators import calc_atr
from strategy.registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "lookback": 30,
    "z_entry": 2.5,
    "z_exit": 0.5,
    "z_stop": 4.0,
    "volume_divergence": True,
    "volume_ratio_threshold": 0.8,
    "atr_period": 14,
    "sl_atr_mult": 2.0,
    "max_hold_bars": 24,
    "cooldown_bars": 4,
    "max_risk_per_trade": 0.02,
    "position_fraction": 0.35,
}


@auto_register("extreme_reversal")
class ExtremeReversalStrategy(BaseStrategy):
    """Fade extreme price deviations with volume divergence confirmation."""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._c: dict[str, deque[float]] = {}
        self._h: dict[str, deque[float]] = {}
        self._l: dict[str, deque[float]] = {}
        self._v: dict[str, deque[float]] = {}
        self._hold: dict[str, int] = {}
        self._cd: dict[str, int] = {}
        self._entry: dict[str, float] = {}
        self._target: dict[str, float] = {}
        self._buf = 200

    def _init(self, s: str) -> None:
        if s not in self._c:
            self._c[s] = deque(maxlen=self._buf)
            self._h[s] = deque(maxlen=self._buf)
            self._l[s] = deque(maxlen=self._buf)
            self._v[s] = deque(maxlen=self._buf)

    def _price_zscore(self, s: str) -> tuple[float, float] | None:
        closes = list(self._c[s])
        lb = self.get_param("lookback")
        if len(closes) < lb:
            return None
        window = closes[-lb:]
        mean_p = sum(window) / lb
        std_p = math.sqrt(sum((x - mean_p)**2 for x in window) / lb)
        if std_p < 1e-10:
            return None
        z = (closes[-1] - mean_p) / std_p
        return z, mean_p

    def _volume_diverging(self, s: str) -> bool:
        vols = list(self._v[s])
        if len(vols) < 20:
            return False
        avg = sum(vols[-20:]) / 20
        ratio = self.get_param("volume_ratio_threshold")
        return vols[-1] < avg * ratio

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._init(symbol)
        c, h, l = bar["close"], bar["high"], bar["low"]
        vol = bar.get("volume", 0.0)
        self._c[symbol].append(c)
        self._h[symbol].append(h)
        self._l[symbol].append(l)
        self._v[symbol].append(vol)

        atr = calc_atr(self._h[symbol], self._l[symbol], self._c[symbol], self.get_param("atr_period"))
        if atr is None or atr <= 0:
            return []

        z_data = self._price_zscore(symbol)
        if z_data is None:
            return []

        z, mean_p = z_data
        z_entry = self.get_param("z_entry")
        z_exit = self.get_param("z_exit")
        z_stop = self.get_param("z_stop")

        signals: list[Signal] = []
        pos = self.get_position(symbol)
        self._cd[symbol] = max(self._cd.get(symbol, 0) - 1, 0)

        risk = self.get_param("max_risk_per_trade")
        sl_dist = atr * self.get_param("sl_atr_mult")
        pos_size = min(risk / (sl_dist / c) if sl_dist > 0 and c > 0 else 0.35, self.get_param("position_fraction"))

        use_vol_div = self.get_param("volume_divergence")
        vol_div = not use_vol_div or self._volume_diverging(symbol)

        if pos is None:
            if self._cd.get(symbol, 0) > 0:
                return signals

            if z > z_entry and vol_div:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=min(abs(z) / 4, 1.0), price=c,
                    reason=f"EXTREME SHORT z={z:.2f} mean={mean_p:.0f}",
                    metadata={"z_score": z, "mean_price": mean_p, "position_fraction": pos_size},
                ))
                self._hold[symbol] = 0
                self._entry[symbol] = c
                self._target[symbol] = mean_p

            elif z < -z_entry and vol_div:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=min(abs(z) / 4, 1.0), price=c,
                    reason=f"EXTREME LONG z={z:.2f} mean={mean_p:.0f}",
                    metadata={"z_score": z, "mean_price": mean_p, "position_fraction": pos_size},
                ))
                self._hold[symbol] = 0
                self._entry[symbol] = c
                self._target[symbol] = mean_p

        else:
            self._hold[symbol] = self._hold.get(symbol, 0) + 1
            target = self._target.get(symbol, mean_p)
            ex = False
            reason = ""

            if self._hold[symbol] >= self.get_param("max_hold_bars"):
                ex, reason = True, "timeout"
            elif abs(z) < z_exit:
                ex, reason = True, f"reverted z={z:.2f}"
            elif abs(z) > z_stop:
                ex, reason = True, f"z-stop z={z:.2f}"
            elif pos.side.value == "buy" and c >= target:
                ex, reason = True, f"TP at mean {target:.0f}"
            elif pos.side.value == "sell" and c <= target:
                ex, reason = True, f"TP at mean {target:.0f}"
            elif pos.side.value == "buy" and c <= self._entry.get(symbol, c) - atr * self.get_param("sl_atr_mult"):
                ex, reason = True, "SL"
            elif pos.side.value == "sell" and c >= self._entry.get(symbol, c) + atr * self.get_param("sl_atr_mult"):
                ex, reason = True, "SL"

            if ex:
                et = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=et, strength=0.9, price=c,
                    reason=f"EXTREME: {reason}",
                ))
                self._cd[symbol] = self.get_param("cooldown_bars")

        return signals

    async def generate_signals(self, md: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        for s in self.config.symbols:
            if s in md:
                out.extend(await self.on_bar(s, md[s]))
        return out
