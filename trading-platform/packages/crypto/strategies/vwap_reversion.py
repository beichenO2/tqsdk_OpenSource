"""Anchored VWAP Mean Reversion Strategy.

Institutional traders use VWAP as an execution benchmark — buy-side
algorithms target fills below VWAP, sell-side aim above VWAP. This
creates gravitational pull where price repeatedly revisits VWAP levels.

Strategy logic:
1. Compute rolling VWAP and standard deviation bands
2. When price deviates > 2 std from VWAP → expect mean reversion
3. Confirm with volume anomaly (current > 2x average)
4. Enter contrarian when price touches lower band (buy) / upper band (sell)
5. Target: VWAP itself (fair value)

Classic institutional approach, works best in ranging/weak-trend regimes.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig
from strategy.indicators import calc_atr
from strategy.registry import auto_register
from .regime_detector import MarketRegimeDetector

DEFAULT_PARAMS = {
    "vwap_period": 48,
    "std_entry_mult": 2.5,
    "std_tp_mult": 0.3,
    "atr_period": 14,
    "sl_atr_mult": 1.5,
    "max_hold_bars": 24,
    "cooldown_bars": 4,
    "volume_anomaly_mult": 1.8,
    "require_volume_anomaly": True,
    "allowed_regimes": ["ranging", "weak_trend"],
}


def _rolling_vwap(
    closes: list[float], volumes: list[float], highs: list[float], lows: list[float], period: int
) -> tuple[float, float, float] | None:
    """Compute VWAP, upper band, lower band over last `period` bars.

    VWAP = sum(typical_price * volume) / sum(volume)
    Bands = VWAP +/- std_dev of (typical_price - VWAP)
    """
    if len(closes) < period:
        return None

    tp_list = []
    vol_list = []
    for i in range(-period, 0):
        tp = (highs[i] + lows[i] + closes[i]) / 3
        tp_list.append(tp)
        vol_list.append(volumes[i])

    total_vol = sum(vol_list)
    if total_vol <= 0:
        return None

    vwap = sum(tp * v for tp, v in zip(tp_list, vol_list)) / total_vol
    variance = sum(v * (tp - vwap) ** 2 for tp, v in zip(tp_list, vol_list)) / total_vol
    std = math.sqrt(variance) if variance > 0 else 0

    return vwap, vwap + std, vwap - std


@auto_register("vwap_reversion")
class VWAPReversionStrategy(BaseStrategy):
    """Mean reversion to VWAP with volume anomaly confirmation."""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._c: dict[str, deque[float]] = {}
        self._h: dict[str, deque[float]] = {}
        self._l: dict[str, deque[float]] = {}
        self._v: dict[str, deque[float]] = {}
        self._regime: dict[str, MarketRegimeDetector] = {}
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
            self._regime[s] = MarketRegimeDetector()

    def _volume_anomaly(self, s: str) -> bool:
        vols = list(self._v[s])
        if len(vols) < 20:
            return False
        avg_vol = sum(vols[-20:]) / 20
        mult = self.get_param("volume_anomaly_mult")
        return vols[-1] >= avg_vol * mult if avg_vol > 0 else False

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._init(symbol)
        c, h, l = bar["close"], bar["high"], bar["low"]
        vol = bar.get("volume", 0.0)

        self._c[symbol].append(c)
        self._h[symbol].append(h)
        self._l[symbol].append(l)
        self._v[symbol].append(vol)
        self._regime[symbol].update(h, l, c)

        vwap_data = _rolling_vwap(
            list(self._c[symbol]), list(self._v[symbol]),
            list(self._h[symbol]), list(self._l[symbol]),
            self.get_param("vwap_period"),
        )
        if vwap_data is None:
            return []

        vwap, upper, lower = vwap_data
        std = upper - vwap
        entry_band = self.get_param("std_entry_mult")
        regime = self._regime[symbol].current_regime

        atr = calc_atr(self._h[symbol], self._l[symbol], self._c[symbol], self.get_param("atr_period"))
        if atr is None or atr <= 0:
            return []

        signals: list[Signal] = []
        pos = self.get_position(symbol)
        self._cd[symbol] = max(self._cd.get(symbol, 0) - 1, 0)

        if pos is None:
            if self._cd.get(symbol, 0) > 0:
                return signals

            allowed = self.get_param("allowed_regimes")
            if regime.value not in allowed:
                return signals

            vol_ok = not self.get_param("require_volume_anomaly") or self._volume_anomaly(symbol)
            deviation = (c - vwap) / std if std > 0 else 0

            if deviation < -entry_band and vol_ok:
                tp_price = vwap - std * self.get_param("std_tp_mult")
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY, strength=min(abs(deviation) / 3, 1.0), price=c,
                    reason=f"VWAP BUY dev={deviation:.1f}σ vwap={vwap:.0f} [{regime.value}]",
                    metadata={"vwap": vwap, "deviation_sigma": deviation, "regime": regime.value},
                ))
                self._hold[symbol] = 0
                self._entry[symbol] = c
                self._target[symbol] = tp_price

            elif deviation > entry_band and vol_ok:
                tp_price = vwap + std * self.get_param("std_tp_mult")
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY, strength=min(abs(deviation) / 3, 1.0), price=c,
                    reason=f"VWAP SELL dev={deviation:.1f}σ vwap={vwap:.0f} [{regime.value}]",
                    metadata={"vwap": vwap, "deviation_sigma": deviation, "regime": regime.value},
                ))
                self._hold[symbol] = 0
                self._entry[symbol] = c
                self._target[symbol] = tp_price

        else:
            self._hold[symbol] = self._hold.get(symbol, 0) + 1
            entry = self._entry.get(symbol, c)
            target = self._target.get(symbol, vwap)
            ex = False
            reason = ""

            if self._hold[symbol] >= self.get_param("max_hold_bars"):
                ex, reason = True, "timeout"
            elif pos.side.value == "buy":
                if c >= target:
                    ex, reason = True, f"TP at VWAP={target:.0f}"
                elif c <= entry - atr * self.get_param("sl_atr_mult"):
                    ex, reason = True, "SL"
            elif pos.side.value == "sell":
                if c <= target:
                    ex, reason = True, f"TP at VWAP={target:.0f}"
                elif c >= entry + atr * self.get_param("sl_atr_mult"):
                    ex, reason = True, "SL"

            if ex:
                et = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=et, strength=0.9, price=c,
                    reason=f"VWAP: {reason}",
                ))
                self._cd[symbol] = self.get_param("cooldown_bars")

        return signals

    async def generate_signals(self, md: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        for s in self.config.symbols:
            if s in md:
                out.extend(await self.on_bar(s, md[s]))
        return out
