"""Volume Profile + Order Flow Strategy.

Based on professional trader methodology (Trader Dale / Kalena 2026):
1. Volume Profile identifies "Heavy Volume Zones" (HVZ) — institutional accumulation
2. Cumulative Volume Delta (CVD) reveals buying/selling pressure
3. Taker Buy Ratio as order flow proxy (from Binance OHLCV data)

This is a mean-reversion strategy at key levels with flow confirmation:
- Enter when price reaches HVZ AND flow diverges (absorption)
- Exit when price leaves HVZ OR flow confirms (no absorption)

No Level2 data needed — uses taker_buy_volume from Binance public OHLCV.
For higher fidelity, feed open_interest changes as delta proxy.

Key insight from research: "Order flow reveals cause, not effect —
providing 30-90 seconds lead time before candlesticks confirm moves."
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any

import numpy as np

from strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig
from strategy.indicators import calc_atr
from strategy.registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "vp_lookback": 120,
    "vp_bins": 30,
    "hvz_volume_percentile": 0.7,
    "cvd_period": 20,
    "cvd_divergence_threshold": 0.3,
    "taker_ratio_extreme_high": 0.58,
    "taker_ratio_extreme_low": 0.42,
    "atr_period": 14,
    "sl_atr_mult": 1.5,
    "tp_atr_mult": 3.0,
    "max_hold_bars": 48,
    "cooldown_bars": 4,
    "min_bars_in_zone": 3,
}


def _compute_volume_profile(
    closes: list[float], volumes: list[float], n_bins: int = 30
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute volume profile (price bins vs total volume).

    Returns (bin_edges, bin_volumes, poc_price).
    POC = Point of Control = price level with highest traded volume.
    """
    if len(closes) < 10:
        return np.array([]), np.array([]), 0.0

    price_min = min(closes)
    price_max = max(closes)
    if price_max <= price_min:
        return np.array([]), np.array([]), closes[-1]

    edges = np.linspace(price_min, price_max, n_bins + 1)
    bin_vols = np.zeros(n_bins)

    for c, v in zip(closes, volumes):
        idx = min(int((c - price_min) / (price_max - price_min) * n_bins), n_bins - 1)
        bin_vols[idx] += v

    poc_idx = int(np.argmax(bin_vols))
    poc_price = (edges[poc_idx] + edges[poc_idx + 1]) / 2

    return edges, bin_vols, poc_price


def _is_in_hvz(
    price: float, edges: np.ndarray, bin_vols: np.ndarray, percentile: float
) -> bool:
    """Check if price is in a High Volume Zone."""
    if len(edges) < 2 or len(bin_vols) == 0:
        return False

    threshold = np.percentile(bin_vols, percentile * 100)
    for i in range(len(bin_vols)):
        if bin_vols[i] >= threshold:
            if edges[i] <= price <= edges[i + 1]:
                return True
    return False


@auto_register("volume_profile_flow")
class VolumeProfileFlowStrategy(BaseStrategy):
    """Volume Profile zones + Taker flow confirmation."""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._closes: dict[str, deque[float]] = {}
        self._highs: dict[str, deque[float]] = {}
        self._lows: dict[str, deque[float]] = {}
        self._volumes: dict[str, deque[float]] = {}
        self._taker_buy: dict[str, deque[float]] = {}
        self._cvd: dict[str, deque[float]] = {}
        self._bars_in_zone: dict[str, int] = {}
        self._hold: dict[str, int] = {}
        self._cd: dict[str, int] = {}
        self._entry: dict[str, float] = {}
        self._buf = 200

    def _init(self, s: str) -> None:
        if s not in self._closes:
            self._closes[s] = deque(maxlen=self._buf)
            self._highs[s] = deque(maxlen=self._buf)
            self._lows[s] = deque(maxlen=self._buf)
            self._volumes[s] = deque(maxlen=self._buf)
            self._taker_buy[s] = deque(maxlen=self._buf)
            self._cvd[s] = deque(maxlen=self._buf)

    def _update_cvd(self, s: str, volume: float, taker_buy_vol: float) -> None:
        """Cumulative Volume Delta = taker_buy - taker_sell."""
        taker_sell = volume - taker_buy_vol
        delta = taker_buy_vol - taker_sell
        prev_cvd = self._cvd[s][-1] if self._cvd[s] else 0.0
        self._cvd[s].append(prev_cvd + delta)

    def _cvd_diverges(self, s: str, price_direction: str) -> bool:
        """Check if CVD diverges from price (absorption signal)."""
        cvd_list = list(self._cvd[s])
        period = self.get_param("cvd_period")
        if len(cvd_list) < period:
            return False

        cvd_change = (cvd_list[-1] - cvd_list[-period]) / (abs(cvd_list[-period]) + 1e-10)
        threshold = self.get_param("cvd_divergence_threshold")

        if price_direction == "falling" and cvd_change > threshold:
            return True
        if price_direction == "rising" and cvd_change < -threshold:
            return True
        return False

    def _taker_ratio(self, s: str) -> float:
        """Current taker buy ratio = taker_buy_vol / total_vol."""
        if not self._volumes[s] or not self._taker_buy[s]:
            return 0.5
        vol = self._volumes[s][-1]
        tbv = self._taker_buy[s][-1]
        return tbv / vol if vol > 0 else 0.5

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._init(symbol)
        c, h, l = bar["close"], bar["high"], bar["low"]
        vol = bar.get("volume", 0.0)
        tbv = bar.get("taker_buy_volume", vol * 0.5)

        self._closes[symbol].append(c)
        self._highs[symbol].append(h)
        self._lows[symbol].append(l)
        self._volumes[symbol].append(vol)
        self._taker_buy[symbol].append(tbv)
        self._update_cvd(symbol, vol, tbv)

        lookback = self.get_param("vp_lookback")
        closes = list(self._closes[symbol])
        volumes = list(self._volumes[symbol])
        if len(closes) < lookback:
            return []

        vp_closes = closes[-lookback:]
        vp_volumes = volumes[-lookback:]
        edges, bin_vols, poc = _compute_volume_profile(
            vp_closes, vp_volumes, self.get_param("vp_bins")
        )

        in_hvz = _is_in_hvz(c, edges, bin_vols, self.get_param("hvz_volume_percentile"))
        if in_hvz:
            self._bars_in_zone[symbol] = self._bars_in_zone.get(symbol, 0) + 1
        else:
            self._bars_in_zone[symbol] = 0

        atr = calc_atr(self._highs[symbol], self._lows[symbol], self._closes[symbol], self.get_param("atr_period"))
        if atr is None or atr <= 0:
            return []

        signals: list[Signal] = []
        pos = self.get_position(symbol)
        self._cd[symbol] = max(self._cd.get(symbol, 0) - 1, 0)
        taker_r = self._taker_ratio(symbol)

        if pos is None:
            if self._cd.get(symbol, 0) > 0:
                return signals

            min_bars = self.get_param("min_bars_in_zone")
            if not in_hvz or self._bars_in_zone[symbol] < min_bars:
                return signals

            price_falling = len(closes) >= 5 and closes[-1] < closes[-5]
            price_rising = len(closes) >= 5 and closes[-1] > closes[-5]

            if price_falling and c < poc:
                cvd_diverges = self._cvd_diverges(symbol, "falling")
                taker_buying = taker_r >= self.get_param("taker_ratio_extreme_high")
                if cvd_diverges or taker_buying:
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.LONG_ENTRY, strength=0.75, price=c,
                        reason=f"VPF LONG at HVZ, POC={poc:.0f}, CVD_div={cvd_diverges}, taker={taker_r:.2f}",
                        metadata={"poc": poc, "taker_ratio": taker_r, "in_hvz": True},
                    ))
                    self._hold[symbol] = 0
                    self._entry[symbol] = c

            elif price_rising and c > poc:
                cvd_diverges = self._cvd_diverges(symbol, "rising")
                taker_selling = taker_r <= self.get_param("taker_ratio_extreme_low")
                if cvd_diverges or taker_selling:
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.SHORT_ENTRY, strength=0.75, price=c,
                        reason=f"VPF SHORT at HVZ, POC={poc:.0f}, CVD_div={cvd_diverges}, taker={taker_r:.2f}",
                        metadata={"poc": poc, "taker_ratio": taker_r, "in_hvz": True},
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
                elif not in_hvz and self._hold[symbol] > 5:
                    ex, reason = True, "left HVZ"
            elif pos.side.value == "sell":
                if c >= entry + atr * self.get_param("sl_atr_mult"):
                    ex, reason = True, "SL"
                elif c <= entry - atr * self.get_param("tp_atr_mult"):
                    ex, reason = True, "TP"
                elif not in_hvz and self._hold[symbol] > 5:
                    ex, reason = True, "left HVZ"

            if ex:
                et = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=et, strength=0.9, price=c,
                    reason=f"VPF: {reason}",
                ))
                self._cd[symbol] = self.get_param("cooldown_bars")

        return signals

    async def generate_signals(self, md: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        for s in self.config.symbols:
            if s in md:
                out.extend(await self.on_bar(s, md[s]))
        return out
