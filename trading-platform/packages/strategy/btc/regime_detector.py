"""Market regime detector — identifies trending, ranging, and high-volatility states.

Uses a combination of ADX, Bollinger Band width, and realized volatility
to classify the current market micro-regime and output recommended
parameter adjustments for each BTC strategy.
"""

from __future__ import annotations

import enum
import logging
import math
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)


class MarketRegime(str, enum.Enum):
    STRONG_TREND = "strong_trend"
    WEAK_TREND = "weak_trend"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    BREAKOUT = "breakout"
    UNKNOWN = "unknown"


REGIME_PARAMS: dict[MarketRegime, dict[str, Any]] = {
    MarketRegime.STRONG_TREND: {
        "recommended_strategies": ["time_series_momentum", "scalp_momentum"],
        "position_scale": 1.0,
        "stop_loss_mult": 2.5,
        "take_profit_mult": 4.0,
        "signal_threshold_scale": 0.8,
    },
    MarketRegime.WEAK_TREND: {
        "recommended_strategies": ["time_series_momentum", "funding_rate_alpha"],
        "position_scale": 0.7,
        "stop_loss_mult": 2.0,
        "take_profit_mult": 3.0,
        "signal_threshold_scale": 1.0,
    },
    MarketRegime.RANGING: {
        "recommended_strategies": ["funding_rate_alpha", "vol_breakout_scalp"],
        "position_scale": 0.8,
        "stop_loss_mult": 1.5,
        "take_profit_mult": 2.0,
        "signal_threshold_scale": 0.9,
    },
    MarketRegime.HIGH_VOLATILITY: {
        "recommended_strategies": ["funding_rate_alpha"],
        "position_scale": 0.4,
        "stop_loss_mult": 3.0,
        "take_profit_mult": 5.0,
        "signal_threshold_scale": 1.3,
    },
    MarketRegime.BREAKOUT: {
        "recommended_strategies": ["time_series_momentum", "scalp_momentum"],
        "position_scale": 0.6,
        "stop_loss_mult": 2.0,
        "take_profit_mult": 4.0,
        "signal_threshold_scale": 0.7,
    },
    MarketRegime.UNKNOWN: {
        "recommended_strategies": [],
        "position_scale": 0.3,
        "stop_loss_mult": 2.0,
        "take_profit_mult": 3.0,
        "signal_threshold_scale": 1.0,
    },
}


class MarketRegimeDetector:
    """Classify the current market regime from price data.

    Uses three independent indicators and combines them via a voting scheme:
    1. ADX — trend strength
    2. Bollinger Band width — volatility squeeze / expansion
    3. Realized volatility percentile — absolute vol level
    """

    def __init__(
        self,
        adx_period: int = 14,
        bb_period: int = 20,
        bb_std: float = 2.0,
        vol_period: int = 20,
        vol_lookback: int = 100,
        adx_strong: float = 30.0,
        adx_weak: float = 20.0,
        bb_squeeze_pct: float = 0.25,
        bb_expand_pct: float = 0.75,
        vol_high_pct: float = 0.80,
    ) -> None:
        self._adx_period = adx_period
        self._bb_period = bb_period
        self._bb_std = bb_std
        self._vol_period = vol_period
        self._vol_lookback = vol_lookback
        self._adx_strong = adx_strong
        self._adx_weak = adx_weak
        self._bb_squeeze_pct = bb_squeeze_pct
        self._bb_expand_pct = bb_expand_pct
        self._vol_high_pct = vol_high_pct

        self._closes: deque[float] = deque(maxlen=max(vol_lookback + vol_period, 200))
        self._highs: deque[float] = deque(maxlen=200)
        self._lows: deque[float] = deque(maxlen=200)

        self._prev_plus_dm: float = 0
        self._prev_minus_dm: float = 0
        self._prev_tr: float = 0
        self._adx: float | None = None
        self._dx_history: deque[float] = deque(maxlen=adx_period)
        self._bar_count: int = 0

        self._bb_width_history: deque[float] = deque(maxlen=vol_lookback)
        self._vol_history: deque[float] = deque(maxlen=vol_lookback)
        self._last_regime: MarketRegime = MarketRegime.UNKNOWN

    @property
    def current_regime(self) -> MarketRegime:
        return self._last_regime

    def update(self, high: float, low: float, close: float) -> MarketRegime:
        """Feed a new bar and return the detected regime."""
        self._closes.append(close)
        self._highs.append(high)
        self._lows.append(low)
        self._bar_count += 1

        adx = self._update_adx()
        bb_width_pctile = self._update_bb()
        vol_pctile = self._update_vol()

        self._last_regime = self._classify(adx, bb_width_pctile, vol_pctile)
        return self._last_regime

    def get_params(self, regime: MarketRegime | None = None) -> dict[str, Any]:
        """Return recommended strategy parameters for the current (or given) regime."""
        r = regime or self._last_regime
        return dict(REGIME_PARAMS.get(r, REGIME_PARAMS[MarketRegime.UNKNOWN]))

    def _update_adx(self) -> float | None:
        if len(self._highs) < 2:
            return None

        highs = list(self._highs)
        lows = list(self._lows)
        closes = list(self._closes)

        high_diff = highs[-1] - highs[-2]
        low_diff = lows[-2] - lows[-1]
        plus_dm = max(high_diff, 0.0) if high_diff > low_diff else 0.0
        minus_dm = max(low_diff, 0.0) if low_diff > high_diff else 0.0

        tr = max(
            highs[-1] - lows[-1],
            abs(highs[-1] - closes[-2]),
            abs(lows[-1] - closes[-2]),
        )

        p = self._adx_period
        if self._bar_count <= p + 1:
            self._prev_tr += tr
            self._prev_plus_dm += plus_dm
            self._prev_minus_dm += minus_dm
            if self._bar_count < p + 1:
                return None
        else:
            self._prev_tr = self._prev_tr - self._prev_tr / p + tr
            self._prev_plus_dm = self._prev_plus_dm - self._prev_plus_dm / p + plus_dm
            self._prev_minus_dm = self._prev_minus_dm - self._prev_minus_dm / p + minus_dm

        if self._prev_tr == 0:
            return None

        plus_di = 100 * self._prev_plus_dm / self._prev_tr
        minus_di = 100 * self._prev_minus_dm / self._prev_tr
        di_sum = plus_di + minus_di
        dx = 100 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0

        self._dx_history.append(dx)
        if len(self._dx_history) < self._adx_period:
            return None

        if self._adx is None:
            self._adx = sum(self._dx_history) / len(self._dx_history)
        else:
            self._adx = (self._adx * (p - 1) + dx) / p

        return self._adx

    def _update_bb(self) -> float | None:
        if len(self._closes) < self._bb_period:
            return None

        window = list(self._closes)[-self._bb_period:]
        mid = sum(window) / len(window)
        std = math.sqrt(sum((x - mid) ** 2 for x in window) / len(window))
        width = (2 * self._bb_std * std) / mid if mid > 0 else 0

        self._bb_width_history.append(width)
        if len(self._bb_width_history) < 10:
            return None

        sorted_widths = sorted(self._bb_width_history)
        rank = sum(1 for w in sorted_widths if w <= width)
        return rank / len(sorted_widths)

    def _update_vol(self) -> float | None:
        if len(self._closes) < self._vol_period + 1:
            return None

        closes = list(self._closes)
        returns = [
            (closes[i] - closes[i - 1]) / closes[i - 1]
            for i in range(len(closes) - self._vol_period, len(closes))
            if closes[i - 1] != 0
        ]
        if len(returns) < 2:
            return None

        mean_r = sum(returns) / len(returns)
        vol = math.sqrt(sum((r - mean_r) ** 2 for r in returns) / len(returns))
        self._vol_history.append(vol)

        if len(self._vol_history) < 10:
            return None

        sorted_vols = sorted(self._vol_history)
        rank = sum(1 for v in sorted_vols if v <= vol)
        return rank / len(sorted_vols)

    def _classify(
        self,
        adx: float | None,
        bb_pctile: float | None,
        vol_pctile: float | None,
    ) -> MarketRegime:
        if adx is None:
            return MarketRegime.UNKNOWN

        if vol_pctile is not None and vol_pctile >= self._vol_high_pct:
            if bb_pctile is not None and bb_pctile >= self._bb_expand_pct:
                if adx >= self._adx_weak:
                    return MarketRegime.BREAKOUT
                return MarketRegime.HIGH_VOLATILITY
            return MarketRegime.HIGH_VOLATILITY

        if adx >= self._adx_strong:
            return MarketRegime.STRONG_TREND

        if adx >= self._adx_weak:
            return MarketRegime.WEAK_TREND

        if bb_pctile is not None and bb_pctile <= self._bb_squeeze_pct:
            return MarketRegime.RANGING

        return MarketRegime.RANGING

    def summary(self) -> dict[str, Any]:
        return {
            "regime": self._last_regime.value,
            "adx": self._adx,
            "bb_width_history_len": len(self._bb_width_history),
            "vol_history_len": len(self._vol_history),
            "bars_processed": self._bar_count,
            "params": self.get_params(),
        }
