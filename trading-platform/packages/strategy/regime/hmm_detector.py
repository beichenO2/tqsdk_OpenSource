"""HMM Market Regime Detector — Hidden Markov Model 市场状态识别。

研究来源:
- RegimeForecast: "HMM for Market Regimes — A Practical Guide"
- PyQuantLab: "GMM Regime-Switching Momentum"
- Abdullah-BA/RegimeSwitchingMomentumStrategy

检测 3 种市场状态:
- 0: 低波动趋势 (Risk-On / Trending)
- 1: 高波动震荡 (Risk-Off / Choppy)
- 2: 中等波动 (Normal / Transitional)
"""

from __future__ import annotations

import logging
import warnings
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    TRENDING = "trending"
    CHOPPY = "choppy"
    NORMAL = "normal"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class RegimeState:
    regime: MarketRegime
    confidence: float
    volatility: float
    trend_strength: float


class HMMRegimeDetector:
    """Train a Gaussian HMM on rolling returns to detect market regimes.

    Falls back to volatility-based heuristic when HMM training fails
    or insufficient data is available.
    """

    def __init__(
        self,
        n_states: int = 3,
        lookback: int = 252,
        retrain_interval: int = 63,
        min_observations: int = 100,
    ) -> None:
        self.n_states = n_states
        self.lookback = lookback
        self.retrain_interval = retrain_interval
        self.min_observations = min_observations

        self._returns: deque[float] = deque(maxlen=lookback + 50)
        self._prices: deque[float] = deque(maxlen=lookback + 50)
        self._bar_count = 0
        self._model: Any = None
        self._state_map: dict[int, MarketRegime] = {}
        self._last_regime = RegimeState(MarketRegime.UNKNOWN, 0.0, 0.0, 0.0)

    def update(self, price: float) -> RegimeState:
        """Feed a new price, retrain periodically, return current regime."""
        self._prices.append(price)
        if len(self._prices) >= 2 and self._prices[-2] != 0:
            ret = self._prices[-1] / self._prices[-2] - 1
            self._returns.append(ret)

        self._bar_count += 1

        if len(self._returns) < self.min_observations:
            return self._fallback_regime()

        if self._model is None or self._bar_count % self.retrain_interval == 0:
            self._train()

        if self._model is not None:
            return self._predict()

        return self._fallback_regime()

    def _train(self) -> None:
        """Fit GaussianHMM on recent returns."""
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            logger.debug("hmmlearn not installed, using fallback")
            return

        returns = np.array(list(self._returns)[-self.lookback:]).reshape(-1, 1)
        if len(returns) < self.min_observations:
            return

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = GaussianHMM(
                    n_components=self.n_states,
                    covariance_type="full",
                    n_iter=100,
                    random_state=42,
                    tol=0.01,
                )
                model.fit(returns)
                self._model = model
                self._map_states(returns)
            except Exception as e:
                logger.debug("HMM training failed: %s", e)
                self._model = None

    def _map_states(self, returns: np.ndarray) -> None:
        """Map HMM state indices to semantic regime labels by volatility ordering."""
        if self._model is None:
            return

        state_vols = []
        for i in range(self.n_states):
            vol = float(np.sqrt(self._model.covars_[i][0][0]))
            mean_ret = float(self._model.means_[i][0])
            state_vols.append((i, vol, mean_ret))

        state_vols.sort(key=lambda x: x[1])

        if self.n_states == 3:
            self._state_map = {
                state_vols[0][0]: MarketRegime.TRENDING,
                state_vols[1][0]: MarketRegime.NORMAL,
                state_vols[2][0]: MarketRegime.CHOPPY,
            }
        elif self.n_states == 2:
            self._state_map = {
                state_vols[0][0]: MarketRegime.TRENDING,
                state_vols[1][0]: MarketRegime.CHOPPY,
            }
        else:
            for i, (idx, _, _) in enumerate(state_vols):
                if i < self.n_states // 3:
                    self._state_map[idx] = MarketRegime.TRENDING
                elif i < 2 * self.n_states // 3:
                    self._state_map[idx] = MarketRegime.NORMAL
                else:
                    self._state_map[idx] = MarketRegime.CHOPPY

    def _predict(self) -> RegimeState:
        """Predict current regime from most recent returns."""
        returns = np.array(list(self._returns)[-self.lookback:]).reshape(-1, 1)
        try:
            probs = self._model.predict_proba(returns)
            current_probs = probs[-1]
            state_idx = int(np.argmax(current_probs))
            confidence = float(current_probs[state_idx])
            regime = self._state_map.get(state_idx, MarketRegime.UNKNOWN)

            recent = list(self._returns)[-20:]
            vol = float(np.std(recent)) if len(recent) > 1 else 0.0
            trend = abs(float(np.mean(recent))) / vol if vol > 0 else 0.0

            self._last_regime = RegimeState(regime, confidence, vol, trend)
            return self._last_regime
        except Exception:
            return self._fallback_regime()

    def _fallback_regime(self) -> RegimeState:
        """Simple volatility-based regime detection when HMM is unavailable."""
        if len(self._returns) < 20:
            return RegimeState(MarketRegime.UNKNOWN, 0.0, 0.0, 0.0)

        recent = list(self._returns)[-20:]
        vol = float(np.std(recent))
        mean_ret = float(np.mean(recent))
        trend = abs(mean_ret) / vol if vol > 0 else 0.0

        long_vol = float(np.std(list(self._returns)[-60:])) if len(self._returns) >= 60 else vol
        vol_ratio = vol / long_vol if long_vol > 0 else 1.0

        if vol_ratio < 0.7 and trend > 0.3:
            regime = MarketRegime.TRENDING
            confidence = min(0.5 + trend * 0.3, 0.9)
        elif vol_ratio > 1.3:
            regime = MarketRegime.CHOPPY
            confidence = min(0.5 + (vol_ratio - 1) * 0.3, 0.9)
        else:
            regime = MarketRegime.NORMAL
            confidence = 0.5

        self._last_regime = RegimeState(regime, confidence, vol, trend)
        return self._last_regime

    @property
    def current_regime(self) -> RegimeState:
        return self._last_regime

    def to_btc_regime(self) -> str:
        """Map HMM regime to BTC regime_detector vocabulary for cross-system use."""
        _MAP = {
            MarketRegime.TRENDING: "strong_trend",
            MarketRegime.NORMAL: "weak_trend",
            MarketRegime.CHOPPY: "ranging",
            MarketRegime.UNKNOWN: "unknown",
        }
        return _MAP.get(self._last_regime.regime, "unknown")
