"""ML 特征策略 — 使用训练好的 ML 模型生成交易信号。"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

from strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig
from strategy.registry import auto_register

from .base import BaseMLModel
from .feature_defs import BASIC_FEATURES, check_feature_alignment

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_ORDER = tuple(BASIC_FEATURES)

DEFAULT_PARAMS: dict[str, Any] = {
    "long_prob_threshold": 0.55,
    "short_prob_threshold": 0.55,
    "volatility_window": 20,
    "volume_ma_period": 20,
}


@auto_register("ml_feature")
class MLFeatureStrategy(BaseStrategy):
    """用预训练 ``BaseMLModel`` 在每条 K 线上构造特征并输出方向信号。"""

    def __init__(
        self,
        config: StrategyConfig,
        model: BaseMLModel | None = None,
    ) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config = config.model_copy(update={"params": merged})
        super().__init__(config)

        self._ml_model = model or config.params.get("ml_model")
        if self._ml_model is None:
            logger.warning(
                "MLFeatureStrategy started without a BaseMLModel; "
                "pass model=... or config.params['ml_model']."
            )
        elif self._ml_model.meta.feature_columns:
            check_feature_alignment(
                self._ml_model.meta.feature_columns,
                list(DEFAULT_FEATURE_ORDER),
                context="MLFeatureStrategy init",
            )

        self._close_history: dict[str, deque[float]] = {}
        self._volume_history: dict[str, deque[float]] = {}

    def _ensure_buffers(self, symbol: str) -> None:
        w = max(
            self.get_param("volatility_window", 20),
            self.get_param("volume_ma_period", 20),
        ) + 5
        if symbol not in self._close_history:
            self._close_history[symbol] = deque(maxlen=w + 10)
            self._volume_history[symbol] = deque(maxlen=w + 10)

    def _rolling_volatility(self, closes: deque[float], window: int) -> float:
        if len(closes) < window + 1:
            return 0.0
        arr = np.array(list(closes)[-window - 1 :], dtype=np.float64)
        rets = np.diff(arr) / np.where(arr[:-1] != 0, arr[:-1], 1.0)
        return float(np.std(rets)) if len(rets) else 0.0

    def _feature_vector(self, symbol: str, bar: dict[str, Any]) -> dict[str, float]:
        """Assume ``close``/``volume`` for this bar are already appended to deques."""
        o = float(bar.get("open", bar["close"]))
        h = float(bar["high"])
        l = float(bar["low"])
        c = float(bar["close"])
        v = float(bar.get("volume", 0.0))

        closes = self._close_history[symbol]
        vols = self._volume_history[symbol]

        if len(closes) >= 2:
            prev_c = float(closes[-2])
            returns = (c - prev_c) / prev_c if prev_c else 0.0
        else:
            returns = 0.0

        vw = self.get_param("volatility_window", 20)
        volatility = self._rolling_volatility(closes, vw)

        vma_p = self.get_param("volume_ma_period", 20)
        vol_list = list(vols)
        if len(vol_list) >= vma_p:
            vma = sum(vol_list[-vma_p:]) / vma_p
            volume_ratio = v / vma if vma > 0 else 1.0
        else:
            volume_ratio = 1.0

        return {
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v,
            "returns": returns,
            "volatility": volatility,
            "volume_ratio": volume_ratio,
        }

    def _vector_for_model(self, feats: dict[str, float]) -> np.ndarray:
        model = self._ml_model
        if model is None:
            return np.zeros((1, len(DEFAULT_FEATURE_ORDER)), dtype=np.float64)
        order = list(model.meta.feature_columns) if model.meta.feature_columns else list(DEFAULT_FEATURE_ORDER)
        row = [feats.get(name, 0.0) for name in order]
        return np.array([row], dtype=np.float64)

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._ensure_buffers(symbol)

        self._close_history[symbol].append(float(bar["close"]))
        self._volume_history[symbol].append(float(bar.get("volume", 0.0)))

        feats = self._feature_vector(symbol, bar)

        if self._ml_model is None:
            return []

        if not self._ml_model.is_trained:
            logger.debug("ML model not trained yet; skipping bar for %s", symbol)
            return []

        X = self._vector_for_model(feats)
        try:
            result = self._ml_model.predict(X)
        except Exception as exc:
            logger.exception("ML predict failed: %s", exc)
            return []

        proba = result.probabilities
        if not proba or len(proba[0]) < 2:
            p_long = float(result.predictions[0])
            p_short = 1.0 - p_long
        else:
            p0, p1 = float(proba[0][0]), float(proba[0][1])
            p_short, p_long = p0, p1

        long_thr = float(self.get_param("long_prob_threshold", 0.55))
        short_thr = float(self.get_param("short_prob_threshold", 0.55))

        signals: list[Signal] = []
        price = float(bar["close"])

        if p_long >= long_thr and p_long >= p_short:
            sig = Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_type=SignalType.LONG_ENTRY,
                strength=min(p_long, 1.0),
                price=price,
                reason=f"ML long (p_up={p_long:.3f})",
                metadata={"probabilities": proba[0] if proba else [], "features": feats},
            )
            signals.append(sig)
            self.record_signal(sig)
        elif p_short >= short_thr:
            sig = Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_type=SignalType.SHORT_ENTRY,
                strength=min(p_short, 1.0),
                price=price,
                reason=f"ML short (p_down={p_short:.3f})",
                metadata={"probabilities": proba[0] if proba else [], "features": feats},
            )
            signals.append(sig)
            self.record_signal(sig)

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        for sym in self.config.symbols:
            b = market_data.get(sym)
            if b:
                out.extend(await self.on_bar(sym, b))
        return out
