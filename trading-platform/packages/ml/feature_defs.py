"""Canonical feature column definitions — single source of truth.

Three consumption contexts:
  - BASIC_FEATURES: OHLCV + derived (8 cols) — for MLFeatureStrategy + train_ml worker
  - FUTURES_FEATURES: TA indicators (21 cols) — for MLTrainingPipeline
  - CRYPTO_FEATURES: TA + crypto-specific (28 cols) — for CryptoMLPipeline

All training pipelines and runtime strategies should import from here
instead of defining their own column lists.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

BASIC_FEATURES: list[str] = [
    "open", "high", "low", "close", "volume",
    "returns", "volatility", "volume_ratio",
]

FUTURES_FEATURES: list[str] = [
    "rsi", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_middle", "bb_lower", "bb_width",
    "atr", "obv", "vwap",
    "stoch_k", "stoch_d",
    "kc_upper", "kc_middle", "kc_lower",
    "ma_20", "ema_20",
    "returns_1", "returns_5", "vol_20",
]

CRYPTO_FEATURES: list[str] = [
    "rsi", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_middle", "bb_lower", "bb_width",
    "atr", "obv", "vwap",
    "stoch_k", "stoch_d",
    "kc_upper", "kc_middle", "kc_lower",
    "ma_20", "ema_20",
    "returns_1", "returns_5", "returns_10",
    "vol_10", "vol_20", "vol_50",
    "volume_ratio", "taker_ratio",
    "high_low_range", "log_returns",
]


def check_feature_alignment(
    model_features: list[str],
    runtime_features: list[str],
    context: str = "",
) -> bool:
    """Warn if training vs serving features diverge. Returns True if aligned."""
    model_set = set(model_features)
    runtime_set = set(runtime_features)
    missing = model_set - runtime_set
    extra = runtime_set - model_set
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing from runtime: {sorted(missing)}")
        if extra:
            parts.append(f"extra in runtime: {sorted(extra)}")
        logger.warning(
            "Feature alignment issue%s: %s",
            f" ({context})" if context else "",
            "; ".join(parts),
        )
        return False
    if model_features != runtime_features:
        logger.warning(
            "Feature order differs%s — model expects %s",
            f" ({context})" if context else "",
            model_features[:5],
        )
        return False
    return True
