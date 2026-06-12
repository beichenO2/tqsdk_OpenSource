"""Crypto ML training pipeline — trains LightGBM/XGBoost on BTC OHLCV data.

Analogous to training_pipeline.py but uses CryptoDataLoader instead of
FuturesDataLoader. Generates crypto-specific features including taker ratio,
multi-timeframe momentum, and volume profile analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from crypto.datahub.crypto_loader import CryptoDataLoader
from features.engine import FeatureEngine

from ml.base import MLFramework, MLModelMeta, TrainResult
from ml.xgboost_model import XGBoostModel
from ml.lightgbm_model import LightGBMModel

logger = logging.getLogger(__name__)

CRYPTO_FEATURE_FACTORS = [
    "rsi", "macd", "bollinger_bands", "atr", "obv", "vwap",
    "stochastic", "keltner_channel", "ma", "ema",
]

CRYPTO_FEATURE_COLUMNS = [
    "rsi", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_middle", "bb_lower", "bb_width",
    "atr", "obv", "vwap",
    "stoch_k", "stoch_d",
    "kc_upper", "kc_middle", "kc_lower",
    "ma_20", "ema_20",
    "returns_1", "returns_5", "returns_10",
    "vol_10", "vol_20", "vol_50",
    "volume_ratio", "taker_ratio",
    "high_low_range",
    "log_returns",
]


@dataclass
class CryptoPipelineConfig:
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    start_date: str | None = None
    end_date: str | None = None
    target_lookahead: int = 5
    target_threshold: float = 0.001
    test_ratio: float = 0.2
    val_ratio: float = 0.15
    frameworks: list[str] = field(default_factory=lambda: ["xgboost", "lightgbm"])
    save_dir: str = "models"
    data_dir: str | None = None
    xgboost_params: dict[str, Any] = field(default_factory=dict)
    lightgbm_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class CryptoPipelineReport:
    symbol: str
    timeframe: str
    total_bars: int = 0
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0
    feature_count: int = 0
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    best_model: str = ""
    best_val_score: float = 0.0


def _add_crypto_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add crypto-specific derived features beyond standard TA indicators."""
    df = df.copy()

    if "returns_1" not in df.columns:
        df["returns_1"] = df["close"].pct_change(1)
    if "returns_5" not in df.columns:
        df["returns_5"] = df["close"].pct_change(5)

    df["returns_10"] = df["close"].pct_change(10)
    df["returns_20"] = df["close"].pct_change(20)

    df["vol_10"] = df["returns_1"].rolling(10).std()
    df["vol_20"] = df["returns_1"].rolling(20).std()
    df["vol_50"] = df["returns_1"].rolling(50).std()

    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    if "volume" in df.columns:
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
        df["volume_ma5"] = df["volume"].rolling(5).mean()
        df["volume_ma20"] = df["volume"].rolling(20).mean()
    if "taker_buy_volume" in df.columns and "volume" in df.columns:
        df["taker_ratio"] = df["taker_buy_volume"] / df["volume"].replace(0, np.nan)
        df["taker_ratio"] = df["taker_ratio"].fillna(0.5)
    else:
        df["taker_ratio"] = 0.5

    df["high_low_range"] = (df["high"] - df["low"]) / df["close"]

    for p in [5, 10, 20]:
        df[f"momentum_{p}"] = df["close"] / df["close"].shift(p) - 1

    df["price_vs_ma20"] = df["close"] / df["close"].rolling(20).mean() - 1
    df["price_vs_ma50"] = df["close"] / df["close"].rolling(50).mean() - 1

    if "trades" in df.columns:
        df["avg_trade_size"] = df["volume"] / df["trades"].replace(0, np.nan)
        df["avg_trade_size"] = df["avg_trade_size"].fillna(0)

    return df


def _create_crypto_target(
    df: pd.DataFrame, lookahead: int = 5, threshold: float = 0.001
) -> pd.Series:
    """Ternary classification: 1 (up), 0 (flat), -1 (down) based on future returns."""
    future_ret = df["close"].shift(-lookahead) / df["close"] - 1.0
    target = pd.Series(0, index=df.index, dtype=int)
    target[future_ret > threshold] = 1
    target[future_ret < -threshold] = -1
    target[future_ret.isna()] = np.nan
    return target


class CryptoMLPipeline:
    """End-to-end ML pipeline for crypto data."""

    def __init__(self, config: CryptoPipelineConfig | None = None, **kwargs: Any):
        if config is None:
            config = CryptoPipelineConfig(**kwargs)
        self.config = config
        self.loader = CryptoDataLoader(data_dir=config.data_dir)
        self.feature_engine = FeatureEngine()

    async def run(self) -> CryptoPipelineReport:
        cfg = self.config
        report = CryptoPipelineReport(symbol=cfg.symbol, timeframe=cfg.timeframe)

        logger.info("=== Crypto ML Pipeline: %s %s ===", cfg.symbol, cfg.timeframe)

        bars = self.loader.load(cfg.symbol, cfg.timeframe, cfg.start_date, cfg.end_date)
        if bars.empty:
            logger.error("No data loaded for %s %s", cfg.symbol, cfg.timeframe)
            return report

        report.total_bars = len(bars)
        logger.info("Loaded %d bars", len(bars))

        try:
            bars = self.feature_engine.compute_factors(bars, CRYPTO_FEATURE_FACTORS)
        except Exception as e:
            logger.warning("Some TA factors failed (continuing): %s", e)

        bars = _add_crypto_features(bars)

        bars["target"] = _create_crypto_target(bars, cfg.target_lookahead, cfg.target_threshold)

        available = [c for c in CRYPTO_FEATURE_COLUMNS if c in bars.columns]
        extra = [c for c in bars.columns if c.startswith(("momentum_", "price_vs_", "volume_ma", "avg_trade"))]
        available.extend(extra)
        available = list(dict.fromkeys(available))
        report.feature_count = len(available)

        bars = bars.dropna(subset=available + ["target"])
        bars = bars[bars["target"].isin([-1, 0, 1])]

        if len(bars) < 200:
            logger.error("Too few samples after cleaning: %d", len(bars))
            return report

        X = bars[available].values.astype(np.float64)
        y = bars["target"].values.astype(np.int32)
        y_mapped = (y + 1).astype(np.int32)

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y_mapped, test_size=cfg.test_ratio, shuffle=False,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=cfg.val_ratio, shuffle=False,
        )

        report.train_samples = len(X_train)
        report.val_samples = len(X_val)
        report.test_samples = len(X_test)
        logger.info("Split: train=%d val=%d test=%d", len(X_train), len(X_val), len(X_test))

        save_dir = Path(cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for fw in cfg.frameworks:
            try:
                result = await self._train_one(
                    fw, available, X_train, y_train, X_val, y_val, X_test, y_test, save_dir,
                )
                report.results[fw] = result
                val = result.get("val_accuracy", 0.0)
                if val and val > report.best_val_score:
                    report.best_val_score = val
                    report.best_model = fw
            except Exception as exc:
                logger.exception("Training failed for %s: %s", fw, exc)
                report.results[fw] = {"error": str(exc)}

        logger.info("=== Best: %s (val_acc=%.4f) ===", report.best_model, report.best_val_score)
        return report

    async def _train_one(
        self,
        framework: str,
        feature_columns: list[str],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        save_dir: Path,
    ) -> dict[str, Any]:
        cfg = self.config
        meta = MLModelMeta(
            model_id=f"crypto_{cfg.symbol}_{cfg.timeframe}_{framework}",
            name=f"{framework.upper()} Crypto {cfg.symbol}",
            framework=MLFramework(framework),
            feature_columns=feature_columns,
            target_column="target",
            hyperparams=(cfg.xgboost_params if framework == "xgboost" else cfg.lightgbm_params),
        )

        model: XGBoostModel | LightGBMModel
        if framework == "xgboost":
            model = XGBoostModel(meta)
        else:
            model = LightGBMModel(meta)

        logger.info("Training %s on %s...", framework, cfg.symbol)
        train_result: TrainResult = await model.train(X_train, y_train, X_val, y_val)
        logger.info(
            "%s train=%.4f val=%s (%.1fs)",
            framework, train_result.train_score,
            f"{train_result.val_score:.4f}" if train_result.val_score else "N/A",
            train_result.duration_seconds,
        )

        test_metrics = model.evaluate(X_test, y_test)
        logger.info("%s test: %s", framework, test_metrics)

        model_path = save_dir / f"{meta.model_id}.model"
        model.save(str(model_path))

        fi = model.get_feature_importance()

        return {
            "train_accuracy": train_result.train_score,
            "val_accuracy": train_result.val_score,
            "test_metrics": test_metrics,
            "duration_seconds": train_result.duration_seconds,
            "best_iteration": train_result.best_iteration,
            "model_path": str(model_path),
            "feature_importance": fi,
        }
