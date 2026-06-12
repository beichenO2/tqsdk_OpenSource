"""端到端 ML 训练管线 — 数据加载 → 特征计算 → 训练 → 评估 → 保存。

用法:
    pipeline = MLTrainingPipeline(instrument="rb", timeframe="5m")
    report = await pipeline.run()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from datahub.futures_loader import FuturesDataLoader
from features.engine import FeatureEngine

from .base import MLFramework, MLModelMeta, TrainResult
from .xgboost_model import XGBoostModel
from .lightgbm_model import LightGBMModel

logger = logging.getLogger(__name__)

from .feature_defs import FUTURES_FEATURES

FEATURE_FACTORS = [
    "rsi", "macd", "bollinger_bands", "atr", "obv", "vwap",
    "stochastic", "keltner_channel", "ma", "ema",
]

FEATURE_COLUMNS = FUTURES_FEATURES


@dataclass
class PipelineConfig:
    instrument: str = "rb"
    timeframe: str = "5m"
    start_date: str | None = "2024-01-01"
    end_date: str | None = "2024-12-31"
    target_lookahead: int = 5
    test_ratio: float = 0.2
    val_ratio: float = 0.15
    frameworks: list[str] = field(default_factory=lambda: ["xgboost", "lightgbm"])
    save_dir: str = "models"
    cache_dir: str | None = ".cache/bars"
    xgboost_params: dict[str, Any] = field(default_factory=dict)
    lightgbm_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineReport:
    instrument: str
    timeframe: str
    total_bars: int = 0
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0
    feature_count: int = 0
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    best_model: str = ""
    best_val_score: float = 0.0


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add return and volatility features not covered by factor registry."""
    df = df.copy()
    df["returns_1"] = df["close"].pct_change(1)
    df["returns_5"] = df["close"].pct_change(5)
    df["vol_20"] = df["returns_1"].rolling(20).std()
    return df


def _create_target(df: pd.DataFrame, lookahead: int = 5) -> pd.Series:
    """Binary classification: 1 if future return is positive, 0 otherwise."""
    future_ret = df["close"].shift(-lookahead) / df["close"] - 1.0
    return (future_ret > 0).astype(int)


class MLTrainingPipeline:
    """Orchestrates the full ML training pipeline with real futures data."""

    def __init__(self, config: PipelineConfig | None = None, **kwargs: Any):
        if config is None:
            config = PipelineConfig(**kwargs)
        self.config = config
        self.loader = FuturesDataLoader()
        self.feature_engine = FeatureEngine()

    async def run(self) -> PipelineReport:
        cfg = self.config
        report = PipelineReport(instrument=cfg.instrument, timeframe=cfg.timeframe)

        logger.info("=== ML Pipeline: %s %s ===", cfg.instrument, cfg.timeframe)

        # 1. Load data
        bars = self.loader.load_main_contract_bars(
            cfg.instrument,
            cfg.timeframe,
            cfg.start_date,
            cfg.end_date,
            cache_dir=cfg.cache_dir,
        )
        if bars.empty:
            logger.error("No data loaded for %s", cfg.instrument)
            return report

        report.total_bars = len(bars)
        logger.info("Loaded %d bars", len(bars))

        # 2. Compute features
        bars = self.feature_engine.compute_factors(bars, FEATURE_FACTORS)
        bars = _add_derived_features(bars)

        # 3. Create target
        bars["target"] = _create_target(bars, cfg.target_lookahead)

        available_features = [c for c in FEATURE_COLUMNS if c in bars.columns]
        report.feature_count = len(available_features)

        # 4. Drop NaN and split
        bars = bars.dropna(subset=available_features + ["target"])
        if len(bars) < 100:
            logger.error("Too few samples after cleaning: %d", len(bars))
            return report

        X = bars[available_features].values.astype(np.float64)
        y = bars["target"].values.astype(np.int32)

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=cfg.test_ratio, shuffle=False,
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

        # 5. Train each framework
        for fw in cfg.frameworks:
            try:
                result = await self._train_one(
                    fw, available_features, X_train, y_train, X_val, y_val, X_test, y_test, save_dir,
                )
                report.results[fw] = result
                val = result.get("val_accuracy", 0.0)
                if val > report.best_val_score:
                    report.best_val_score = val
                    report.best_model = fw
            except Exception as exc:
                logger.exception("Training failed for %s: %s", fw, exc)
                report.results[fw] = {"error": str(exc)}

        logger.info("=== Best model: %s (val_acc=%.4f) ===", report.best_model, report.best_val_score)
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
            model_id=f"{cfg.instrument}_{cfg.timeframe}_{framework}",
            name=f"{framework.upper()} {cfg.instrument}",
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

        logger.info("Training %s...", framework)
        train_result: TrainResult = await model.train(X_train, y_train, X_val, y_val)
        logger.info(
            "%s train_acc=%.4f val_acc=%s duration=%.1fs",
            framework,
            train_result.train_score,
            f"{train_result.val_score:.4f}" if train_result.val_score else "N/A",
            train_result.duration_seconds,
        )

        test_metrics = model.evaluate(X_test, y_test)
        logger.info("%s test metrics: %s", framework, test_metrics)

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
