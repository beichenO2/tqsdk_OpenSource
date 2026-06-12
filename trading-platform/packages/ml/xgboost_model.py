"""XGBoost 模型实现 — 用于价格方向预测和特征重要性分析。"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

from .base import (
    BaseMLModel,
    MLFramework,
    MLModelMeta,
    MLModelStatus,
    PredictResult,
    TrainResult,
)

_DEFAULT_HYPERPARAMS: dict[str, Any] = {
    "max_depth": 6,
    "n_estimators": 100,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "objective": "binary:logistic",
}

try:
    import xgboost as xgb
except Exception as _e:  # pragma: no cover - optional / broken native lib
    xgb = None  # type: ignore[assignment]
    _XGBOOST_IMPORT_ERROR = _e
else:
    _XGBOOST_IMPORT_ERROR = None

try:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )
except Exception as _e:  # pragma: no cover - optional dependency
    accuracy_score = None  # type: ignore[assignment]
    precision_score = None  # type: ignore[assignment]
    recall_score = None  # type: ignore[assignment]
    f1_score = None  # type: ignore[assignment]
    _SKLEARN_METRICS_IMPORT_ERROR = _e
else:
    _SKLEARN_METRICS_IMPORT_ERROR = None


def _require_xgboost() -> Any:
    if xgb is None:
        raise ImportError(
            "The 'xgboost' package is required for XGBoostModel. "
            "Install with: pip install xgboost. "
            "On macOS, if the native library fails to load, try: brew install libomp. "
            f"Original error: {_XGBOOST_IMPORT_ERROR!r}"
        ) from _XGBOOST_IMPORT_ERROR
    return xgb


def _require_sklearn_metrics() -> None:
    if accuracy_score is None:
        raise ImportError(
            "The 'scikit-learn' package is required for training metrics and evaluate(). "
            "Install it with: pip install scikit-learn. "
            f"Original error: {_SKLEARN_METRICS_IMPORT_ERROR!r}"
        ) from _SKLEARN_METRICS_IMPORT_ERROR


class XGBoostModel(BaseMLModel):
    """Wraps ``xgboost.XGBClassifier`` with the shared ``BaseMLModel`` API."""

    def __init__(self, meta: MLModelMeta) -> None:
        if meta.framework != MLFramework.XGBOOST:
            meta = meta.model_copy(update={"framework": MLFramework.XGBOOST})
        merged_hp = {**_DEFAULT_HYPERPARAMS, **meta.hyperparams}
        meta = meta.model_copy(update={"hyperparams": merged_hp})
        super().__init__(meta)
        _require_xgboost()

    def _build_estimator(self) -> Any:
        _require_xgboost()
        return xgb.XGBClassifier(**self.meta.hyperparams)

    async def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> TrainResult:
        _require_xgboost()
        _require_sklearn_metrics()

        self.meta.status = MLModelStatus.TRAINING
        t0 = time.perf_counter()

        self._model = self._build_estimator()

        try:
            if X_val is not None and y_val is not None:
                fit_kwargs: dict[str, Any] = {
                    "eval_set": [(X_val, y_val)],
                    "verbose": False,
                }
                try:
                    self._model.set_params(early_stopping_rounds=10)
                except (TypeError, ValueError):
                    fit_kwargs["early_stopping_rounds"] = 10
                self._model.fit(X_train, y_train, **fit_kwargs)
            else:
                self._model.fit(X_train, y_train, verbose=False)

            y_train_pred = self._model.predict(X_train)
            train_acc = float(accuracy_score(y_train, y_train_pred))

            val_acc: Optional[float] = None
            if X_val is not None and y_val is not None:
                y_val_pred = self._model.predict(X_val)
                val_acc = float(accuracy_score(y_val, y_val_pred))

            duration = time.perf_counter() - t0
            best_iter: Optional[int] = None
            if hasattr(self._model, "best_iteration") and self._model.best_iteration is not None:
                best_iter = int(self._model.best_iteration)

            self.meta.status = MLModelStatus.TRAINED
            self.meta.trained_at = datetime.now(UTC)
            self.meta.metrics = {
                "train_accuracy": train_acc,
                **({"val_accuracy": val_acc} if val_acc is not None else {}),
            }

            return TrainResult(
                train_score=train_acc,
                val_score=val_acc,
                metrics=dict(self.meta.metrics),
                duration_seconds=duration,
                best_iteration=best_iter,
            )
        except Exception:
            self.meta.status = MLModelStatus.FAILED
            raise

    def predict(self, X: np.ndarray) -> PredictResult:
        _require_xgboost()
        if not self.is_trained or self._model is None:
            raise RuntimeError("Model is not trained; call train() or load() before predict().")

        preds = self._model.predict(X)
        proba = self._model.predict_proba(X)
        fi = self.get_feature_importance()

        return PredictResult(
            predictions=[float(x) for x in preds],
            probabilities=[row.tolist() for row in proba],
            feature_importance=fi,
        )

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
        _require_sklearn_metrics()
        if not self.is_trained or self._model is None:
            raise RuntimeError("Model is not trained; call train() or load() before evaluate().")

        y_pred = self._model.predict(X_test)
        labels = np.unique(np.concatenate([y_test, y_pred]))
        average = "binary" if len(labels) <= 2 else "weighted"

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average=average, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average=average, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, average=average, zero_division=0)),
        }
        return metrics

    def save(self, path: str) -> str:
        _require_xgboost()
        import json
        from pathlib import Path

        if self._model is None:
            raise RuntimeError("No model to save; train or load first.")
        self._model.save_model(path)
        self.meta.artifact_path = path

        meta_path = Path(path).with_suffix(".meta.json")
        meta_data = {
            "model_id": self.meta.model_id,
            "framework": self.meta.framework.value if hasattr(self.meta.framework, 'value') else str(self.meta.framework),
            "feature_columns": list(self.meta.feature_columns),
            "target_column": self.meta.target_column,
            "hyperparams": dict(self.meta.hyperparams),
        }
        if self.meta.trained_at:
            meta_data["trained_at"] = self.meta.trained_at.isoformat()
        if self.meta.metrics:
            meta_data["metrics"] = dict(self.meta.metrics)
        with open(meta_path, "w") as f:
            json.dump(meta_data, f, indent=2, default=str)

        return path

    def load(self, path: str) -> None:
        _require_xgboost()
        import json
        from pathlib import Path

        self._model = self._build_estimator()
        self._model.load_model(path)
        self.meta.artifact_path = path
        self.meta.status = MLModelStatus.TRAINED

        meta_path = Path(path).with_suffix(".meta.json")
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta_data = json.load(f)
                if "feature_columns" in meta_data:
                    self.meta.feature_columns = meta_data["feature_columns"]
                if "hyperparams" in meta_data:
                    self.meta.hyperparams = meta_data["hyperparams"]
            except Exception as exc:
                logger.warning("Failed to load XGBoost meta from %s: %s", meta_path, exc)

    def get_feature_importance(self) -> Optional[dict[str, float]]:
        if self._model is None:
            return None
        imps = self._model.feature_importances_
        cols = list(self.meta.feature_columns)
        n = len(imps)
        if not cols or len(cols) != n:
            cols = [f"f{i}" for i in range(n)]
        return {cols[i]: float(imps[i]) for i in range(n)}
