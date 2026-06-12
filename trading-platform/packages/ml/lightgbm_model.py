"""LightGBM 模型实现 — 与 XGBoostModel 同接口，适合大规模特征和快速迭代。"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any, Optional

import numpy as np

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
    "n_estimators": 200,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "num_leaves": 31,
    "min_child_samples": 20,
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
}

try:
    import lightgbm as lgb
except Exception as _e:
    lgb = None  # type: ignore[assignment]
    _LGB_IMPORT_ERROR = _e
else:
    _LGB_IMPORT_ERROR = None

try:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
except Exception as _e:
    accuracy_score = None  # type: ignore[assignment]
    precision_score = recall_score = f1_score = None  # type: ignore[assignment]
    _SKLEARN_IMPORT_ERROR = _e
else:
    _SKLEARN_IMPORT_ERROR = None


def _require_lgb() -> Any:
    if lgb is None:
        raise ImportError(
            f"lightgbm is required. Install with: pip install lightgbm. "
            f"Original error: {_LGB_IMPORT_ERROR!r}"
        ) from _LGB_IMPORT_ERROR
    return lgb


def _require_sklearn() -> None:
    if accuracy_score is None:
        raise ImportError(
            f"scikit-learn is required for metrics. "
            f"Original error: {_SKLEARN_IMPORT_ERROR!r}"
        ) from _SKLEARN_IMPORT_ERROR


class LightGBMModel(BaseMLModel):
    """Wraps ``lightgbm.LGBMClassifier`` with the shared ``BaseMLModel`` API."""

    def __init__(self, meta: MLModelMeta) -> None:
        if meta.framework != MLFramework.LIGHTGBM:
            meta = meta.model_copy(update={"framework": MLFramework.LIGHTGBM})
        merged_hp = {**_DEFAULT_HYPERPARAMS, **meta.hyperparams}
        meta = meta.model_copy(update={"hyperparams": merged_hp})
        super().__init__(meta)
        _require_lgb()

    def _build_estimator(self) -> Any:
        _require_lgb()
        hp = dict(self.meta.hyperparams)
        hp.pop("metric", None)
        return lgb.LGBMClassifier(**hp)

    async def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> TrainResult:
        _require_lgb()
        _require_sklearn()

        self.meta.status = MLModelStatus.TRAINING
        t0 = time.perf_counter()
        self._model = self._build_estimator()

        try:
            fit_kwargs: dict[str, Any] = {}
            if X_val is not None and y_val is not None:
                fit_kwargs["eval_set"] = [(X_val, y_val)]
                fit_kwargs["callbacks"] = [lgb.early_stopping(20, verbose=False)]
            self._model.fit(X_train, y_train, **fit_kwargs)

            y_train_pred = self._model.predict(X_train)
            train_acc = float(accuracy_score(y_train, y_train_pred))

            val_acc: Optional[float] = None
            if X_val is not None and y_val is not None:
                y_val_pred = self._model.predict(X_val)
                val_acc = float(accuracy_score(y_val, y_val_pred))

            duration = time.perf_counter() - t0
            best_iter: Optional[int] = getattr(self._model, "best_iteration_", None)

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
        if not self.is_trained or self._model is None:
            raise RuntimeError("Model is not trained; call train() first.")
        preds = self._model.predict(X)
        proba = self._model.predict_proba(X)
        fi = self.get_feature_importance()
        return PredictResult(
            predictions=[float(x) for x in preds],
            probabilities=[row.tolist() for row in proba],
            feature_importance=fi,
        )

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
        _require_sklearn()
        if not self.is_trained or self._model is None:
            raise RuntimeError("Model is not trained.")
        y_pred = self._model.predict(X_test)
        labels = np.unique(np.concatenate([y_test, y_pred]))
        average = "binary" if len(labels) <= 2 else "weighted"
        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average=average, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average=average, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, average=average, zero_division=0)),
        }

    def save(self, path: str) -> str:
        import json as _json
        if self._model is None:
            raise RuntimeError("No model to save.")
        self._model.booster_.save_model(path)
        self.meta.artifact_path = path
        meta_path = path + ".meta.json"
        with open(meta_path, "w") as f:
            _json.dump({
                "feature_columns": list(self.meta.feature_columns),
                "target_column": self.meta.target_column,
                "metrics": dict(self.meta.metrics),
                "hyperparams": dict(self.meta.hyperparams),
            }, f, indent=2)
        return path

    def load(self, path: str) -> None:
        import json as _json
        from pathlib import Path
        _require_lgb()
        self._model = self._build_estimator()
        booster = lgb.Booster(model_file=path)
        self._model._Booster = booster
        n_models = booster.num_model_per_iteration()
        n_classes = 2 if n_models == 1 else n_models
        self._model._n_classes = n_classes
        self._model.fitted_ = True
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.classes_ = np.arange(n_classes)
        self._model._le = le
        self._model._n_features = booster.num_feature()
        self._model._n_features_in = booster.num_feature()
        self.meta.artifact_path = path
        self.meta.status = MLModelStatus.TRAINED
        meta_path = Path(path + ".meta.json")
        if meta_path.exists():
            with open(meta_path) as f:
                saved = _json.load(f)
            if saved.get("feature_columns"):
                self.meta.feature_columns = saved["feature_columns"]

    def get_feature_importance(self) -> Optional[dict[str, float]]:
        if self._model is None:
            return None
        imps = self._model.feature_importances_
        cols = list(self.meta.feature_columns)
        n = len(imps)
        if not cols or len(cols) != n:
            cols = [f"f{i}" for i in range(n)]
        total = float(np.sum(imps)) or 1.0
        return {cols[i]: float(imps[i]) / total for i in range(n)}
