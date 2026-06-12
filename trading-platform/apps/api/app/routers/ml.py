"""ML 模型训练与管理 API — 训练、列表、预测接口。"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from core.exceptions import MLUnavailableError, ModelNotFoundError

logger = logging.getLogger(__name__)

_ML_IMPORT_ERROR: str | None = None
try:
    from ml.base import MLFramework, MLModelMeta, MLModelStatus
    from ml.xgboost_model import XGBoostModel
except ImportError as exc:
    _ML_IMPORT_ERROR = str(exc)
    XGBoostModel = None  # type: ignore[misc, assignment]
    MLModelMeta = None  # type: ignore[misc, assignment]
    MLFramework = None  # type: ignore[misc, assignment]
    MLModelStatus = None  # type: ignore[misc, assignment]

router = APIRouter(prefix="/ml", tags=["ml"])

MODEL_DIR = os.environ.get("ML_MODEL_DIR", "models")

FEATURE_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "returns", "volatility", "volume_ratio",
]

_loaded_models: dict[str, Any] = {}


class TrainRequest(BaseModel):
    n_bars: int = Field(default=2000, ge=200, le=50000, description="合成数据 K 线数量")
    max_depth: int = Field(default=6, ge=1, le=15)
    n_estimators: int = Field(default=100, ge=10, le=1000)
    learning_rate: float = Field(default=0.1, gt=0, le=1.0)
    subsample: float = Field(default=0.8, gt=0, le=1.0)
    train_ratio: float = Field(default=0.6, gt=0.1, lt=1.0)
    val_ratio: float = Field(default=0.2, gt=0.0, lt=0.5)
    volatility_window: int = Field(default=20, ge=5, le=100)
    volume_ma_period: int = Field(default=20, ge=5, le=100)


class TrainResponse(BaseModel):
    model_id: str
    model_path: str
    report_path: str
    train_accuracy: float
    val_accuracy: Optional[float] = None
    test_metrics: dict[str, float]
    feature_importance: Optional[dict[str, float]] = None
    duration_seconds: float
    data_info: dict[str, Any]


class ModelInfo(BaseModel):
    model_id: str
    model_path: str
    report: Optional[dict[str, Any]] = None


class PredictRequest(BaseModel):
    model_id: str = Field(..., description="模型 ID (文件名不含扩展名)")
    features: dict[str, float] = Field(
        ...,
        description="特征字典: open, high, low, close, volume, returns, volatility, volume_ratio",
    )


class PredictResponse(BaseModel):
    prediction: int = Field(description="0=跌, 1=涨")
    probability_up: float
    probability_down: float
    feature_importance: Optional[dict[str, float]] = None


def _require_ml() -> None:
    if _ML_IMPORT_ERROR is not None:
        raise MLUnavailableError(
            "ML 依赖未加载",
            detail={"import_error": _ML_IMPORT_ERROR},
        )


def _load_training_ohlcv(n_bars: int) -> dict[str, Any]:
    """Load real OHLCV data from parquet for ML training."""
    import numpy as np
    import pandas as pd
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    search_dirs = [
        repo / "data" / "futures_cache",
        repo / "data" / "crypto_cache",
        repo / ".cache" / "bars",
    ]

    for d in search_dirs:
        if not d.exists():
            continue
        for fp in sorted(d.glob("**/*.parquet")):
            try:
                df = pd.read_parquet(fp)
                if not {"open", "high", "low", "close", "volume"}.issubset(df.columns):
                    continue
                if len(df) < n_bars:
                    continue
                df = df.tail(n_bars).reset_index(drop=True)
                return {
                    "open": df["open"].to_numpy(dtype=np.float64),
                    "high": df["high"].to_numpy(dtype=np.float64),
                    "low": df["low"].to_numpy(dtype=np.float64),
                    "close": df["close"].to_numpy(dtype=np.float64),
                    "volume": df["volume"].to_numpy(dtype=np.float64),
                }
            except Exception as exc:
                logger.debug("Skipping data file %s: %s", fp, exc)
                continue

    from core.exceptions import DataNotAvailableError
    raise DataNotAvailableError(
        f"No parquet file with >= {n_bars} bars found in {[str(d) for d in search_dirs]}"
    )


def _compute_features(ohlcv: dict[str, Any], vol_w: int, vma_p: int) -> Any:
    import numpy as np

    closes = ohlcv["close"]
    volumes = ohlcv["volume"]
    n = len(closes)

    returns = np.zeros(n)
    returns[1:] = np.diff(closes) / np.where(closes[:-1] != 0, closes[:-1], 1.0)

    volatility = np.zeros(n)
    for i in range(vol_w + 1, n):
        volatility[i] = np.std(returns[i - vol_w: i])

    volume_ratio = np.ones(n)
    for i in range(vma_p, n):
        vma = np.mean(volumes[i - vma_p: i])
        volume_ratio[i] = volumes[i] / vma if vma > 0 else 1.0

    return np.column_stack([
        ohlcv["open"], ohlcv["high"], ohlcv["low"], closes, volumes,
        returns, volatility, volume_ratio,
    ])


@router.post("/train", response_model=TrainResponse)
async def train_model(req: TrainRequest) -> TrainResponse:
    _require_ml()
    import numpy as np

    assert XGBoostModel is not None
    assert MLModelMeta is not None
    assert MLFramework is not None

    ohlcv = _load_training_ohlcv(req.n_bars)
    warmup = max(req.volatility_window, req.volume_ma_period) + 2
    X_all = _compute_features(ohlcv, req.volatility_window, req.volume_ma_period)

    closes = ohlcv["close"]
    y_all = np.zeros(len(closes), dtype=np.int32)
    y_all[:-1] = (closes[1:] > closes[:-1]).astype(np.int32)

    X_all = X_all[warmup:-1]
    y_all = y_all[warmup:-1]
    n = len(X_all)

    n_train = int(n * req.train_ratio)
    n_val = int(n * req.val_ratio)
    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_val, y_val = X_all[n_train:n_train + n_val], y_all[n_train:n_train + n_val]
    X_test, y_test = X_all[n_train + n_val:], y_all[n_train + n_val:]

    model_id = f"xgb_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    hyperparams = {
        "max_depth": req.max_depth,
        "n_estimators": req.n_estimators,
        "learning_rate": req.learning_rate,
        "subsample": req.subsample,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "objective": "binary:logistic",
    }

    meta = MLModelMeta(
        model_id=model_id,
        name="XGBoost Price Direction",
        framework=MLFramework.XGBOOST,
        feature_columns=list(FEATURE_COLUMNS),
        target_column="direction",
        hyperparams=hyperparams,
    )

    model = XGBoostModel(meta)
    train_result = await model.train(X_train, y_train, X_val, y_val)
    test_metrics = model.evaluate(X_test, y_test)

    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"{model_id}.json")
    model.save(model_path)

    fi = model.get_feature_importance()

    data_info = {
        "source": f"parquet ({req.n_bars} bars)",
        "total_samples": n,
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
    }

    report = {
        "model_id": model_id,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "hyperparams": hyperparams,
        "data_info": data_info,
        "train_result": train_result.model_dump(),
        "test_metrics": test_metrics,
        "feature_columns": FEATURE_COLUMNS,
    }
    report_path = os.path.join(MODEL_DIR, f"{model_id}_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    _loaded_models[model_id] = model

    return TrainResponse(
        model_id=model_id,
        model_path=model_path,
        report_path=report_path,
        train_accuracy=train_result.train_score,
        val_accuracy=train_result.val_score,
        test_metrics=test_metrics,
        feature_importance=fi,
        duration_seconds=train_result.duration_seconds,
        data_info=data_info,
    )


@router.get("/models", response_model=list[ModelInfo])
async def list_models() -> list[ModelInfo]:
    model_dir = Path(MODEL_DIR)
    if not model_dir.exists():
        return []

    models: list[ModelInfo] = []
    for path in sorted(model_dir.glob("*.json")):
        if path.name.endswith("_report.json"):
            continue
        model_id = path.stem
        report_path = path.with_name(f"{model_id}_report.json")
        report_data = None
        if report_path.exists():
            try:
                with open(report_path) as f:
                    report_data = json.load(f)
            except Exception as exc:
                logger.warning("Failed to load report %s: %s", report_path.name, exc)
        models.append(ModelInfo(model_id=model_id, model_path=str(path), report=report_data))

    return models


@router.get("/models/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str) -> ModelInfo:
    model_path = Path(MODEL_DIR) / f"{model_id}.json"
    if not model_path.exists():
        raise ModelNotFoundError(f"模型 {model_id} 不存在")

    report_path = model_path.with_name(f"{model_id}_report.json")
    report_data = None
    if report_path.exists():
        try:
            with open(report_path) as f:
                report_data = json.load(f)
        except Exception as exc:
            logger.warning("Failed to load report %s: %s", report_path.name, exc)

    return ModelInfo(model_id=model_id, model_path=str(model_path), report=report_data)


@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    _require_ml()
    import numpy as np

    assert XGBoostModel is not None
    assert MLModelMeta is not None
    assert MLFramework is not None

    if req.model_id in _loaded_models:
        model = _loaded_models[req.model_id]
    else:
        model_path = Path(MODEL_DIR) / f"{req.model_id}.json"
        if not model_path.exists():
            raise ModelNotFoundError(f"模型 {req.model_id} 不存在")

        report_path = model_path.with_name(f"{req.model_id}_report.json")
        hp: dict[str, Any] = {}
        if report_path.exists():
            with open(report_path) as f:
                hp = json.load(f).get("hyperparams", {})

        meta = MLModelMeta(
            model_id=req.model_id,
            name="XGBoost Price Direction",
            framework=MLFramework.XGBOOST,
            feature_columns=list(FEATURE_COLUMNS),
            target_column="direction",
            hyperparams=hp,
        )
        model = XGBoostModel(meta)
        model.load(str(model_path))
        _loaded_models[req.model_id] = model

    row = [req.features.get(col, 0.0) for col in FEATURE_COLUMNS]
    X = np.array([row], dtype=np.float64)
    result = model.predict(X)

    pred = int(result.predictions[0])
    proba = result.probabilities
    if proba and len(proba[0]) >= 2:
        p_down, p_up = float(proba[0][0]), float(proba[0][1])
    else:
        p_up = float(result.predictions[0])
        p_down = 1.0 - p_up

    return PredictResponse(
        prediction=pred,
        probability_up=p_up,
        probability_down=p_down,
        feature_importance=result.feature_importance,
    )
