"""ML 模型训练与管理 API — 训练、列表、预测接口。"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from core.exceptions import MLUnavailableError, ModelNotFoundError, TradingPlatformError

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
ML_TRAIN_TIMEOUT_S = int(os.environ.get("ML_TRAIN_TIMEOUT_S", "600"))

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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _worker_env() -> dict[str, str]:
    repo = _repo_root()
    paths = [
        str(repo),
        str(repo / "apps" / "api"),
        str(repo / "packages" / "core"),
        str(repo / "packages"),
    ]
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def _build_train_worker_cmd(req: TrainRequest) -> list[str]:
    return [
        sys.executable,
        "-m",
        "apps.worker.train_ml",
        "--parquet",
        "--api-json",
        "--framework",
        "xgboost",
        "--bars",
        str(req.n_bars),
        "--model-dir",
        MODEL_DIR,
        "--train-ratio",
        str(req.train_ratio),
        "--val-ratio",
        str(req.val_ratio),
        "--max-depth",
        str(req.max_depth),
        "--n-estimators",
        str(req.n_estimators),
        "--lr",
        str(req.learning_rate),
        "--subsample",
        str(req.subsample),
        "--volatility-window",
        str(req.volatility_window),
        "--volume-ma-period",
        str(req.volume_ma_period),
    ]


async def _run_train_subprocess(req: TrainRequest) -> dict[str, Any]:
    """Spawn isolated worker subprocess — keeps OpenMP runtimes out of API process."""
    cmd = _build_train_worker_cmd(req)
    env = _worker_env()
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        cwd=str(_repo_root()),
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=ML_TRAIN_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise TradingPlatformError(
            f"ML training timed out after {ML_TRAIN_TIMEOUT_S}s",
            code="ML_TRAIN_TIMEOUT",
            status_code=504,
        )

    if proc.returncode != 0:
        stderr_text = stderr.decode("utf-8", errors="replace")
        raise TradingPlatformError(
            "ML training worker failed",
            code="ML_TRAIN_FAILED",
            status_code=500,
            detail={"stderr": stderr_text[-2000:]},
        )

    try:
        return json.loads(stdout.decode("utf-8"))
    except json.JSONDecodeError as exc:
        stdout_text = stdout.decode("utf-8", errors="replace")
        raise TradingPlatformError(
            "ML training worker returned invalid JSON",
            code="ML_TRAIN_FAILED",
            status_code=500,
            detail={"stdout_tail": stdout_text[-2000:], "parse_error": str(exc)},
        ) from exc


@router.post("/train", response_model=TrainResponse)
async def train_model(req: TrainRequest) -> TrainResponse:
    _require_ml()
    result = await _run_train_subprocess(req)
    return TrainResponse(**result)


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
