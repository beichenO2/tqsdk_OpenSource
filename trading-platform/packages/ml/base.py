"""ML 模型基础接口 — 所有监督学习模型的抽象基类。

提供统一的 train / predict / evaluate / save / load 接口，
支持 XGBoost、LightGBM、PyTorch 等不同框架的实现。
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field


class MLFramework(str, enum.Enum):
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    PYTORCH = "pytorch"
    SKLEARN = "sklearn"
    CUSTOM = "custom"


class MLModelStatus(str, enum.Enum):
    UNTRAINED = "untrained"
    TRAINING = "training"
    TRAINED = "trained"
    EVALUATING = "evaluating"
    DEPLOYED = "deployed"
    FAILED = "failed"


class MLModelMeta(BaseModel):
    """ML 模型元信息。"""

    model_id: str
    name: str
    framework: MLFramework
    version: str = "1.0.0"
    status: MLModelStatus = MLModelStatus.UNTRAINED
    feature_columns: list[str] = Field(default_factory=list)
    target_column: str = ""
    hyperparams: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    artifact_path: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    trained_at: Optional[datetime] = None


class TrainResult(BaseModel):
    """训练结果。"""

    train_score: float
    val_score: Optional[float] = None
    metrics: dict[str, float] = Field(default_factory=dict)
    duration_seconds: float = 0.0
    epochs: Optional[int] = None
    best_iteration: Optional[int] = None


class PredictResult(BaseModel):
    """预测结果。"""

    predictions: list[float] = Field(default_factory=list)
    probabilities: Optional[list[list[float]]] = None
    feature_importance: Optional[dict[str, float]] = None


class BaseMLModel(ABC):
    """所有 ML 模型的抽象基类。"""

    def __init__(self, meta: MLModelMeta) -> None:
        self.meta = meta
        self._model: Any = None

    @property
    def model_id(self) -> str:
        return self.meta.model_id

    @property
    def is_trained(self) -> bool:
        return self.meta.status in (MLModelStatus.TRAINED, MLModelStatus.DEPLOYED)

    @abstractmethod
    async def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> TrainResult:
        """训练模型。"""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> PredictResult:
        """推理预测。"""
        ...

    @abstractmethod
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
        """评估模型在测试集上的表现。"""
        ...

    @abstractmethod
    def save(self, path: str) -> str:
        """保存模型到磁盘，返回保存路径。"""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """从磁盘加载模型。"""
        ...

    def get_feature_importance(self) -> Optional[dict[str, float]]:
        """获取特征重要性（不支持的模型返回 None）。"""
        return None
