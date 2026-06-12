"""ML model registry, experiments, and versioning."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Optional
from uuid import uuid4

import enum

from sqlalchemy import (
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.db.base import Base, TimestampMixin


class ModelFramework(str, enum.Enum):
    PYTORCH = "pytorch"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    STABLE_BASELINES3 = "stable_baselines3"
    SKLEARN = "sklearn"
    CUSTOM = "custom"


class ModelStatus(str, enum.Enum):
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATING = "validating"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"


class MLModel(TimestampMixin, Base):
    __tablename__ = "ml_models"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    slug: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    framework: Mapped[ModelFramework] = mapped_column(
        Enum(ModelFramework, name="model_framework_enum"), nullable=False
    )
    strategy_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("strategies.id")
    )
    description: Mapped[Optional[str]] = mapped_column(Text)
    target_variable: Mapped[Optional[str]] = mapped_column(String(128))
    feature_set_json: Mapped[Optional[str]] = mapped_column(Text)

    versions: Mapped[list["MLModelVersion"]] = relationship(
        back_populates="model", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_ml_models_strategy_id", "strategy_id"),
    )


class MLModelVersion(TimestampMixin, Base):
    __tablename__ = "ml_model_versions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    model_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("ml_models.id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[ModelStatus] = mapped_column(
        Enum(ModelStatus, name="model_status_enum"),
        default=ModelStatus.TRAINING,
        nullable=False,
    )
    artifact_path: Mapped[Optional[str]] = mapped_column(String(512))
    metrics_json: Mapped[Optional[str]] = mapped_column(Text)
    hyperparams_json: Mapped[Optional[str]] = mapped_column(Text)
    training_started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    training_finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    model: Mapped["MLModel"] = relationship(back_populates="versions")

    __table_args__ = (
        Index("ix_ml_model_versions_model_version", "model_id", "version", unique=True),
        Index("ix_ml_model_versions_status", "status"),
    )


class MLExperiment(TimestampMixin, Base):
    __tablename__ = "ml_experiments"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    model_version_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("ml_model_versions.id"), nullable=False
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    config_json: Mapped[Optional[str]] = mapped_column(Text)
    results_json: Mapped[Optional[str]] = mapped_column(Text)
    dataset_ref: Mapped[Optional[str]] = mapped_column(String(512))
    train_score: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 6))
    val_score: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 6))
    test_score: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 6))
    notes: Mapped[Optional[str]] = mapped_column(Text)

    __table_args__ = (
        Index("ix_ml_experiments_model_version_id", "model_version_id"),
        Index("ix_ml_experiments_user_id", "user_id"),
    )
