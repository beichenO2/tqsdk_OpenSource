"""Strategy definition, versioning, and parameter models."""

from __future__ import annotations

from typing import Optional
from uuid import uuid4

import enum

from sqlalchemy import (
    Boolean,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.db.base import Base, TimestampMixin


class StrategyType(str, enum.Enum):
    RULE_BASED = "rule_based"
    ML_SUPERVISED = "ml_supervised"
    RL = "rl"
    HYBRID = "hybrid"


class StrategyStatus(str, enum.Enum):
    DRAFT = "draft"
    BACKTESTING = "backtesting"
    PAPER_TRADING = "paper_trading"
    LIVE = "live"
    PAUSED = "paused"
    RETIRED = "retired"


class Strategy(TimestampMixin, Base):
    __tablename__ = "strategies"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    slug: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    owner_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=False
    )
    strategy_type: Mapped[StrategyType] = mapped_column(
        Enum(StrategyType, name="strategy_type_enum"), nullable=False
    )
    status: Mapped[StrategyStatus] = mapped_column(
        Enum(StrategyStatus, name="strategy_status_enum"),
        default=StrategyStatus.DRAFT,
        nullable=False,
    )
    description: Mapped[Optional[str]] = mapped_column(Text)
    target_instruments: Mapped[Optional[str]] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    owner: Mapped["User"] = relationship(back_populates="strategies")  # type: ignore[name-defined]  # noqa: F821
    versions: Mapped[list["StrategyVersion"]] = relationship(
        back_populates="strategy", cascade="all, delete-orphan"
    )
    params: Mapped[list["StrategyParam"]] = relationship(
        back_populates="strategy", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_strategies_owner_id", "owner_id"),
        Index("ix_strategies_status", "status"),
    )


class StrategyVersion(TimestampMixin, Base):
    __tablename__ = "strategy_versions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    strategy_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("strategies.id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    code_snapshot: Mapped[Optional[str]] = mapped_column(Text)
    config_json: Mapped[Optional[str]] = mapped_column(Text)
    changelog: Mapped[Optional[str]] = mapped_column(Text)

    strategy: Mapped["Strategy"] = relationship(back_populates="versions")

    __table_args__ = (
        Index("ix_strategy_versions_strategy_version", "strategy_id", "version", unique=True),
    )


class StrategyParam(TimestampMixin, Base):
    __tablename__ = "strategy_params"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    strategy_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("strategies.id", ondelete="CASCADE"), nullable=False
    )
    key: Mapped[str] = mapped_column(String(64), nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    param_type: Mapped[str] = mapped_column(String(16), default="string")
    description: Mapped[Optional[str]] = mapped_column(String(255))

    strategy: Mapped["Strategy"] = relationship(back_populates="params")

    __table_args__ = (
        Index("ix_strategy_params_strategy_key", "strategy_id", "key", unique=True),
    )
