"""Backtest run, trade, and metrics models."""

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
    Numeric,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.db.base import Base, TimestampMixin


class BacktestStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BacktestRun(TimestampMixin, Base):
    __tablename__ = "backtest_runs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=False
    )
    strategy_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("strategies.id"), nullable=False
    )
    strategy_version_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("strategy_versions.id")
    )
    name: Mapped[Optional[str]] = mapped_column(String(255))
    status: Mapped[BacktestStatus] = mapped_column(
        Enum(BacktestStatus, name="backtest_status_enum"),
        default=BacktestStatus.QUEUED,
        nullable=False,
    )
    start_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    initial_capital: Mapped[Decimal] = mapped_column(Numeric(18, 4), nullable=False)
    instruments_json: Mapped[Optional[str]] = mapped_column(Text)
    params_json: Mapped[Optional[str]] = mapped_column(Text)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    result_path: Mapped[Optional[str]] = mapped_column(String(512))

    trades: Mapped[list["BacktestTrade"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    metrics: Mapped[list["BacktestMetric"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_backtest_runs_user_id", "user_id"),
        Index("ix_backtest_runs_strategy_id", "strategy_id"),
        Index("ix_backtest_runs_status", "status"),
    )


class BacktestTrade(Base):
    __tablename__ = "backtest_trades"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("backtest_runs.id", ondelete="CASCADE"), nullable=False
    )
    instrument_symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    price: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    commission: Mapped[Decimal] = mapped_column(Numeric(18, 8), default=0)
    pnl: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 8))
    traded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    signal_name: Mapped[Optional[str]] = mapped_column(String(64))

    run: Mapped["BacktestRun"] = relationship(back_populates="trades")

    __table_args__ = (
        Index("ix_backtest_trades_run_id", "run_id"),
    )


class BacktestMetric(Base):
    __tablename__ = "backtest_metrics"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("backtest_runs.id", ondelete="CASCADE"), nullable=False
    )
    metric_name: Mapped[str] = mapped_column(String(64), nullable=False)
    metric_value: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)

    run: Mapped["BacktestRun"] = relationship(back_populates="metrics")

    __table_args__ = (
        Index("ix_backtest_metrics_run_name", "run_id", "metric_name", unique=True),
    )
