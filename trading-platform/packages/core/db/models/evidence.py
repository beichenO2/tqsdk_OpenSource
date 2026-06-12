"""Evidence chain and decision log models for explainability."""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import uuid4

import enum

from sqlalchemy import (
    DateTime,
    Enum,
    ForeignKey,
    Index,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from core.db.base import Base, TimestampMixin


class EvidenceType(str, enum.Enum):
    SIGNAL = "signal"
    RISK_CHECK = "risk_check"
    ORDER_DECISION = "order_decision"
    POSITION_CHANGE = "position_change"
    MODEL_PREDICTION = "model_prediction"
    FEATURE_VALUE = "feature_value"
    MANUAL_OVERRIDE = "manual_override"


class EvidenceRecord(TimestampMixin, Base):
    """Immutable audit trail entry linking a decision to its evidence."""

    __tablename__ = "evidence_records"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=False
    )
    strategy_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("strategies.id")
    )
    order_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("orders.id")
    )
    evidence_type: Mapped[EvidenceType] = mapped_column(
        Enum(EvidenceType, name="evidence_type_enum"), nullable=False
    )
    instrument_symbol: Mapped[Optional[str]] = mapped_column(String(32))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    explanation: Mapped[Optional[str]] = mapped_column(Text)
    parent_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("evidence_records.id")
    )

    __table_args__ = (
        Index("ix_evidence_records_user_id", "user_id"),
        Index("ix_evidence_records_order_id", "order_id"),
        Index("ix_evidence_records_strategy_id", "strategy_id"),
        Index("ix_evidence_records_type", "evidence_type"),
        Index("ix_evidence_records_timestamp", "timestamp"),
    )


class DecisionLog(TimestampMixin, Base):
    """High-level decision log aggregating evidence into a narrative."""

    __tablename__ = "decision_logs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=False
    )
    strategy_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("strategies.id")
    )
    order_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("orders.id")
    )
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    instrument_symbol: Mapped[Optional[str]] = mapped_column(String(32))
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_ids_json: Mapped[Optional[str]] = mapped_column(Text)
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_decision_logs_user_id", "user_id"),
        Index("ix_decision_logs_strategy_id", "strategy_id"),
        Index("ix_decision_logs_decided_at", "decided_at"),
    )
