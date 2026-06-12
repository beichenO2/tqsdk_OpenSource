"""Risk rule and alert models."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Optional
from uuid import uuid4

import enum

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from core.db.base import Base, TimestampMixin


class RiskRuleType(str, enum.Enum):
    MAX_POSITION = "max_position"
    MAX_ORDER_SIZE = "max_order_size"
    MAX_DRAWDOWN = "max_drawdown"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    ORDER_RATE_LIMIT = "order_rate_limit"
    CONCENTRATION_LIMIT = "concentration_limit"
    CUSTOM = "custom"


class RiskSeverity(str, enum.Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    BLOCK = "block"


class RiskRule(TimestampMixin, Base):
    __tablename__ = "risk_rules"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=False
    )
    strategy_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("strategies.id")
    )
    rule_type: Mapped[RiskRuleType] = mapped_column(
        Enum(RiskRuleType, name="risk_rule_type_enum"), nullable=False
    )
    severity: Mapped[RiskSeverity] = mapped_column(
        Enum(RiskSeverity, name="risk_severity_enum"),
        default=RiskSeverity.WARNING,
    )
    instrument_filter: Mapped[Optional[str]] = mapped_column(String(64))
    threshold_value: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    cooldown_seconds: Mapped[int] = mapped_column(Integer, default=0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    description: Mapped[Optional[str]] = mapped_column(Text)
    config_json: Mapped[Optional[str]] = mapped_column(Text)

    __table_args__ = (
        Index("ix_risk_rules_user_id", "user_id"),
        Index("ix_risk_rules_strategy_id", "strategy_id"),
    )


class RiskAlert(TimestampMixin, Base):
    __tablename__ = "risk_alerts"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    rule_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("risk_rules.id"), nullable=False
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=False
    )
    severity: Mapped[RiskSeverity] = mapped_column(
        Enum(RiskSeverity, name="risk_severity_enum", create_type=False), nullable=False
    )
    triggered_value: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    threshold_value: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    order_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("orders.id")
    )

    __table_args__ = (
        Index("ix_risk_alerts_user_id", "user_id"),
        Index("ix_risk_alerts_created_at", "created_at"),
        Index("ix_risk_alerts_severity", "severity"),
    )
