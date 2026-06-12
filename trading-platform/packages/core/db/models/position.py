"""Position and position snapshot models."""

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
from sqlalchemy.orm import Mapped, mapped_column

from core.db.base import Base, TimestampMixin


class PositionSide(str, enum.Enum):
    LONG = "long"
    SHORT = "short"


class Position(TimestampMixin, Base):
    """Real-time position state, one row per user × instrument × side."""

    __tablename__ = "positions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=False
    )
    strategy_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("strategies.id")
    )
    instrument_symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    exchange_code: Mapped[str] = mapped_column(String(16), nullable=False)
    side: Mapped[PositionSide] = mapped_column(
        Enum(PositionSide, name="position_side_enum"), nullable=False
    )
    quantity: Mapped[Decimal] = mapped_column(Numeric(18, 8), default=0)
    avg_entry_price: Mapped[Decimal] = mapped_column(Numeric(18, 8), default=0)
    unrealized_pnl: Mapped[Decimal] = mapped_column(Numeric(18, 8), default=0)
    realized_pnl: Mapped[Decimal] = mapped_column(Numeric(18, 8), default=0)
    margin_used: Mapped[Decimal] = mapped_column(Numeric(18, 8), default=0)
    last_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 8))
    last_updated: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    metadata_json: Mapped[Optional[str]] = mapped_column(Text)

    __table_args__ = (
        Index(
            "ix_positions_user_instrument_side",
            "user_id",
            "instrument_symbol",
            "exchange_code",
            "side",
            unique=True,
        ),
        Index("ix_positions_strategy_id", "strategy_id"),
    )


class PositionSnapshot(Base):
    """End-of-day snapshot for audit trail and P&L attribution."""

    __tablename__ = "position_snapshots"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    position_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("positions.id"), nullable=False
    )
    snapshot_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    avg_entry_price: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    mark_price: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    unrealized_pnl: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    realized_pnl: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)

    __table_args__ = (
        Index("ix_position_snapshots_date", "snapshot_date"),
        Index("ix_position_snapshots_position_date", "position_id", "snapshot_date", unique=True),
    )
