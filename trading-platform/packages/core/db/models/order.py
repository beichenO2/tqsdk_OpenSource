"""Order and fill (execution) models."""

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


class OrderSide(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, enum.Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    FOK = "fok"
    FAK = "fak"


class OffsetFlag(str, enum.Enum):
    OPEN = "open"
    CLOSE = "close"
    CLOSE_TODAY = "close_today"
    CLOSE_YESTERDAY = "close_yesterday"


class OrderStatus(str, enum.Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class Order(TimestampMixin, Base):
    __tablename__ = "orders"

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
    side: Mapped[OrderSide] = mapped_column(
        Enum(OrderSide, name="order_side_enum"), nullable=False
    )
    order_type: Mapped[OrderType] = mapped_column(
        Enum(OrderType, name="order_type_enum"), nullable=False
    )
    offset: Mapped[OffsetFlag] = mapped_column(
        Enum(OffsetFlag, name="offset_flag_enum"), default=OffsetFlag.OPEN
    )
    price: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 8))
    quantity: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    filled_quantity: Mapped[Decimal] = mapped_column(Numeric(18, 8), default=0)
    avg_fill_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 8))
    status: Mapped[OrderStatus] = mapped_column(
        Enum(OrderStatus, name="order_status_enum"),
        default=OrderStatus.PENDING,
        nullable=False,
    )
    broker_order_id: Mapped[Optional[str]] = mapped_column(String(128))
    stop_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 8))
    time_in_force: Mapped[str] = mapped_column(String(8), default="GTC")
    submitted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    filled_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    cancelled_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    reject_reason: Mapped[Optional[str]] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(16), default="manual")
    metadata_json: Mapped[Optional[str]] = mapped_column(Text)

    fills: Mapped[list["Fill"]] = relationship(back_populates="order", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_orders_user_id", "user_id"),
        Index("ix_orders_strategy_id", "strategy_id"),
        Index("ix_orders_status", "status"),
        Index("ix_orders_symbol_exchange", "instrument_symbol", "exchange_code"),
        Index("ix_orders_created_at", "created_at"),
    )


class Fill(TimestampMixin, Base):
    __tablename__ = "fills"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    order_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("orders.id", ondelete="CASCADE"), nullable=False
    )
    price: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    commission: Mapped[Decimal] = mapped_column(Numeric(18, 8), default=0)
    filled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    broker_fill_id: Mapped[Optional[str]] = mapped_column(String(128))

    order: Mapped["Order"] = relationship(back_populates="fills")

    __table_args__ = (
        Index("ix_fills_order_id", "order_id"),
        Index("ix_fills_filled_at", "filled_at"),
    )
