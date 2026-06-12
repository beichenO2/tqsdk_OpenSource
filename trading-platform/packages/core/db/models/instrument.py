"""Instrument and exchange models — covers futures and crypto."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Optional
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    Date,
    Enum,
    Index,
    Numeric,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from core.db.base import Base, TimestampMixin

import enum


class AssetClass(str, enum.Enum):
    FUTURES = "futures"
    CRYPTO = "crypto"
    STOCK = "stock"
    OPTION = "option"


class Exchange(TimestampMixin, Base):
    """Trading venue — SHFE, DCE, CZCE, INE, GFEX, Binance, OKX, etc."""

    __tablename__ = "exchanges"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    code: Mapped[str] = mapped_column(String(16), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    asset_class: Mapped[AssetClass] = mapped_column(
        Enum(AssetClass, name="asset_class_enum"), nullable=False
    )
    timezone: Mapped[str] = mapped_column(String(64), default="Asia/Shanghai")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class Instrument(TimestampMixin, Base):
    """A tradeable contract / symbol."""

    __tablename__ = "instruments"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    exchange_code: Mapped[str] = mapped_column(String(16), nullable=False)
    asset_class: Mapped[AssetClass] = mapped_column(
        Enum(AssetClass, name="asset_class_enum", create_type=False), nullable=False
    )
    name: Mapped[Optional[str]] = mapped_column(String(128))
    currency: Mapped[str] = mapped_column(String(8), default="CNY")
    tick_size: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    lot_size: Mapped[Decimal] = mapped_column(Numeric(18, 8), default=1)
    multiplier: Mapped[Decimal] = mapped_column(Numeric(18, 4), default=1)
    margin_rate: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))
    expire_date: Mapped[Optional[date]] = mapped_column(Date)
    underlying: Mapped[Optional[str]] = mapped_column(String(32))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    metadata_json: Mapped[Optional[str]] = mapped_column(Text)

    __table_args__ = (
        Index("ix_instruments_symbol_exchange", "symbol", "exchange_code", unique=True),
        Index("ix_instruments_asset_class", "asset_class"),
    )
