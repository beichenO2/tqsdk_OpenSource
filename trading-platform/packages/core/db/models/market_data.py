"""Data source and subscription models for market data management."""

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
from sqlalchemy.orm import Mapped, mapped_column

from core.db.base import Base, TimestampMixin


class DataSourceType(str, enum.Enum):
    TQSDK = "tqsdk"
    EXCHANGE_WS = "exchange_ws"
    EXCHANGE_REST = "exchange_rest"
    CSV_FILE = "csv_file"
    PARQUET_FILE = "parquet_file"
    DUCKDB = "duckdb"
    EXTERNAL_API = "external_api"


class DataFrequency(str, enum.Enum):
    TICK = "tick"
    S1 = "1s"
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


class DataSource(TimestampMixin, Base):
    __tablename__ = "data_sources"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    source_type: Mapped[DataSourceType] = mapped_column(
        Enum(DataSourceType, name="data_source_type_enum"), nullable=False
    )
    config_json: Mapped[Optional[str]] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    priority: Mapped[int] = mapped_column(Integer, default=0)
    description: Mapped[Optional[str]] = mapped_column(Text)


class DataSubscription(TimestampMixin, Base):
    __tablename__ = "data_subscriptions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=False
    )
    data_source_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("data_sources.id"), nullable=False
    )
    instrument_symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    exchange_code: Mapped[str] = mapped_column(String(16), nullable=False)
    frequency: Mapped[DataFrequency] = mapped_column(
        Enum(DataFrequency, name="data_frequency_enum"), nullable=False
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    __table_args__ = (
        Index("ix_data_subs_user_id", "user_id"),
        Index(
            "ix_data_subs_unique",
            "user_id",
            "data_source_id",
            "instrument_symbol",
            "frequency",
            unique=True,
        ),
    )
