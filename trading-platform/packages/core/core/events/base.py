"""基础领域事件."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class DomainEvent(BaseModel):
    """所有领域事件的基类."""

    event_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    payload: dict[str, Any] = Field(default_factory=dict)


class OrderEvent(DomainEvent):
    event_type: str = "order"
    order_id: str = ""
    strategy_id: str = ""


class TradeEvent(DomainEvent):
    event_type: str = "trade"
    trade_id: str = ""
    order_id: str = ""


class PositionEvent(DomainEvent):
    event_type: str = "position"
    symbol: str = ""
