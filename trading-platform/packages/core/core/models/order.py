"""委托单领域模型."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

from pydantic import BaseModel, Field

from core.enums.direction import Direction, Offset
from core.enums.order_status import OrderStatus
from core.enums.market import Exchange


class Order(BaseModel):
    """表示一笔委托单."""

    order_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    strategy_id: str
    symbol: str
    exchange: Exchange
    direction: Direction
    offset: Offset
    price: Decimal
    volume: int
    filled_volume: int = 0
    avg_fill_price: Decimal | None = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = {"frozen": False}

    @property
    def remaining(self) -> int:
        return self.volume - self.filled_volume
