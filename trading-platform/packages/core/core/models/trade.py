"""成交记录领域模型."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

from pydantic import BaseModel, Field

from core.enums.direction import Direction, Offset
from core.enums.market import Exchange


class Trade(BaseModel):
    """一笔成交记录."""

    trade_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    order_id: str
    strategy_id: str
    symbol: str
    exchange: Exchange
    direction: Direction
    offset: Offset
    price: Decimal
    volume: int
    commission: Decimal = Decimal("0")
    traded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
