"""Tick 行情领域模型."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel


class Tick(BaseModel):
    """逐笔行情快照."""

    symbol: str
    datetime: datetime
    last_price: Decimal
    highest: Decimal
    lowest: Decimal
    volume: int
    amount: Decimal
    open_interest: int | None = None
    bid_price1: Decimal | None = None
    bid_volume1: int | None = None
    ask_price1: Decimal | None = None
    ask_volume1: int | None = None
