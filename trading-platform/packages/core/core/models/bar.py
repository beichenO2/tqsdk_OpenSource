"""K 线 (OHLCV) 领域模型."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, model_validator


class Bar(BaseModel):
    """标准 OHLCV K线."""

    symbol: str
    datetime: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    open_interest: int | None = None
    duration_seconds: int = 60

    @model_validator(mode="after")
    def validate_price_range(self) -> Bar:
        if self.high < self.low:
            raise ValueError("high must be greater than or equal to low")
        return self
