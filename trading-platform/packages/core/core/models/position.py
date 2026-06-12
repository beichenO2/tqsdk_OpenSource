"""持仓领域模型."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel

from core.enums.direction import Direction
from core.enums.market import Exchange


class Position(BaseModel):
    """账户在某合约上的持仓."""

    symbol: str
    exchange: Exchange
    direction: Direction
    volume: int = 0
    available: int = 0
    avg_price: Decimal = Decimal("0")
    margin: Decimal = Decimal("0")
    float_pnl: Decimal = Decimal("0")
    close_pnl: Decimal = Decimal("0")

    @property
    def frozen(self) -> int:
        return self.volume - self.available
