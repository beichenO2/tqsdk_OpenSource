"""BTC 交易所数据模型 — Pydantic v2 schemas."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Exchange(str, Enum):
    BINANCE = "BINANCE"
    OKX = "OKX"
    BYBIT = "BYBIT"
    WEEX = "WEEX"

    @classmethod
    def _missing_(cls, value: object) -> Exchange | None:
        if isinstance(value, str):
            upper = value.upper()
            for member in cls:
                if member.value == upper:
                    return member
        return None


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"
    STOP_LIMIT = "stop_limit"
    STOP_MARKET = "stop_market"


class OrderStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class TimeInForce(str, Enum):
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"


class Ticker(BaseModel):
    exchange: Exchange
    symbol: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    volume_24h: Decimal
    timestamp: datetime


class OHLCV(BaseModel):
    exchange: Exchange
    symbol: str
    interval: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    timestamp: datetime


class OrderBook(BaseModel):
    exchange: Exchange
    symbol: str
    bids: list[tuple[Decimal, Decimal]]
    asks: list[tuple[Decimal, Decimal]]
    timestamp: datetime


class Trade(BaseModel):
    exchange: Exchange
    symbol: str
    trade_id: str
    side: Side
    price: Decimal
    quantity: Decimal
    timestamp: datetime


class OrderRequest(BaseModel):
    exchange: Exchange
    symbol: str
    side: Side
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    client_order_id: Optional[str] = None


class OrderResponse(BaseModel):
    exchange: Exchange
    order_id: str
    client_order_id: Optional[str] = None
    symbol: str
    side: Side
    order_type: OrderType
    status: OrderStatus
    quantity: Decimal
    filled_quantity: Decimal = Decimal("0")
    price: Optional[Decimal] = None
    avg_fill_price: Optional[Decimal] = None
    created_at: datetime
    updated_at: datetime


class Position(BaseModel):
    exchange: Exchange
    symbol: str
    side: Side
    quantity: Decimal
    entry_price: Decimal
    unrealized_pnl: Decimal
    leverage: int = 1
    timestamp: datetime


class Balance(BaseModel):
    exchange: Exchange
    asset: str
    free: Decimal
    locked: Decimal
    total: Decimal = Field(default=Decimal("0"))

    def model_post_init(self, __context: object) -> None:
        if self.total == Decimal("0"):
            object.__setattr__(self, "total", self.free + self.locked)


class ExchangeCredentials(BaseModel):
    """Credentials for exchange API access. Values are encrypted at rest."""

    exchange: Exchange
    api_key: str
    api_secret: str
    passphrase: Optional[str] = None
    testnet: bool = False
