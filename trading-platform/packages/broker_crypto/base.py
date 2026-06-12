"""加密货币交易所接口抽象基类。

由 Ch33 创建接口定义桩，Ch32 负责具体交易所适配器实现。
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class ExchangeConfig(BaseModel):
    """交易所连接配置。"""

    exchange_id: str
    api_key_ref: str = Field(description="凭证引用 ID（由 Ch37 安全模块管理）")
    testnet: bool = True
    rate_limit_per_sec: int = 10
    timeout_ms: int = 5000
    extra: dict[str, Any] = Field(default_factory=dict)


class OrderSide(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, enum.Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, enum.Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderResult(BaseModel):
    """下单结果。"""

    order_id: str
    exchange_order_id: str | None = None
    status: OrderStatus
    filled_qty: float = 0.0
    avg_fill_price: float | None = None
    fee: float = 0.0
    fee_currency: str = "USDT"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    raw_response: dict[str, Any] = Field(default_factory=dict)


class TickerData(BaseModel):
    """最新行情 Ticker。"""

    symbol: str
    bid: float
    ask: float
    last: float
    volume_24h: float
    timestamp: datetime


class OrderBookData(BaseModel):
    """订单簿快照。"""

    symbol: str
    bids: list[tuple[float, float]]  # (price, qty)
    asks: list[tuple[float, float]]
    timestamp: datetime


class CryptoExchange(ABC):
    """加密货币交易所的统一抽象接口。"""

    def __init__(self, config: ExchangeConfig) -> None:
        self.config = config

    @abstractmethod
    async def connect(self) -> None:
        """建立与交易所的连接。"""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """断开连接。"""
        ...

    @abstractmethod
    async def get_ticker(self, symbol: str) -> TickerData:
        ...

    @abstractmethod
    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBookData:
        ...

    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        qty: float,
        price: float | None = None,
        stop_price: float | None = None,
    ) -> OrderResult:
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> OrderResult:
        ...

    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderResult:
        ...

    @abstractmethod
    async def get_balance(self, currency: str = "USDT") -> dict[str, float]:
        """返回 {'available': x, 'frozen': y, 'total': z}"""
        ...

    @abstractmethod
    async def get_positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    async def subscribe_ticker(self, symbol: str, callback: Any) -> None:
        """订阅实时行情推送。"""
        ...

    @abstractmethod
    async def subscribe_orderbook(self, symbol: str, callback: Any) -> None:
        """订阅订单簿推送。"""
        ...
