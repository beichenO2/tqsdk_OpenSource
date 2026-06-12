"""交易所适配器抽象基类 — 所有具体交易所实现必须继承此类。"""

from __future__ import annotations

import abc
from typing import AsyncIterator

from .models import (
    Balance,
    Exchange,
    ExchangeCredentials,
    OHLCV,
    OrderBook,
    OrderRequest,
    OrderResponse,
    Position,
    Ticker,
    Trade,
)


class ExchangeAdapter(abc.ABC):
    """Abstract exchange adapter that every concrete implementation must fulfill."""

    def __init__(self, credentials: ExchangeCredentials) -> None:
        self._credentials = credentials

    @property
    @abc.abstractmethod
    def exchange(self) -> Exchange:
        ...

    # ── Market Data (REST) ──────────────────────────────────────────

    @abc.abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker:
        ...

    @abc.abstractmethod
    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        ...

    @abc.abstractmethod
    async def get_ohlcv(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 500,
    ) -> list[OHLCV]:
        ...

    @abc.abstractmethod
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]:
        ...

    # ── Market Data (WebSocket) ─────────────────────────────────────

    @abc.abstractmethod
    async def stream_ticker(self, symbol: str) -> AsyncIterator[Ticker]:
        ...

    @abc.abstractmethod
    async def stream_orderbook(self, symbol: str) -> AsyncIterator[OrderBook]:
        ...

    @abc.abstractmethod
    async def stream_trades(self, symbol: str) -> AsyncIterator[Trade]:
        ...

    @abc.abstractmethod
    async def stream_ohlcv(
        self, symbol: str, interval: str = "1m"
    ) -> AsyncIterator[OHLCV]:
        ...

    # ── Trading ─────────────────────────────────────────────────────

    @abc.abstractmethod
    async def place_order(self, request: OrderRequest) -> OrderResponse:
        ...

    @abc.abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> OrderResponse:
        ...

    @abc.abstractmethod
    async def get_order(self, order_id: str, symbol: str) -> OrderResponse:
        ...

    @abc.abstractmethod
    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]:
        ...

    # ── Account ─────────────────────────────────────────────────────

    @abc.abstractmethod
    async def get_balances(self) -> list[Balance]:
        ...

    @abc.abstractmethod
    async def get_positions(self, symbol: str | None = None) -> list[Position]:
        ...

    # ── Lifecycle ───────────────────────────────────────────────────

    @abc.abstractmethod
    async def connect(self) -> None:
        """Initialize HTTP session and authenticate."""
        ...

    @abc.abstractmethod
    async def disconnect(self) -> None:
        """Clean up connections and resources."""
        ...

    async def __aenter__(self) -> "ExchangeAdapter":
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.disconnect()
