"""Market data service — thin layer over TqMarketAdapter for routes and tests."""

from __future__ import annotations

from typing import Any

from broker_tqsdk.adapter import TqMarketAdapter
from core.models.bar import Bar
from core.models.tick import Tick


def create_market_adapter(api: Any = None) -> TqMarketAdapter:
    """Factory for TqMarketAdapter. Use ``api=None`` until lifespan wires a real TqApi."""
    return TqMarketAdapter(api=api)


class MarketService:
    """Delegates to :class:`TqMarketAdapter`; inject a placeholder adapter in tests."""

    __slots__ = ("_adapter",)

    def __init__(self, adapter: TqMarketAdapter) -> None:
        self._adapter = adapter

    @property
    def adapter(self) -> TqMarketAdapter:
        return self._adapter

    async def get_quote(self, symbol: str) -> Tick | None:
        return await self._adapter.get_quote(symbol)

    async def get_klines(
        self, symbol: str, *, duration_seconds: int, data_length: int
    ) -> list[Bar]:
        return await self._adapter.get_klines(
            symbol, duration_seconds=duration_seconds, data_length=data_length
        )

    async def list_instruments(self, exchange_id: str | None) -> list[dict[str, str]]:
        return await self._adapter.list_instruments(exchange_id=exchange_id)
