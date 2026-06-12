"""BlockBeats (律动) data provider — 1500+ crypto news sources, fund flows, macro data.

API docs: https://api.theblockbeats.news/
Requires: BLOCKBEATS_API_KEY environment variable
"""

from __future__ import annotations

import os
import logging
from datetime import datetime
from typing import Any, AsyncIterator, Optional

import aiohttp

from datahub.models import OHLCV, MarketSnapshot, TickData, TimeFrame
from datahub.providers.base import DataProvider

logger = logging.getLogger(__name__)

BASE_URL = "https://api.theblockbeats.news/v1"


class BlockBeatsProvider(DataProvider):
    """Adapter for BlockBeats news and macro data API."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.getenv("BLOCKBEATS_API_KEY", "")
        self._session: aiohttp.ClientSession | None = None

    @property
    def name(self) -> str:
        return "blockbeats"

    @property
    def supported_exchanges(self) -> list[str]:
        return ["binance", "okx", "coinbase", "bybit"]

    async def connect(self) -> None:
        self._session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=aiohttp.ClientTimeout(total=30),
        )

    async def disconnect(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def _request(self, endpoint: str, params: dict | None = None) -> dict:
        if not self._session:
            await self.connect()
        assert self._session is not None
        async with self._session.get(f"{BASE_URL}{endpoint}", params=params) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def get_news(
        self, category: str = "all", limit: int = 20
    ) -> list[dict[str, Any]]:
        """Get latest crypto news from 1500+ sources."""
        try:
            data = await self._request("/news", {"category": category, "limit": limit})
            return data.get("data", [])
        except Exception as e:
            logger.warning("BlockBeats news fetch failed: %s", e)
            return []

    async def get_fund_flows(self, symbol: str = "BTC") -> dict[str, Any]:
        """Get fund flow data for a symbol."""
        try:
            data = await self._request("/fund-flows", {"symbol": symbol})
            return data.get("data", {})
        except Exception as e:
            logger.warning("BlockBeats fund flows fetch failed: %s", e)
            return {}

    async def get_macro_data(self) -> dict[str, Any]:
        """Get macro economic indicators relevant to crypto."""
        try:
            data = await self._request("/macro")
            return data.get("data", {})
        except Exception as e:
            logger.warning("BlockBeats macro data fetch failed: %s", e)
            return {}

    async def get_ohlcv(
        self, symbol: str, timeframe: TimeFrame,
        start: datetime, end: Optional[datetime] = None, limit: Optional[int] = None,
    ) -> list[OHLCV]:
        return []

    async def subscribe_ticks(self, symbols: list[str]) -> AsyncIterator[TickData]:
        raise NotImplementedError("BlockBeats does not provide tick data")
        yield  # type: ignore[misc]

    async def get_snapshot(self, symbol: str) -> MarketSnapshot:
        raise NotImplementedError("Use CoinMarketCap for price snapshots")
