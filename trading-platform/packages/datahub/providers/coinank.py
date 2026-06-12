"""CoinAnk data provider — contract positions, funding rates, liquidation data.

API docs: https://coinank.com/api
Requires: COINANK_API_KEY environment variable
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

BASE_URL = "https://api.coinank.com/api"


class CoinAnkProvider(DataProvider):
    """Adapter for CoinAnk derivatives data API."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.getenv("COINANK_API_KEY", "")
        self._session: aiohttp.ClientSession | None = None

    @property
    def name(self) -> str:
        return "coinank"

    @property
    def supported_exchanges(self) -> list[str]:
        return ["binance", "okx", "bybit", "bitget", "deribit"]

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

    async def get_open_interest(
        self, symbol: str = "BTC", exchange: str | None = None
    ) -> dict[str, Any]:
        """Get aggregate open interest data."""
        params: dict[str, Any] = {"symbol": symbol}
        if exchange:
            params["exchange"] = exchange
        try:
            data = await self._request("/openInterest", params)
            return data.get("data", {})
        except Exception as e:
            logger.warning("CoinAnk open interest fetch failed: %s", e)
            return {}

    async def get_funding_rates(
        self, symbol: str = "BTC", exchange: str | None = None
    ) -> list[dict[str, Any]]:
        """Get current and historical funding rates."""
        params: dict[str, Any] = {"symbol": symbol}
        if exchange:
            params["exchange"] = exchange
        try:
            data = await self._request("/fundingRate", params)
            return data.get("data", [])
        except Exception as e:
            logger.warning("CoinAnk funding rates fetch failed: %s", e)
            return []

    async def get_liquidations(
        self, symbol: str = "BTC", period: str = "24h"
    ) -> dict[str, Any]:
        """Get liquidation data (longs/shorts/total)."""
        try:
            data = await self._request(
                "/liquidation", {"symbol": symbol, "period": period}
            )
            return data.get("data", {})
        except Exception as e:
            logger.warning("CoinAnk liquidation fetch failed: %s", e)
            return {}

    async def get_long_short_ratio(
        self, symbol: str = "BTC", exchange: str = "binance"
    ) -> dict[str, Any]:
        """Get long/short ratio data."""
        try:
            data = await self._request(
                "/longShortRatio", {"symbol": symbol, "exchange": exchange}
            )
            return data.get("data", {})
        except Exception as e:
            logger.warning("CoinAnk long/short ratio fetch failed: %s", e)
            return {}

    async def get_ohlcv(
        self, symbol: str, timeframe: TimeFrame,
        start: datetime, end: Optional[datetime] = None, limit: Optional[int] = None,
    ) -> list[OHLCV]:
        return []

    async def subscribe_ticks(self, symbols: list[str]) -> AsyncIterator[TickData]:
        raise NotImplementedError("CoinAnk does not provide tick data")
        yield  # type: ignore[misc]

    async def get_snapshot(self, symbol: str) -> MarketSnapshot:
        raise NotImplementedError("Use CoinMarketCap for price snapshots")
