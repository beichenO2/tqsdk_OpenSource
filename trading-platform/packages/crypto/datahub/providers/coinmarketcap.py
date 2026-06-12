"""CoinMarketCap data provider — 60 API endpoints, prices, market cap, research.

API docs: https://coinmarketcap.com/api/documentation/v1/
Requires: CMC_API_KEY environment variable
"""

from __future__ import annotations

import os
import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, AsyncIterator, Optional

import aiohttp

from datahub.models import OHLCV, MarketSnapshot, TickData, TimeFrame
from datahub.providers.base import DataProvider

logger = logging.getLogger(__name__)

BASE_URL = "https://pro-api.coinmarketcap.com/v1"


class CoinMarketCapProvider(DataProvider):
    """Adapter for CoinMarketCap PRO API."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.getenv("CMC_API_KEY", "")
        self._session: aiohttp.ClientSession | None = None

    @property
    def name(self) -> str:
        return "coinmarketcap"

    @property
    def supported_exchanges(self) -> list[str]:
        return ["binance", "okx", "coinbase", "bybit", "kraken"]

    async def connect(self) -> None:
        self._session = aiohttp.ClientSession(
            headers={"X-CMC_PRO_API_KEY": self._api_key},
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

    async def get_latest_quotes(
        self, symbols: list[str] | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Get latest price quotes for top cryptocurrencies."""
        params: dict[str, Any] = {"limit": limit, "convert": "USD"}
        if symbols:
            params["symbol"] = ",".join(symbols)
            endpoint = "/cryptocurrency/quotes/latest"
        else:
            endpoint = "/cryptocurrency/listings/latest"
        try:
            data = await self._request(endpoint, params)
            return data.get("data", [])
        except Exception as e:
            logger.warning("CMC quotes fetch failed: %s", e)
            return []

    async def get_global_metrics(self) -> dict[str, Any]:
        """Get global crypto market metrics (market cap, dominance, etc.)."""
        try:
            data = await self._request("/global-metrics/quotes/latest")
            return data.get("data", {})
        except Exception as e:
            logger.warning("CMC global metrics fetch failed: %s", e)
            return {}

    async def get_exchange_info(self, exchange_slug: str) -> dict[str, Any]:
        """Get exchange info and rankings."""
        try:
            data = await self._request(
                "/exchange/info", {"slug": exchange_slug}
            )
            return data.get("data", {})
        except Exception as e:
            logger.warning("CMC exchange info fetch failed: %s", e)
            return {}

    async def get_snapshot(self, symbol: str) -> MarketSnapshot:
        quotes = await self.get_latest_quotes([symbol.replace("USDT", "")])
        if not quotes:
            raise ValueError(f"No CMC data for {symbol}")
        q = quotes[0] if isinstance(quotes, list) else list(quotes.values())[0]
        usd = q.get("quote", {}).get("USD", {})
        price = Decimal(str(usd.get("price", 0)))
        vol = Decimal(str(usd.get("volume_24h", 0)))
        return MarketSnapshot(
            symbol=symbol,
            last_price=price,
            bid=price,
            ask=price,
            volume_24h=vol,
            timestamp=datetime.utcnow(),
        )

    async def get_ohlcv(
        self, symbol: str, timeframe: TimeFrame,
        start: datetime, end: Optional[datetime] = None, limit: Optional[int] = None,
    ) -> list[OHLCV]:
        return []

    async def subscribe_ticks(self, symbols: list[str]) -> AsyncIterator[TickData]:
        raise NotImplementedError("CMC does not provide real-time tick data")
        yield  # type: ignore[misc]
