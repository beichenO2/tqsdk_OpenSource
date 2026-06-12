"""Dune Analytics MCP provider — natural language on-chain data queries.

API docs: https://docs.dune.com/api-reference/
Requires: DUNE_API_KEY environment variable
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

BASE_URL = "https://api.dune.com/api/v1"


class DuneProvider(DataProvider):
    """Adapter for Dune Analytics on-chain data API."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.getenv("DUNE_API_KEY", "")
        self._session: aiohttp.ClientSession | None = None

    @property
    def name(self) -> str:
        return "dune"

    @property
    def supported_exchanges(self) -> list[str]:
        return ["ethereum", "bitcoin", "solana", "arbitrum", "base"]

    async def connect(self) -> None:
        self._session = aiohttp.ClientSession(
            headers={"X-DUNE-API-KEY": self._api_key},
            timeout=aiohttp.ClientTimeout(total=120),
        )

    async def disconnect(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def _request(
        self, method: str, endpoint: str,
        params: dict | None = None, json: dict | None = None,
    ) -> dict:
        if not self._session:
            await self.connect()
        assert self._session is not None
        async with self._session.request(
            method, f"{BASE_URL}{endpoint}", params=params, json=json,
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def execute_query(self, query_id: int, params: dict | None = None) -> str:
        """Execute a Dune query and return execution_id for polling."""
        body: dict[str, Any] = {}
        if params:
            body["query_parameters"] = params
        try:
            data = await self._request("POST", f"/query/{query_id}/execute", json=body)
            return data.get("execution_id", "")
        except Exception as e:
            logger.warning("Dune query execution failed: %s", e)
            return ""

    async def get_execution_result(self, execution_id: str) -> dict[str, Any]:
        """Get results of a completed query execution."""
        try:
            data = await self._request("GET", f"/execution/{execution_id}/results")
            return data.get("result", {})
        except Exception as e:
            logger.warning("Dune result fetch failed: %s", e)
            return {}

    async def get_execution_status(self, execution_id: str) -> str:
        """Check status of a query execution."""
        try:
            data = await self._request("GET", f"/execution/{execution_id}/status")
            return data.get("state", "UNKNOWN")
        except Exception as e:
            logger.warning("Dune status check failed: %s", e)
            return "ERROR"

    async def run_query_and_wait(
        self, query_id: int, params: dict | None = None, max_wait: int = 60,
    ) -> dict[str, Any]:
        """Execute query and poll until complete."""
        import asyncio
        exec_id = await self.execute_query(query_id, params)
        if not exec_id:
            return {}
        for _ in range(max_wait // 5):
            status = await self.get_execution_status(exec_id)
            if status == "QUERY_STATE_COMPLETED":
                return await self.get_execution_result(exec_id)
            if status in ("QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED", "ERROR"):
                logger.warning("Dune query %d failed: %s", query_id, status)
                return {}
            await asyncio.sleep(5)
        logger.warning("Dune query %d timed out after %ds", query_id, max_wait)
        return {}

    async def get_latest_result(self, query_id: int) -> dict[str, Any]:
        """Get the latest cached result for a query (no execution cost)."""
        try:
            data = await self._request("GET", f"/query/{query_id}/results")
            return data.get("result", {})
        except Exception as e:
            logger.warning("Dune latest result fetch failed: %s", e)
            return {}

    async def get_ohlcv(
        self, symbol: str, timeframe: TimeFrame,
        start: datetime, end: Optional[datetime] = None, limit: Optional[int] = None,
    ) -> list[OHLCV]:
        return []

    async def subscribe_ticks(self, symbols: list[str]) -> AsyncIterator[TickData]:
        raise NotImplementedError("Dune does not provide tick data")
        yield  # type: ignore[misc]

    async def get_snapshot(self, symbol: str) -> MarketSnapshot:
        raise NotImplementedError("Use CoinMarketCap for price snapshots")
