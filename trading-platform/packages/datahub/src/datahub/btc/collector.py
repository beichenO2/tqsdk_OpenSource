"""BTC 多交易所数据采集器 — 聚合多个交易所的行情数据。"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Callable

from broker_crypto import (
    ExchangeAdapter,
    ExchangeCredentials,
    Exchange,
    OHLCV,
    OrderBook,
    Ticker,
    Trade,
    create_adapter,
)

logger = logging.getLogger(__name__)


class BTCDataCollector:
    """Collects market data from multiple crypto exchanges concurrently."""

    def __init__(self, credentials: list[ExchangeCredentials]) -> None:
        self._credentials = credentials
        self._adapters: dict[Exchange, ExchangeAdapter] = {}
        self._running = False

    async def start(self) -> None:
        for cred in self._credentials:
            adapter = create_adapter(cred)
            await adapter.connect()
            self._adapters[cred.exchange] = adapter
            logger.info("Collector started for %s", cred.exchange.value)
        self._running = True

    async def stop(self) -> None:
        self._running = False
        for exchange, adapter in self._adapters.items():
            await adapter.disconnect()
            logger.info("Collector stopped for %s", exchange.value)
        self._adapters.clear()

    # ── Snapshot queries ────────────────────────────────────────────

    async def get_all_tickers(self, symbol: str) -> dict[Exchange, Ticker]:
        """Fetch ticker from all connected exchanges concurrently."""
        tasks = {
            ex: asyncio.create_task(adapter.get_ticker(symbol))
            for ex, adapter in self._adapters.items()
        }
        results: dict[Exchange, Ticker] = {}
        for ex, task in tasks.items():
            try:
                results[ex] = await task
            except Exception:
                logger.exception("Failed to get ticker from %s", ex.value)
        return results

    async def get_best_price(self, symbol: str) -> tuple[Exchange, Ticker] | None:
        """Find the exchange with the best (lowest) ask price."""
        tickers = await self.get_all_tickers(symbol)
        if not tickers:
            return None
        return min(tickers.items(), key=lambda x: x[1].ask)

    async def get_all_orderbooks(
        self, symbol: str, depth: int = 20
    ) -> dict[Exchange, OrderBook]:
        tasks = {
            ex: asyncio.create_task(adapter.get_orderbook(symbol, depth))
            for ex, adapter in self._adapters.items()
        }
        results: dict[Exchange, OrderBook] = {}
        for ex, task in tasks.items():
            try:
                results[ex] = await task
            except Exception:
                logger.exception("Failed to get orderbook from %s", ex.value)
        return results

    async def get_historical_ohlcv(
        self,
        exchange: Exchange,
        symbol: str,
        interval: str = "1m",
        limit: int = 500,
    ) -> list[OHLCV]:
        adapter = self._adapters.get(exchange)
        if adapter is None:
            raise ValueError(f"Exchange {exchange} not connected")
        return await adapter.get_ohlcv(symbol, interval, limit)

    # ── Streaming (merged multi-exchange) ───────────────────────────

    async def stream_trades_merged(
        self,
        symbol: str,
        on_trade: Callable[[Trade], None] | None = None,
    ) -> AsyncIterator[Trade]:
        """Merge trade streams from all connected exchanges into a single stream."""
        queue: asyncio.Queue[Trade] = asyncio.Queue()

        async def _feed(adapter: ExchangeAdapter) -> None:
            try:
                async for trade in adapter.stream_trades(symbol):
                    await queue.put(trade)
                    if on_trade:
                        on_trade(trade)
            except Exception:
                logger.exception(
                    "Trade stream error from %s", adapter.exchange.value
                )

        tasks = [
            asyncio.create_task(_feed(adapter))
            for adapter in self._adapters.values()
        ]

        try:
            while self._running:
                try:
                    trade = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield trade
                except asyncio.TimeoutError:
                    continue
        finally:
            for t in tasks:
                t.cancel()

    async def __aenter__(self) -> "BTCDataCollector":
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()
