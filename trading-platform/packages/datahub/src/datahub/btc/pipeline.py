"""BTC 数据管道 — 编排采集→清洗→存储的完整流程。"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from broker_crypto import (
    ExchangeCredentials,
    OHLCV,
    Trade,
)

from .collector import BTCDataCollector
from .storage import ParquetStorage

logger = logging.getLogger(__name__)


class BTCDataPipeline:
    """End-to-end BTC data pipeline: collect → clean → store."""

    def __init__(
        self,
        credentials: list[ExchangeCredentials],
        storage_dir: str = "./data/btc",
    ) -> None:
        self._collector = BTCDataCollector(credentials)
        self._storage = ParquetStorage(storage_dir)
        self._running = False

    # ── Batch ingestion ─────────────────────────────────────────────

    async def ingest_ohlcv(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 500,
    ) -> dict[str, Any]:
        """Fetch OHLCV from all exchanges and store in Parquet."""
        results: dict[str, Any] = {}

        for exchange, adapter in self._collector._adapters.items():
            try:
                candles = await adapter.get_ohlcv(symbol, interval, limit)
                cleaned = self._clean_ohlcv(candles)
                records = [self._ohlcv_to_dict(c) for c in cleaned]

                path = self._storage.write_ohlcv(
                    records,
                    exchange=exchange.value,
                    symbol=symbol,
                    interval=interval,
                )
                results[exchange.value] = {
                    "count": len(records),
                    "path": str(path),
                }
                logger.info(
                    "Ingested %d candles from %s for %s/%s",
                    len(records), exchange.value, symbol, interval,
                )
            except Exception:
                logger.exception("OHLCV ingestion failed for %s", exchange.value)
                results[exchange.value] = {"error": True}

        return results

    async def ingest_trades(self, symbol: str, limit: int = 1000) -> dict[str, Any]:
        """Fetch recent trades from all exchanges and store."""
        results: dict[str, Any] = {}

        for exchange, adapter in self._collector._adapters.items():
            try:
                trades = await adapter.get_recent_trades(symbol, limit)
                records = [self._trade_to_dict(t) for t in trades]

                path = self._storage.write_trades(
                    records,
                    exchange=exchange.value,
                    symbol=symbol,
                )
                results[exchange.value] = {
                    "count": len(records),
                    "path": str(path),
                }
            except Exception:
                logger.exception("Trade ingestion failed for %s", exchange.value)
                results[exchange.value] = {"error": True}

        return results

    async def ingest_orderbook_snapshot(
        self, symbol: str, depth: int = 20
    ) -> dict[str, Any]:
        """Capture orderbook snapshots from all exchanges."""
        results: dict[str, Any] = {}
        books = await self._collector.get_all_orderbooks(symbol, depth)

        for exchange, book in books.items():
            snapshot = {
                "exchange": exchange.value,
                "symbol": symbol,
                "bids": [(str(p), str(q)) for p, q in book.bids],
                "asks": [(str(p), str(q)) for p, q in book.asks],
                "timestamp": book.timestamp.isoformat(),
            }
            path = self._storage.write_orderbook_snapshot(
                snapshot, exchange=exchange.value, symbol=symbol
            )
            results[exchange.value] = {"path": str(path)}

        return results

    # ── Continuous streaming ingestion ──────────────────────────────

    async def run_continuous(
        self,
        symbol: str,
        trade_buffer_size: int = 500,
        flush_interval_sec: float = 60.0,
    ) -> None:
        """Run continuous trade ingestion: buffer trades, flush periodically."""
        self._running = True
        buffer: list[dict[str, Any]] = []
        last_flush = asyncio.get_running_loop().time()

        logger.info("Starting continuous BTC data pipeline for %s", symbol)

        async for trade in self._collector.stream_trades_merged(symbol):
            if not self._running:
                break

            buffer.append(self._trade_to_dict(trade))
            now = asyncio.get_running_loop().time()

            should_flush = (
                len(buffer) >= trade_buffer_size
                or (now - last_flush) >= flush_interval_sec
            )

            if should_flush and buffer:
                exchange_groups: dict[str, list[dict]] = {}
                for rec in buffer:
                    exchange_groups.setdefault(rec["exchange"], []).append(rec)

                for ex, records in exchange_groups.items():
                    self._storage.write_trades(records, exchange=ex, symbol=symbol)

                logger.info("Flushed %d trades for %s", len(buffer), symbol)
                buffer.clear()
                last_flush = now

        if buffer:
            for ex_name in {r["exchange"] for r in buffer}:
                recs = [r for r in buffer if r["exchange"] == ex_name]
                self._storage.write_trades(recs, exchange=ex_name, symbol=symbol)

    def stop(self) -> None:
        self._running = False

    # ── Data cleaning ───────────────────────────────────────────────

    @staticmethod
    def _clean_ohlcv(candles: list[OHLCV]) -> list[OHLCV]:
        """Remove duplicates and sort by timestamp."""
        seen: set[str] = set()
        cleaned: list[OHLCV] = []
        for c in candles:
            key = f"{c.exchange}_{c.symbol}_{c.timestamp.isoformat()}"
            if key not in seen:
                seen.add(key)
                cleaned.append(c)
        cleaned.sort(key=lambda c: c.timestamp)
        return cleaned

    @staticmethod
    def _ohlcv_to_dict(c: OHLCV) -> dict[str, Any]:
        return {
            "exchange": c.exchange.value,
            "symbol": c.symbol,
            "interval": c.interval,
            "open": float(c.open),
            "high": float(c.high),
            "low": float(c.low),
            "close": float(c.close),
            "volume": float(c.volume),
            "timestamp": c.timestamp.isoformat(),
        }

    @staticmethod
    def _trade_to_dict(t: Trade) -> dict[str, Any]:
        return {
            "exchange": t.exchange.value,
            "symbol": t.symbol,
            "trade_id": t.trade_id,
            "side": t.side.value,
            "price": float(t.price),
            "quantity": float(t.quantity),
            "timestamp": t.timestamp.isoformat(),
        }

    # ── Lifecycle ───────────────────────────────────────────────────

    async def start(self) -> None:
        await self._collector.start()

    async def close(self) -> None:
        self.stop()
        await self._collector.stop()

    async def __aenter__(self) -> "BTCDataPipeline":
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
