"""CryptoMarketAdapter — 将 broker_crypto 行情桥接为核心 Bar/Tick 模型。

与期货侧 TqMarketAdapter 保持相同接口，使 MarketService 和 FeatureEngine
可以透明地消费 BTC 数据。
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from core.models.bar import Bar
from core.models.tick import Tick

from .base import ExchangeAdapter
from .factory import create_adapter
from .models import ExchangeCredentials, OHLCV, Ticker

logger = logging.getLogger(__name__)

_DURATION_TO_INTERVAL: dict[int, str] = {
    60: "1m",
    180: "3m",
    300: "5m",
    900: "15m",
    1800: "30m",
    3600: "1h",
    7200: "2h",
    14400: "4h",
    21600: "6h",
    28800: "8h",
    43200: "12h",
    86400: "1d",
    259200: "3d",
    604800: "1w",
}

_CRYPTO_INSTRUMENTS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
]


class CryptoMarketAdapter:
    """行情适配层：从加密交易所获取数据并转为核心 Bar/Tick 模型。

    接口与 TqMarketAdapter 对齐，支持同一个 MarketService 使用。
    """

    def __init__(
        self,
        credentials: ExchangeCredentials | None = None,
        adapter: ExchangeAdapter | None = None,
    ) -> None:
        self._adapter = adapter
        self._credentials = credentials

    def set_adapter(self, adapter: ExchangeAdapter) -> None:
        self._adapter = adapter

    async def connect(self) -> None:
        if self._adapter is None and self._credentials is not None:
            self._adapter = create_adapter(self._credentials)
        if self._adapter is not None:
            await self._adapter.connect()

    async def disconnect(self) -> None:
        if self._adapter is not None:
            await self._adapter.disconnect()

    async def get_quote(self, symbol: str) -> Tick | None:
        if self._adapter is None:
            return None
        try:
            ticker: Ticker = await self._adapter.get_ticker(symbol)
            return Tick(
                symbol=symbol,
                datetime=ticker.timestamp,
                last_price=ticker.last,
                highest=ticker.ask,
                lowest=ticker.bid,
                volume=int(ticker.volume_24h),
                amount=ticker.last * ticker.volume_24h,
                bid_price1=ticker.bid,
                bid_volume1=None,
                ask_price1=ticker.ask,
                ask_volume1=None,
            )
        except Exception:
            logger.exception("Failed to get quote for %s", symbol)
            return None

    async def get_klines(
        self,
        symbol: str,
        *,
        duration_seconds: int = 60,
        data_length: int = 200,
    ) -> list[Bar]:
        if self._adapter is None:
            return []

        interval = _DURATION_TO_INTERVAL.get(duration_seconds)
        if interval is None:
            interval = f"{duration_seconds // 60}m" if duration_seconds < 3600 else "1h"

        try:
            candles: list[OHLCV] = await self._adapter.get_ohlcv(
                symbol, interval=interval, limit=data_length
            )
            return [
                Bar(
                    symbol=symbol,
                    datetime=c.timestamp,
                    open=c.open,
                    high=c.high,
                    low=c.low,
                    close=c.close,
                    volume=int(c.volume),
                    duration_seconds=duration_seconds,
                )
                for c in candles
            ]
        except Exception:
            logger.exception("Failed to get klines for %s", symbol)
            return []

    async def list_instruments(
        self, exchange_id: str | None = None
    ) -> list[dict[str, str]]:
        """列出可交易的加密货币对。

        exchange_id 在这里被忽略（加密侧通过 credentials 决定交易所）。
        """
        return [{"symbol": s} for s in _CRYPTO_INSTRUMENTS]

    async def subscribe_ticks(self, symbol: str) -> AsyncIterator[Tick]:
        """订阅实时行情（通过 WebSocket trade stream 模拟）。"""
        if self._adapter is None:
            return

        async for trade in self._adapter.stream_trades(symbol):
            yield Tick(
                symbol=symbol,
                datetime=trade.timestamp,
                last_price=trade.price,
                highest=trade.price,
                lowest=trade.price,
                volume=int(trade.quantity),
                amount=trade.price * trade.quantity,
            )

    async def get_ohlcv_dataframe(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """返回适合 FeatureEngine 消费的 dict 列表（OHLCV + exchange 元数据）。

        FeatureEngine 期望 DataFrame 包含 open/high/low/close/volume 列。
        """
        if self._adapter is None:
            return []

        candles = await self._adapter.get_ohlcv(symbol, interval=interval, limit=limit)
        return [
            {
                "datetime": c.timestamp,
                "open": float(c.open),
                "high": float(c.high),
                "low": float(c.low),
                "close": float(c.close),
                "volume": float(c.volume),
                "exchange": c.exchange.value,
                "symbol": c.symbol,
            }
            for c in candles
        ]

    async def __aenter__(self) -> CryptoMarketAdapter:
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.disconnect()
