"""实时数据源 — 统一接口，支持期货(TqSdk)和加密(Binance WebSocket)。

LiveFeed 提供统一的 on_bar 回调接口，底层可对接：
- TqSdk API 订阅期货行情
- Binance WebSocket 订阅加密 K 线
- 或混合模式同时订阅两个市场
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)

OnBarCallback = Callable[[str, dict[str, Any]], Any]


class TqSdkLiveFeed:
    """期货实时数据源 — 通过 TqSdk API 获取 K 线。

    TqSdk 使用 asyncio 事件循环轮询 wait_update() 获取最新数据。
    每根 K 线闭合时调用 on_bar 回调。
    """

    def __init__(
        self,
        symbols: list[str],
        interval: str = "5m",
        on_bar: OnBarCallback | None = None,
        tq_api: Any = None,
    ) -> None:
        self.symbols = symbols
        self.interval = interval
        self._on_bar = on_bar
        self._tq_api = tq_api
        self._running = False
        self._klines: dict[str, Any] = {}
        self._last_ids: dict[str, int] = {}

    def set_tq_api(self, api: Any) -> None:
        self._tq_api = api

    async def start(self) -> None:
        if self._tq_api is None:
            logger.error("TqSdk API not set — cannot start live feed")
            return

        self._running = True
        dur_map = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600}
        duration = dur_map.get(self.interval, 300)

        for symbol in self.symbols:
            kl = self._tq_api.get_kline_serial(symbol, duration, data_length=10)
            self._klines[symbol] = kl
            self._last_ids[symbol] = kl.iloc[-1].id if len(kl) > 0 else -1

        logger.info("TqSdkLiveFeed started: %d symbols, interval=%s", len(self.symbols), self.interval)

        while self._running:
            try:
                self._tq_api.wait_update()
                for symbol in self.symbols:
                    kl = self._klines[symbol]
                    if len(kl) == 0:
                        continue
                    current_id = kl.iloc[-1].id
                    if current_id != self._last_ids.get(symbol, -1):
                        self._last_ids[symbol] = current_id
                        row = kl.iloc[-2] if len(kl) > 1 else kl.iloc[-1]
                        bar = {
                            "timestamp": datetime.fromtimestamp(
                                row.datetime / 1e9, tz=timezone.utc
                            ).isoformat(),
                            "open": float(row.open),
                            "high": float(row.high),
                            "low": float(row.low),
                            "close": float(row.close),
                            "volume": float(row.volume),
                        }
                        if self._on_bar:
                            result = self._on_bar(symbol, bar)
                            if asyncio.iscoroutine(result):
                                await result
            except Exception as e:
                if self._running:
                    logger.warning("TqSdkLiveFeed error: %s, retrying in 5s", e)
                    await asyncio.sleep(5)

    async def stop(self) -> None:
        self._running = False
        logger.info("TqSdkLiveFeed stopped")


class TqGatewayLiveFeed:
    """Poll TqSdk Gateway for closed kline bars (no local TqApi)."""

    def __init__(
        self,
        symbols: list[str],
        interval: str = "5m",
        on_bar: OnBarCallback | None = None,
        gateway_url: str | None = None,
    ) -> None:
        import os

        self.symbols = symbols
        self.interval = interval
        self._on_bar = on_bar
        self._gateway_url = (gateway_url or os.getenv("TQSDK_GATEWAY_URL", "http://127.0.0.1:12890")).rstrip("/")
        self._running = False
        self._last_ts: dict[str, int] = {}

    async def start(self) -> None:
        import httpx

        self._running = True
        dur_map = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600}
        duration = dur_map.get(self.interval, 300)
        poll_seconds = min(max(duration // 10, 5), 60)
        logger.info(
            "TqGatewayLiveFeed started: %d symbols, interval=%s, gateway=%s",
            len(self.symbols), self.interval, self._gateway_url,
        )

        async with httpx.AsyncClient(base_url=self._gateway_url, timeout=30.0) as client:
            while self._running:
                for symbol in self.symbols:
                    try:
                        resp = await client.get(
                            f"/api/v1/market/klines/{symbol}",
                            params={"duration": duration, "length": 3},
                        )
                        resp.raise_for_status()
                        items = resp.json().get("items", [])
                        if len(items) < 2:
                            continue
                        closed = items[-2]
                        ts = int(closed["datetime"])
                        if self._last_ts.get(symbol) == ts:
                            continue
                        self._last_ts[symbol] = ts
                        bar = {
                            "timestamp": datetime.fromtimestamp(ts / 1e9, tz=timezone.utc).isoformat(),
                            "open": float(closed["open"]),
                            "high": float(closed["high"]),
                            "low": float(closed["low"]),
                            "close": float(closed["close"]),
                            "volume": float(closed["volume"]),
                        }
                        if self._on_bar:
                            result = self._on_bar(symbol, bar)
                            if asyncio.iscoroutine(result):
                                await result
                    except Exception as e:
                        if self._running:
                            logger.warning("TqGatewayLiveFeed error for %s: %s", symbol, e)
                await asyncio.sleep(poll_seconds)

    async def stop(self) -> None:
        self._running = False
        logger.info("TqGatewayLiveFeed stopped")


class UnifiedLiveFeed:
    """统一实时数据源 — 同时支持期货和加密市场。

    根据 symbol 格式自动路由：
    - 含 '.' 的（如 'SHFE.cu2401'）→ TqSdk feed
    - 全大写字母（如 'BTCUSDT'）→ Binance feed
    """

    def __init__(
        self,
        futures_symbols: list[str] | None = None,
        crypto_symbols: list[str] | None = None,
        futures_interval: str = "5m",
        crypto_interval: str = "1m",
        on_bar: OnBarCallback | None = None,
        tq_api: Any = None,
        gateway_url: str | None = None,
    ) -> None:
        self._on_bar = on_bar
        self._futures_feed: TqSdkLiveFeed | TqGatewayLiveFeed | None = None
        self._crypto_feed: Any = None
        self._tasks: list[asyncio.Task] = []

        if futures_symbols:
            if tq_api is not None:
                self._futures_feed = TqSdkLiveFeed(
                    symbols=futures_symbols,
                    interval=futures_interval,
                    on_bar=on_bar,
                    tq_api=tq_api,
                )
            else:
                self._futures_feed = TqGatewayLiveFeed(
                    symbols=futures_symbols,
                    interval=futures_interval,
                    on_bar=on_bar,
                    gateway_url=gateway_url,
                )

        if crypto_symbols:
            from .realtime_feed import BinanceKlineFeed
            self._crypto_feed = BinanceKlineFeed(
                symbols=crypto_symbols,
                interval=crypto_interval,
                on_bar=on_bar,
            )

    async def start(self) -> None:
        if self._futures_feed:
            self._tasks.append(asyncio.create_task(self._futures_feed.start()))
        if self._crypto_feed:
            self._tasks.append(asyncio.create_task(self._crypto_feed.start()))
        logger.info(
            "UnifiedLiveFeed started: futures=%s, crypto=%s",
            bool(self._futures_feed), bool(self._crypto_feed),
        )

    async def stop(self) -> None:
        if self._futures_feed:
            await self._futures_feed.stop()
        if self._crypto_feed:
            await self._crypto_feed.stop()
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()
        logger.info("UnifiedLiveFeed stopped")
