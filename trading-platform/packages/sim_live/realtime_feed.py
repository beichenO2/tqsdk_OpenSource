"""实时数据源 — 从 Binance WebSocket 获取 K 线数据驱动模拟实盘。

使用 Binance 的公开 kline WebSocket stream:
- wss://stream.binance.com:9443/ws/{symbol}@kline_{interval}

每根 K 线闭合时推送 bar 到 PaperScheduler。
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)


class BinanceKlineFeed:
    """Binance WebSocket K 线数据流。

    用法:
        feed = BinanceKlineFeed(
            symbols=["btcusdt", "ethusdt"],
            interval="1m",
            on_bar=my_callback,
        )
        await feed.start()
    """

    WS_BASE = "wss://stream.binance.com:9443/stream?streams="

    def __init__(
        self,
        symbols: list[str],
        interval: str = "1m",
        on_bar: Callable[[str, dict[str, Any]], Any] | None = None,
    ) -> None:
        self.symbols = [s.lower() for s in symbols]
        self.interval = interval
        self._on_bar = on_bar
        self._ws = None
        self._running = False

    def _build_url(self) -> str:
        streams = "/".join(f"{s}@kline_{self.interval}" for s in self.symbols)
        return f"{self.WS_BASE}{streams}"

    async def start(self) -> None:
        """启动 WebSocket 连接。需要安装 websockets 库。"""
        try:
            import websockets
        except ImportError:
            logger.error("需要安装 websockets: pip install websockets")
            return

        self._running = True
        url = self._build_url()
        logger.info("Connecting to Binance: %s", url)

        while self._running:
            try:
                async with websockets.connect(url) as ws:
                    self._ws = ws
                    logger.info("Binance WS connected, %d streams", len(self.symbols))
                    async for msg in ws:
                        if not self._running:
                            break
                        await self._handle_message(msg)
            except Exception as e:
                if self._running:
                    logger.warning("WS disconnected: %s, reconnecting in 5s...", e)
                    await asyncio.sleep(5)

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()

    async def _handle_message(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        # combined stream format: {"stream": "...", "data": {...}}
        kline_data = data.get("data", data)
        k = kline_data.get("k")
        if not k:
            return

        # 只在 K 线闭合时处理
        if not k.get("x", False):
            return

        symbol = k["s"].upper()  # e.g., "BTCUSDT"
        bar = {
            "timestamp": datetime.fromtimestamp(k["t"] / 1000, tz=timezone.utc).isoformat(),
            "open": float(k["o"]),
            "high": float(k["h"]),
            "low": float(k["l"]),
            "close": float(k["c"]),
            "volume": float(k["v"]),
            "taker_buy_volume": float(k.get("V", 0)),
        }

        logger.debug("Bar closed: %s %s close=%.2f", symbol, bar["timestamp"], bar["close"])

        if self._on_bar:
            result = self._on_bar(symbol, bar)
            if asyncio.iscoroutine(result):
                await result


class RealtimePaperEngine:
    """整合 BinanceKlineFeed + PaperScheduler 的实时模拟引擎。

    用法:
        engine = RealtimePaperEngine(
            symbols=["BTCUSDT", "ETHUSDT"],
            interval="1m",
        )
        await engine.start()
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        interval: str = "1m",
        report_interval: int = 60,
    ) -> None:
        if symbols is None:
            symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
        self.symbols = symbols
        self.interval = interval
        self.report_interval = report_interval
        self._bar_count = 0

        from .account_manager import AccountManager
        from .strategy_factory import create_all_strategies
        from .strategy_catalog import get_catalog
        from .paper_scheduler import PaperScheduler

        self.accounts = AccountManager(crypto_count=100, futures_count=0)
        self.strategies = create_all_strategies(market="crypto")

        for entry in get_catalog("crypto"):
            self.accounts.assign_strategy(entry["account_id"], entry["name"])

        self.scheduler = PaperScheduler(self.accounts, self.strategies)

        self.feed = BinanceKlineFeed(
            symbols=symbols,
            interval=interval,
            on_bar=self._on_bar,
        )

    async def _on_bar(self, symbol: str, bar: dict[str, Any]) -> None:
        """处理来自 Binance 的实时 K 线。"""
        self._bar_count += 1
        market_data = {symbol: bar}

        stats = await self.scheduler.run_bar(bar["timestamp"], market_data)

        if self._bar_count % self.report_interval == 0:
            summary = self.accounts.summary()
            logger.info(
                "Live bar #%d | signals=%d | crypto_avg=%.2f%%",
                self._bar_count,
                stats["total_signals"],
                summary["crypto"]["avg_return"],
            )
            # 保存快照
            from .reporter import PaperReporter
            reporter = PaperReporter(self.accounts)
            reporter.save_full_report("output/paper_trading_live")

    async def start(self) -> None:
        logger.info(
            "RealtimePaperEngine starting: %d symbols, interval=%s",
            len(self.symbols), self.interval,
        )
        await self.feed.start()

    async def stop(self) -> None:
        await self.feed.stop()
        logger.info("RealtimePaperEngine stopped after %d bars", self._bar_count)
