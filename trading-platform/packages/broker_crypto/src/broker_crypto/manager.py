"""BTC Broker Manager - 多交易所统一管理入口

作为 BTC 子系统的核心协调器：
- 通过 factory 管理多交易所适配器生命周期
- 提供统一的交易/行情访问接口
- 与 Ch27 风控模块集成（通过 pre_trade_check 回调）
- 跨所行情聚合（套利分析）
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from .base import ExchangeAdapter
from .factory import create_adapter
from .models import (
    Balance,
    Exchange,
    ExchangeCredentials,
    OrderBook,
    OrderRequest,
    OrderResponse,
    Position,
    Ticker,
    Trade,
)

logger = logging.getLogger(__name__)

PreTradeCheckFn = Callable[[OrderRequest], Awaitable[tuple[bool, str]]]


class BTCBrokerManager:
    """多交易所统一管理器

    使用方式：
        manager = BTCBrokerManager()
        await manager.register_adapter(Exchange.WEEX, weex_adapter)
        order = await manager.place_order(request)
    """

    def __init__(self) -> None:
        self._adapters: dict[Exchange, ExchangeAdapter] = {}
        self._pre_trade_check: PreTradeCheckFn | None = None
        self._running = False

    @property
    def exchanges(self) -> list[Exchange]:
        return list(self._adapters.keys())

    def get_adapter(self, exchange: Exchange) -> ExchangeAdapter:
        adapter = self._adapters.get(exchange)
        if adapter is None:
            raise KeyError(f"未注册的交易所: {exchange.value}")
        return adapter

    # ── 风控集成 ──

    def set_pre_trade_check(self, fn: PreTradeCheckFn) -> None:
        """注入风控前置检查（由 Ch27 RiskEngine 提供）"""
        self._pre_trade_check = fn

    # ── 生命周期 ──

    async def add_exchange(self, credentials: ExchangeCredentials) -> None:
        """创建并连接一个交易所适配器"""
        adapter = create_adapter(credentials)
        await adapter.connect()
        self._adapters[credentials.exchange] = adapter
        logger.info("已连接交易所: %s", credentials.exchange.value)

    async def register_adapter(self, exchange: Exchange, adapter: ExchangeAdapter) -> None:
        """Register a pre-built adapter (e.g. PolarPrivate B-class WEEX)."""
        await adapter.connect()
        self._adapters[exchange] = adapter
        logger.info("已连接交易所: %s", exchange.value)

    async def remove_exchange(self, exchange: Exchange) -> None:
        adapter = self._adapters.pop(exchange, None)
        if adapter:
            await adapter.disconnect()
            logger.info("已断开交易所: %s", exchange.value)

    async def connect_all(self) -> dict[Exchange, bool]:
        results: dict[Exchange, bool] = {}
        for ex, adapter in self._adapters.items():
            try:
                await adapter.connect()
                results[ex] = True
            except Exception:
                results[ex] = False
                logger.exception("连接失败: %s", ex.value)
        self._running = True
        return results

    async def disconnect_all(self) -> None:
        for ex, adapter in self._adapters.items():
            try:
                await adapter.disconnect()
            except Exception:
                logger.exception("断开失败: %s", ex.value)
        self._adapters.clear()
        self._running = False

    # ── 行情 ──

    async def get_ticker(self, exchange: Exchange, symbol: str) -> Ticker:
        return await self.get_adapter(exchange).get_ticker(symbol)

    async def get_all_tickers(self, symbol: str) -> dict[Exchange, Ticker]:
        """从所有交易所获取同一品种行情（跨所套利分析）"""
        results: dict[Exchange, Ticker] = {}
        tasks = {
            ex: asyncio.create_task(adapter.get_ticker(symbol))
            for ex, adapter in self._adapters.items()
        }
        for ex, task in tasks.items():
            try:
                results[ex] = await task
            except Exception:
                logger.warning("获取 %s 行情失败: %s", ex.value, symbol)
        return results

    async def get_orderbook(
        self, exchange: Exchange, symbol: str, depth: int = 20
    ) -> OrderBook:
        return await self.get_adapter(exchange).get_orderbook(symbol, depth)

    async def get_recent_trades(
        self, exchange: Exchange, symbol: str, limit: int = 100
    ) -> list[Trade]:
        return await self.get_adapter(exchange).get_recent_trades(symbol, limit)

    # ── 交易 ──

    async def place_order(self, request: OrderRequest) -> OrderResponse:
        """下单（带风控前置检查）"""
        if self._pre_trade_check is not None:
            allowed, reason = await self._pre_trade_check(request)
            if not allowed:
                logger.warning(
                    "风控拒绝: %s %s %s — %s",
                    request.exchange.value,
                    request.symbol,
                    request.side.value,
                    reason,
                )
                raise PermissionError(f"风控拒绝: {reason}")

        adapter = self.get_adapter(request.exchange)
        return await adapter.place_order(request)

    async def cancel_order(
        self, exchange: Exchange, order_id: str, symbol: str
    ) -> OrderResponse:
        return await self.get_adapter(exchange).cancel_order(order_id, symbol)

    async def get_order(
        self, exchange: Exchange, order_id: str, symbol: str
    ) -> OrderResponse:
        return await self.get_adapter(exchange).get_order(order_id, symbol)

    async def get_open_orders(
        self, exchange: Exchange, symbol: str | None = None
    ) -> list[OrderResponse]:
        return await self.get_adapter(exchange).get_open_orders(symbol)

    # ── 账户 ──

    async def get_balances(self, exchange: Exchange) -> list[Balance]:
        return await self.get_adapter(exchange).get_balances()

    async def get_all_balances(self) -> dict[Exchange, list[Balance]]:
        results: dict[Exchange, list[Balance]] = {}
        for ex, adapter in self._adapters.items():
            try:
                results[ex] = await adapter.get_balances()
            except Exception:
                logger.exception("获取 %s 余额失败", ex.value)
                results[ex] = []
        return results

    async def get_positions(
        self, exchange: Exchange, symbol: str | None = None
    ) -> list[Position]:
        return await self.get_adapter(exchange).get_positions(symbol)

    async def get_all_positions(self) -> dict[Exchange, list[Position]]:
        results: dict[Exchange, list[Position]] = {}
        for ex, adapter in self._adapters.items():
            try:
                results[ex] = await adapter.get_positions()
            except Exception:
                logger.exception("获取 %s 持仓失败", ex.value)
                results[ex] = []
        return results
