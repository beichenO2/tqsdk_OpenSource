"""Execution engine — orchestrates order management, position tracking,
broker communication, and risk checks."""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from typing import Optional

from core.models.order import Order

from execution.broker_adapter import BrokerAdapter
from execution.order_manager import OrderManager, OrderRequest
from execution.position_manager import PositionManager

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """High-level execution facade wiring order, position, broker, and risk."""

    def __init__(self, broker: BrokerAdapter) -> None:
        self._broker = broker
        self.order_manager = OrderManager(broker)
        self.position_manager = PositionManager()

        self.order_manager.on_trade(self.position_manager.on_trade)

        self._running = False
        self._reconcile_task: Optional[asyncio.Task] = None

    def set_risk_checker(self, check_fn) -> None:
        self.order_manager.set_pre_trade_check(check_fn)

    async def start(self) -> None:
        await self._broker.connect()
        self._running = True

        # 闭市/网关慢时持仓同步可能超时 — 降级启动，交给 reconcile 循环补偿
        try:
            positions = await self._broker.query_positions()
            self.position_manager.sync_from_broker(positions)
        except Exception as e:
            logger.warning(
                "Initial position sync failed (%s: %s) — starting with empty "
                "positions; reconcile loop will retry",
                type(e).__name__, e,
            )

        self._reconcile_task = asyncio.create_task(self._reconcile_loop())
        logger.info("Execution engine started")

    async def stop(self) -> None:
        self._running = False
        if self._reconcile_task:
            self._reconcile_task.cancel()
            try:
                await self._reconcile_task
            except asyncio.CancelledError:
                pass

        active = self.order_manager.get_active_orders()
        if active:
            logger.info("Cancelling %d active orders on shutdown", len(active))
            await self.order_manager.cancel_all()

        await self._broker.disconnect()
        logger.info("Execution engine stopped")

    async def place_order(self, request: OrderRequest) -> Order:
        return await self.order_manager.submit(request)

    async def cancel_order(self, order_id: str) -> bool:
        return await self.order_manager.cancel(order_id)

    async def cancel_all(self, symbol: Optional[str] = None) -> int:
        return await self.order_manager.cancel_all(symbol)

    def update_market_prices(
        self,
        prices: dict[str, Decimal],
        multipliers: Optional[dict[str, int]] = None,
    ) -> None:
        self.position_manager.update_prices(prices, multipliers)

    async def get_account_info(self) -> dict:
        return await self._broker.get_account_info()

    async def _reconcile_loop(self) -> None:
        """Periodically sync positions with broker."""
        while self._running:
            try:
                await asyncio.sleep(10.0)
                positions = await self._broker.query_positions()
                self.position_manager.sync_from_broker(positions)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Reconciliation error")
                await asyncio.sleep(30.0)
