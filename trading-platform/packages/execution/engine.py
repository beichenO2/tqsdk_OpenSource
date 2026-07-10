"""Execution engine — orchestrates order management, position tracking,
broker communication, and risk checks."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Optional

from core.enums.order_status import OrderStatus
from core.models.order import Order
from core.models.trade import Trade

from execution.broker_adapter import BrokerAdapter
from execution.order_manager import OrderManager, OrderRequest
from execution.position_manager import PositionManager

logger = logging.getLogger(__name__)

ORDER_POLL_INTERVAL_S = 2.0


class ExecutionEngine:
    """High-level execution facade wiring order, position, broker, and risk."""

    def __init__(
        self,
        broker: BrokerAdapter,
        event_bus: Any | None = None,
        order_poll_interval: float | None = None,
    ) -> None:
        self._broker = broker
        self.order_manager = OrderManager(broker)
        self.position_manager = PositionManager()

        self.order_manager.on_trade(self.position_manager.on_trade)

        self._running = False
        self._reconcile_task: Optional[asyncio.Task] = None
        self._order_poll_task: Optional[asyncio.Task] = None
        self._order_poll_interval = (
            order_poll_interval if order_poll_interval is not None else ORDER_POLL_INTERVAL_S
        )
        self._event_bus = event_bus

    def _get_event_bus(self):
        if self._event_bus is not None:
            return self._event_bus
        from event_bus import EventBus

        return EventBus.get_instance()

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
        self._order_poll_task = asyncio.create_task(self._order_poll_loop())
        logger.info("Execution engine started")

    async def stop(self) -> None:
        self._running = False
        for task in (self._reconcile_task, self._order_poll_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._reconcile_task = None
        self._order_poll_task = None

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

    async def _order_poll_loop(self) -> None:
        """Poll non-terminal orders and sync fills / status from broker."""
        while self._running:
            try:
                await asyncio.sleep(self._order_poll_interval)
                await self._poll_active_orders()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Order poll loop error")

    async def _poll_active_orders(self) -> None:
        active = self.order_manager.get_active_orders()
        for local in active:
            try:
                broker_order = await self._broker.query_order(local.order_id)
            except Exception as exc:
                logger.warning(
                    "query_order failed for %s (%s: %s) — will retry next poll",
                    local.order_id,
                    type(exc).__name__,
                    exc,
                )
                continue
            if broker_order is None:
                continue
            await self._apply_broker_order_update(local, broker_order)

    async def _apply_broker_order_update(self, local: Order, broker: Order) -> None:
        prev_filled = local.filled_volume
        new_filled = broker.filled_volume

        if new_filled > prev_filled:
            delta = new_filled - prev_filled
            fill_price = broker.avg_fill_price or local.price
            trade = Trade(
                order_id=local.order_id,
                strategy_id=local.strategy_id,
                symbol=local.symbol,
                exchange=local.exchange,
                direction=local.direction,
                offset=local.offset,
                price=fill_price,
                volume=delta,
            )
            self.order_manager.on_fill(trade)
            await self._emit_trade_fill(local, trade, new_filled, broker.status)
            return

        if broker.status != local.status:
            if broker.status == OrderStatus.CANCELLED:
                local.status = OrderStatus.CANCELLED
                local.updated_at = datetime.now(UTC)
                self.order_manager._fire_order_callbacks(local)
                await self._get_event_bus().emit(
                    "order_cancelled",
                    {
                        "order_id": local.order_id,
                        "symbol": local.symbol,
                        "status": broker.status.value,
                    },
                )
            elif broker.status == OrderStatus.REJECTED:
                local.status = OrderStatus.REJECTED
                local.updated_at = datetime.now(UTC)
                self.order_manager._fire_order_callbacks(local)
                await self._get_event_bus().emit(
                    "order_rejected",
                    {
                        "order_id": local.order_id,
                        "symbol": local.symbol,
                        "status": broker.status.value,
                    },
                )
            elif broker.status == OrderStatus.FILLED:
                local.status = OrderStatus.FILLED
                local.updated_at = datetime.now(UTC)
                self.order_manager._fire_order_callbacks(local)

    async def _emit_trade_fill(
        self,
        local: Order,
        trade: Trade,
        filled_volume: int,
        broker_status: OrderStatus,
    ) -> None:
        bus = self._get_event_bus()
        payload = {
            "order_id": local.order_id,
            "symbol": local.symbol,
            "volume": trade.volume,
            "price": str(trade.price),
            "filled_volume": filled_volume,
            "order_status": broker_status.value,
        }
        await bus.emit("trade_fill", payload)
        if broker_status == OrderStatus.PARTIAL_FILLED:
            await bus.emit("order_partially_filled", payload)
