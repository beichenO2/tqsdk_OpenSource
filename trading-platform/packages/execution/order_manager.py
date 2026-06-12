"""Order lifecycle management — create, track, cancel, and reconcile orders."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import Callable, Optional

from core.enums.direction import Direction, Offset
from core.enums.order_status import OrderStatus
from core.models.order import Order
from core.models.trade import Trade

from execution.broker_adapter import BrokerAdapter

logger = logging.getLogger(__name__)

OrderCallback = Callable[[Order], None]
TradeCallback = Callable[[Trade], None]
PreTradeCheck = Callable[["OrderRequest"], tuple[bool, str]]


class OrderRequest:
    """Internal DTO for order submission."""

    __slots__ = (
        "symbol", "exchange", "direction", "offset",
        "price", "volume", "strategy_id", "tag",
    )

    def __init__(
        self,
        symbol: str,
        exchange: str,
        direction: Direction,
        offset: Offset,
        price: Decimal,
        volume: int,
        strategy_id: str = "",
        tag: str = "",
    ) -> None:
        self.symbol = symbol
        self.exchange = exchange
        self.direction = direction
        self.offset = offset
        self.price = price
        self.volume = volume
        self.strategy_id = strategy_id
        self.tag = tag


class OrderManager:
    """Manages the full order lifecycle."""

    def __init__(self, broker: BrokerAdapter) -> None:
        self._broker = broker
        self._orders: dict[str, Order] = {}
        self._on_order_callbacks: list[OrderCallback] = []
        self._on_trade_callbacks: list[TradeCallback] = []
        self._pre_trade_check: Optional[PreTradeCheck] = None
        self._lock = asyncio.Lock()

    def set_pre_trade_check(self, check_fn: PreTradeCheck) -> None:
        self._pre_trade_check = check_fn

    def on_order(self, callback: OrderCallback) -> None:
        self._on_order_callbacks.append(callback)

    def on_trade(self, callback: TradeCallback) -> None:
        self._on_trade_callbacks.append(callback)

    async def submit(self, request: OrderRequest) -> Order:
        """Submit a new order through risk checks → broker."""
        if self._pre_trade_check:
            passed, reason = self._pre_trade_check(request)
            if not passed:
                from core.enums.market import Exchange as ExchangeEnum
                try:
                    exchange = ExchangeEnum(request.exchange)
                except ValueError:
                    exchange = ExchangeEnum.SHFE

                order = Order(
                    strategy_id=request.strategy_id,
                    symbol=request.symbol,
                    exchange=exchange,
                    direction=request.direction,
                    offset=request.offset,
                    price=request.price,
                    volume=request.volume,
                    status=OrderStatus.REJECTED,
                )
                logger.warning(
                    "Order rejected by risk: symbol=%s reason=%s",
                    request.symbol, reason,
                )
                self._fire_order_callbacks(order)
                return order

        async with self._lock:
            broker_order = await self._broker.submit_order(
                symbol=request.symbol,
                direction=request.direction,
                offset=request.offset,
                price=request.price,
                volume=request.volume,
                strategy_id=request.strategy_id,
            )
            self._orders[broker_order.order_id] = broker_order
            logger.info(
                "Order submitted: id=%s symbol=%s dir=%s vol=%d",
                broker_order.order_id, broker_order.symbol,
                broker_order.direction.value, broker_order.volume,
            )
            self._fire_order_callbacks(broker_order)
            return broker_order

    async def cancel(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if not order:
            logger.warning("Cancel failed: order %s not found", order_id)
            return False

        active_statuses = {OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL_FILLED}
        if order.status not in active_statuses:
            logger.warning("Cancel failed: order %s status=%s", order_id, order.status)
            return False

        success = await self._broker.cancel_order(order_id)
        if success:
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now(UTC)
            self._fire_order_callbacks(order)
            logger.info("Order cancelled: %s", order_id)
        return success

    async def cancel_all(self, symbol: Optional[str] = None) -> int:
        cancelled = 0
        active_statuses = {OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL_FILLED}
        for order_id, order in list(self._orders.items()):
            if order.status not in active_statuses:
                continue
            if symbol and order.symbol != symbol:
                continue
            if await self.cancel(order_id):
                cancelled += 1
        return cancelled

    def on_fill(self, trade: Trade) -> None:
        """Process an incoming fill notification from the broker."""
        order = self._orders.get(trade.order_id)
        if not order:
            logger.error("Fill for unknown order: %s", trade.order_id)
            return

        order.filled_volume += trade.volume
        if order.avg_fill_price is None:
            order.avg_fill_price = trade.price
        else:
            prev_total = order.avg_fill_price * (order.filled_volume - trade.volume)
            order.avg_fill_price = (prev_total + trade.price * trade.volume) / Decimal(order.filled_volume)

        order.status = OrderStatus.FILLED if order.filled_volume >= order.volume else OrderStatus.PARTIAL_FILLED
        order.updated_at = datetime.now(UTC)

        self._fire_order_callbacks(order)
        self._fire_trade_callbacks(trade)
        logger.info(
            "Fill: order=%s vol=%d price=%s filled=%d/%d",
            trade.order_id, trade.volume, trade.price,
            order.filled_volume, order.volume,
        )

    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    def get_active_orders(self, symbol: Optional[str] = None) -> list[Order]:
        active_statuses = {OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL_FILLED}
        orders = [o for o in self._orders.values() if o.status in active_statuses]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    def get_all_orders(self) -> list[Order]:
        return list(self._orders.values())

    def _fire_order_callbacks(self, order: Order) -> None:
        for cb in self._on_order_callbacks:
            try:
                cb(order)
            except Exception:
                logger.exception("Error in order callback")

    def _fire_trade_callbacks(self, trade: Trade) -> None:
        for cb in self._on_trade_callbacks:
            try:
                cb(trade)
            except Exception:
                logger.exception("Error in trade callback")
