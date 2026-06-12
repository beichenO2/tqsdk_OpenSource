"""模拟 BrokerAdapter - 回测用模拟交易所适配器。

对接 Ch27 的 BrokerAdapter 抽象接口，在回测中提供模拟撮合。
将回测引擎的 MatchingEngine 包装为符合 BrokerAdapter 接口的实现。
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class SimBrokerAdapter:
    """回测专用模拟 Broker。

    实现 packages/execution/broker_adapter.py 的 BrokerAdapter 接口，
    但使用同步调用（回测不需要真正的网络 I/O）。
    """

    def __init__(
        self,
        commission_rate: Decimal = Decimal("0.0001"),
        slippage_ticks: int = 1,
        tick_size: Decimal = Decimal("1"),
        lot_size: int = 1,
    ) -> None:
        self._commission_rate = commission_rate
        self._slippage_ticks = slippage_ticks
        self._tick_size = tick_size
        self._lot_size = lot_size

        self._orders: dict[str, dict] = {}
        self._trades: list[dict] = []
        self._connected = False
        self._current_prices: dict[str, Decimal] = {}

    async def connect(self) -> None:
        self._connected = True
        logger.info("SimBroker connected")

    async def disconnect(self) -> None:
        self._connected = False
        logger.info("SimBroker disconnected")

    async def is_connected(self) -> bool:
        return self._connected

    async def submit_order(self, request: object) -> dict:
        """提交订单到模拟撮合。

        接受 execution.order_manager.OrderRequest 或普通 dict。
        """
        if hasattr(request, "model_dump"):
            req = request.model_dump()  # type: ignore[union-attr]
        elif hasattr(request, "__dict__"):
            req = vars(request)
        else:
            req = dict(request)  # type: ignore[call-overload]

        order_id = uuid4().hex[:16]
        instrument = req.get("instrument", req.get("symbol", ""))
        price = Decimal(str(req.get("price", 0)))
        volume = int(req.get("volume", 0))

        order = {
            "order_id": order_id,
            "instrument": instrument,
            "direction": req.get("direction", "LONG"),
            "offset": req.get("offset", "OPEN"),
            "order_type": req.get("order_type", "LIMIT"),
            "price": price,
            "volume": volume,
            "filled_volume": 0,
            "avg_fill_price": None,
            "status": "SUBMITTED",
            "created_at": datetime.now(UTC),
        }
        self._orders[order_id] = order

        current_price = self._current_prices.get(instrument)
        if current_price is not None:
            self._try_fill(order, current_price)

        return order

    async def cancel_order(self, order_id: str, broker_order_id: Optional[str] = None) -> bool:
        order = self._orders.get(order_id)
        if order and order["status"] in ("SUBMITTED", "PENDING", "PARTIAL_FILLED"):
            order["status"] = "CANCELLED"
            return True
        return False

    async def query_order(self, order_id: str) -> Optional[dict]:
        return self._orders.get(order_id)

    async def query_trades(self, order_id: str) -> list[dict]:
        return [t for t in self._trades if t["order_id"] == order_id]

    def update_price(self, instrument: str, price: Decimal) -> None:
        """更新行情价格，并尝试撮合挂单。"""
        self._current_prices[instrument] = price
        for order in self._orders.values():
            if order["instrument"] == instrument and order["status"] in ("SUBMITTED", "PARTIAL_FILLED"):
                self._try_fill(order, price)

    def _try_fill(self, order: dict, market_price: Decimal) -> None:
        order_type = order["order_type"]
        direction = order["direction"]
        limit_price = order["price"]

        should_fill = False
        fill_price = market_price

        if order_type == "MARKET":
            slippage = self._tick_size * self._slippage_ticks
            fill_price = market_price + slippage if direction == "LONG" else market_price - slippage
            should_fill = True
        elif order_type == "LIMIT":
            if direction == "LONG" and market_price <= limit_price:
                fill_price = min(limit_price, market_price)
                should_fill = True
            elif direction == "SHORT" and market_price >= limit_price:
                fill_price = max(limit_price, market_price)
                should_fill = True

        if should_fill:
            remaining = order["volume"] - order["filled_volume"]
            commission = fill_price * remaining * self._lot_size * self._commission_rate

            trade = {
                "trade_id": uuid4().hex[:16],
                "order_id": order["order_id"],
                "instrument": order["instrument"],
                "direction": direction,
                "offset": order["offset"],
                "price": fill_price,
                "volume": remaining,
                "commission": commission,
                "traded_at": datetime.now(UTC),
            }
            self._trades.append(trade)

            order["filled_volume"] = order["volume"]
            order["avg_fill_price"] = fill_price
            order["status"] = "FILLED"
