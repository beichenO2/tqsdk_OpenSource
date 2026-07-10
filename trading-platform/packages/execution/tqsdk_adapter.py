"""Concrete BrokerAdapter wrapping Ch26's TqBrokerClient."""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Optional

from core.enums.direction import Direction, Offset
from core.enums.market import Exchange
from core.enums.order_status import OrderStatus
from core.models.order import Order
from core.models.position import Position

from execution.broker_adapter import BrokerAdapter

logger = logging.getLogger(__name__)


def _map_gateway_status(row: dict[str, Any]) -> OrderStatus:
    if row.get("is_error"):
        return OrderStatus.REJECTED
    filled = int(row.get("filled_volume", 0))
    volume_left = int(row.get("volume_left", 0))
    tq_status = str(row.get("status", "ALIVE")).upper()
    if tq_status == "ALIVE":
        return OrderStatus.PARTIAL_FILLED if filled > 0 else OrderStatus.SUBMITTED
    if volume_left == 0 and filled > 0:
        return OrderStatus.FILLED
    if filled > 0:
        return OrderStatus.CANCELLED
    last_msg = str(row.get("last_msg", "")).lower()
    if "拒" in last_msg or "error" in last_msg or "fail" in last_msg:
        return OrderStatus.REJECTED
    return OrderStatus.CANCELLED


def _parse_gateway_order(row: dict[str, Any]) -> Order:
    symbol = row["symbol"]
    exchange_str = symbol.split(".")[0] if "." in symbol else "UNKNOWN"
    try:
        exchange = Exchange(exchange_str)
    except ValueError:
        exchange = Exchange.SHFE

    direction = Direction.LONG if row.get("direction") == "BUY" else Direction.SHORT
    offset_raw = str(row.get("offset", "OPEN")).upper()
    if offset_raw == "CLOSETODAY":
        offset = Offset.CLOSE_TODAY
    else:
        offset = Offset(offset_raw)

    filled_volume = int(row.get("filled_volume", 0))
    avg_raw = row.get("avg_price")
    avg_fill_price = Decimal(str(avg_raw)) if avg_raw not in (None, 0, 0.0) else None

    return Order(
        order_id=str(row["order_id"]),
        strategy_id="",
        symbol=symbol,
        exchange=exchange,
        direction=direction,
        offset=offset,
        price=Decimal(str(row.get("limit_price", 0))),
        volume=int(row.get("volume", 0)),
        filled_volume=filled_volume,
        avg_fill_price=avg_fill_price,
        status=_map_gateway_status(row),
    )


class TqSdkBrokerAdapter(BrokerAdapter):
    """Adapts TqBrokerClient to the BrokerAdapter interface.

    This avoids modifying Ch26's code — we wrap it instead.
    """

    def __init__(self, client) -> None:
        self._client = client

    async def connect(self) -> None:
        await self._client.connect()

    async def disconnect(self) -> None:
        await self._client.disconnect()

    async def is_connected(self) -> bool:
        if hasattr(self._client, "_connected"):
            return bool(self._client._connected)
        return self._client._api is not None

    async def submit_order(
        self,
        symbol: str,
        direction: Direction,
        offset: Offset,
        price: Decimal,
        volume: int,
        strategy_id: str = "",
    ) -> Order:
        order_id = await self._client.place_order(
            symbol=symbol,
            direction=direction,
            offset=offset,
            price=price,
            volume=volume,
        )
        exchange_str = symbol.split(".")[0] if "." in symbol else "UNKNOWN"
        try:
            exchange = Exchange(exchange_str)
        except ValueError:
            exchange = Exchange.SHFE

        return Order(
            order_id=order_id,
            strategy_id=strategy_id,
            symbol=symbol,
            exchange=exchange,
            direction=direction,
            offset=offset,
            price=price,
            volume=volume,
            status=OrderStatus.SUBMITTED,
        )

    async def cancel_order(self, order_id: str) -> bool:
        return await self._client.cancel_order(order_id)

    async def query_order(self, order_id: str) -> Optional[Order]:
        if not hasattr(self._client, "query_order"):
            return None
        row = await self._client.query_order(order_id)
        if row is None:
            return None
        return _parse_gateway_order(row)

    async def query_positions(self) -> list[Position]:
        return await self._client.get_positions()

    async def get_account_info(self) -> dict:
        return await self._client.get_account_info()
