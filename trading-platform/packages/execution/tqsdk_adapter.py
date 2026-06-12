"""Concrete BrokerAdapter wrapping Ch26's TqBrokerClient."""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Optional

from core.enums.direction import Direction, Offset
from core.enums.market import Exchange
from core.enums.order_status import OrderStatus
from core.models.order import Order
from core.models.position import Position

from execution.broker_adapter import BrokerAdapter

logger = logging.getLogger(__name__)


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
        return None

    async def query_positions(self) -> list[Position]:
        return await self._client.get_positions()

    async def get_account_info(self) -> dict:
        return await self._client.get_account_info()
