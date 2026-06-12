"""Abstract broker adapter interface.

Wraps broker-specific clients (TqBrokerClient, crypto exchanges) into
a uniform async interface consumed by OrderManager.
"""

from __future__ import annotations

import abc
from decimal import Decimal
from typing import Optional

from core.enums.direction import Direction, Offset
from core.models.order import Order
from core.models.position import Position


class BrokerAdapter(abc.ABC):
    """Uniform interface that all broker implementations must satisfy."""

    @abc.abstractmethod
    async def connect(self) -> None: ...

    @abc.abstractmethod
    async def disconnect(self) -> None: ...

    @abc.abstractmethod
    async def is_connected(self) -> bool: ...

    @abc.abstractmethod
    async def submit_order(
        self,
        symbol: str,
        direction: Direction,
        offset: Offset,
        price: Decimal,
        volume: int,
        strategy_id: str = "",
    ) -> Order: ...

    @abc.abstractmethod
    async def cancel_order(self, order_id: str) -> bool: ...

    @abc.abstractmethod
    async def query_order(self, order_id: str) -> Optional[Order]: ...

    @abc.abstractmethod
    async def query_positions(self) -> list[Position]: ...

    @abc.abstractmethod
    async def get_account_info(self) -> dict: ...
