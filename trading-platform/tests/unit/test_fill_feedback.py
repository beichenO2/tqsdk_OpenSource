"""TDD — fill feedback loop: query_order, polling, EventBus."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.enums.direction import Direction, Offset
from core.enums.market import Exchange
from core.enums.order_status import OrderStatus
from event_bus import EventBus
from core.models.order import Order
from execution.engine import ExecutionEngine
from execution.tqsdk_adapter import TqSdkBrokerAdapter


class _RecordingBus(EventBus):
    def __init__(self) -> None:
        super().__init__()
        self.emitted: list[tuple[str, dict[str, Any]]] = []

    async def emit(self, event_type: str, data: dict[str, Any]) -> None:
        self.emitted.append((event_type, data))
        await super().emit(event_type, data)


class _PollableBroker:
    """Minimal broker for order-poll tests."""

    def __init__(self) -> None:
        self._connected = False
        self._orders: dict[str, Order] = {}
        self.query_order_calls = 0
        self._query_error: Exception | None = None

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def is_connected(self) -> bool:
        return self._connected

    async def submit_order(
        self,
        symbol: str,
        direction: Direction,
        offset: Offset,
        price: Decimal,
        volume: int,
        strategy_id: str = "",
    ) -> Order:
        order = Order(
            order_id="broker-1",
            strategy_id=strategy_id or "s1",
            symbol=symbol,
            exchange=Exchange.SHFE,
            direction=direction,
            offset=offset,
            price=price,
            volume=volume,
            status=OrderStatus.SUBMITTED,
        )
        self._orders[order.order_id] = order
        return order

    async def cancel_order(self, order_id: str) -> bool:
        return True

    def set_broker_snapshot(self, order_id: str, snapshot: Order) -> None:
        self._orders[order_id] = snapshot

    def set_query_error(self, exc: Exception) -> None:
        self._query_error = exc

    async def query_order(self, order_id: str) -> Optional[Order]:
        self.query_order_calls += 1
        if self._query_error is not None:
            raise self._query_error
        return self._orders.get(order_id)

    async def query_positions(self) -> list:
        return []

    async def get_account_info(self) -> dict:
        return {"balance": 0, "available": 0}


@pytest.mark.asyncio
async def test_query_order_parses_gateway_http_response() -> None:
    client = MagicMock()
    client._connected = True
    client.query_order = AsyncMock(
        return_value={
            "order_id": "gw-42",
            "symbol": "SHFE.rb2510",
            "direction": "BUY",
            "offset": "OPEN",
            "status": "FINISHED",
            "volume": 2,
            "filled_volume": 2,
            "volume_left": 0,
            "avg_price": 3501.0,
            "limit_price": 3500.0,
            "last_msg": "全部成交",
        }
    )

    adapter = TqSdkBrokerAdapter(client)
    order = await adapter.query_order("gw-42")

    assert order is not None
    assert order.order_id == "gw-42"
    assert order.symbol == "SHFE.rb2510"
    assert order.direction == Direction.LONG
    assert order.offset == Offset.OPEN
    assert order.status == OrderStatus.FILLED
    assert order.filled_volume == 2
    assert order.avg_fill_price == Decimal("3501.0")
    client.query_order.assert_awaited_once_with("gw-42")


@pytest.mark.asyncio
async def test_poll_loop_filled_order_triggers_on_fill_and_event() -> None:
    broker = _PollableBroker()
    bus = _RecordingBus()
    engine = ExecutionEngine(broker, event_bus=bus, order_poll_interval=0.05)

    await engine.start()
    try:
        local = await engine.place_order(
            __import__("execution.order_manager", fromlist=["OrderRequest"]).OrderRequest(
                symbol="SHFE.rb2510",
                exchange="SHFE",
                direction=Direction.LONG,
                offset=Offset.OPEN,
                price=Decimal("3500"),
                volume=2,
                strategy_id="s1",
            )
        )
        assert local.order_id == "broker-1"

        broker.set_broker_snapshot(
            "broker-1",
            Order(
                order_id="broker-1",
                strategy_id="s1",
                symbol="SHFE.rb2510",
                exchange=Exchange.SHFE,
                direction=Direction.LONG,
                offset=Offset.OPEN,
                price=Decimal("3500"),
                volume=2,
                filled_volume=2,
                avg_fill_price=Decimal("3501"),
                status=OrderStatus.FILLED,
            ),
        )

        on_fill = MagicMock()
        engine.order_manager.on_fill = on_fill  # type: ignore[method-assign]

        for _ in range(30):
            await __import__("asyncio").sleep(0.05)
            if on_fill.called:
                break

        assert on_fill.called, "expected on_fill when broker reports FILLED"
        fill_types = [t for t, _ in bus.emitted]
        assert "trade_fill" in fill_types
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_poll_loop_survives_gateway_connection_error() -> None:
    broker = _PollableBroker()
    broker.set_query_error(ConnectionError("gateway unreachable"))
    engine = ExecutionEngine(broker, order_poll_interval=0.05)

    await engine.start()
    try:
        await engine.place_order(
            __import__("execution.order_manager", fromlist=["OrderRequest"]).OrderRequest(
                symbol="SHFE.rb2510",
                exchange="SHFE",
                direction=Direction.LONG,
                offset=Offset.OPEN,
                price=Decimal("3500"),
                volume=1,
                strategy_id="s1",
            )
        )

        before = broker.query_order_calls
        await __import__("asyncio").sleep(0.2)
        after = broker.query_order_calls

        assert after > before, "poll loop should retry after connection errors"
    finally:
        await engine.stop()
