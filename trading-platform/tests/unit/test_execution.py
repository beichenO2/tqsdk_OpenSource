"""Unit tests for OrderManager and ExecutionEngine (mock broker)."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from core.enums.direction import Direction, Offset
from core.enums.order_status import OrderStatus
from execution.engine import ExecutionEngine
from execution.order_manager import OrderManager, OrderRequest
from risk.limits import MaxOrderSizeLimit


@pytest.mark.asyncio
async def test_order_manager_submit_creates_submitted_order(mock_broker) -> None:
    mgr = OrderManager(mock_broker)
    req = OrderRequest(
        symbol="rb2505",
        exchange="SHFE",
        direction=Direction.LONG,
        offset=Offset.OPEN,
        price=Decimal("3400"),
        volume=2,
        strategy_id="s-1",
    )
    order = await mgr.submit(req)
    assert order.status == OrderStatus.SUBMITTED
    assert order.symbol == "rb2505"
    assert order.volume == 2
    assert order.strategy_id == "s-1"
    assert order.order_id.startswith("mock-")


@pytest.mark.asyncio
async def test_order_manager_rejected_order_has_rejected_status_and_no_broker_call(
    mock_broker,
) -> None:
    mgr = OrderManager(mock_broker)

    def deny(_req: OrderRequest) -> tuple[bool, str]:
        return False, "blocked"

    mgr.set_pre_trade_check(deny)
    req = OrderRequest(
        symbol="rb2505",
        exchange="SHFE",
        direction=Direction.LONG,
        offset=Offset.OPEN,
        price=Decimal("3400"),
        volume=1,
    )
    order = await mgr.submit(req)
    assert order.status == OrderStatus.REJECTED
    assert mock_broker.submit_order_calls == 0


@pytest.mark.asyncio
async def test_order_manager_rejection_uses_risk_engine_style_reason(mock_broker) -> None:
    from risk.engine import RiskEngine

    mgr = OrderManager(mock_broker)
    risk = RiskEngine()
    risk.add_limit(MaxOrderSizeLimit(max_volume=5))
    mgr.set_pre_trade_check(risk.pre_trade_check)
    req = OrderRequest(
        symbol="rb2505",
        exchange="SHFE",
        direction=Direction.LONG,
        offset=Offset.OPEN,
        price=Decimal("3400"),
        volume=10,
    )
    order = await mgr.submit(req)
    assert order.status == OrderStatus.REJECTED
    assert mock_broker.submit_order_calls == 0


@pytest.mark.asyncio
async def test_execution_engine_place_order_delegates_without_start(mock_broker) -> None:
    engine = ExecutionEngine(mock_broker)
    req = OrderRequest(
        symbol="cu2506",
        exchange="SHFE",
        direction=Direction.SHORT,
        offset=Offset.OPEN,
        price=Decimal("72000"),
        volume=1,
    )
    order = await engine.place_order(req)
    assert order.status == OrderStatus.SUBMITTED
    assert order.symbol == "cu2506"


@pytest.mark.asyncio
async def test_execution_engine_stop_disconnects_broker(mock_broker) -> None:
    engine = ExecutionEngine(mock_broker)
    await engine.stop()
    assert mock_broker._connected is False


@pytest.mark.asyncio
async def test_execution_engine_start_stop_lifecycle_skips_reconcile_sleep(mock_broker) -> None:
    """Avoid real 10s sleeps in _reconcile_loop while still exercising connect/disconnect."""

    async def noop_reconcile(self: ExecutionEngine) -> None:
        while self._running:
            break

    engine = ExecutionEngine(mock_broker)
    with patch.object(ExecutionEngine, "_reconcile_loop", noop_reconcile):
        await engine.start()
    assert mock_broker._connected is True
    await engine.stop()
    assert mock_broker._connected is False


@pytest.mark.asyncio
async def test_order_manager_cancel_returns_false_for_unknown_order(mock_broker) -> None:
    mgr = OrderManager(mock_broker)
    assert await mgr.cancel("nonexistent") is False


@pytest.mark.asyncio
async def test_order_manager_cancel_invokes_broker_when_order_active(mock_broker) -> None:
    mgr = OrderManager(mock_broker)
    order = await mgr.submit(
        OrderRequest(
            symbol="rb2505",
            exchange="SHFE",
            direction=Direction.LONG,
            offset=Offset.OPEN,
            price=Decimal("1"),
            volume=1,
        )
    )
    mock_broker.cancel_order = AsyncMock(return_value=True)
    ok = await mgr.cancel(order.order_id)
    assert ok is True
    mock_broker.cancel_order.assert_awaited_once_with(order.order_id)
