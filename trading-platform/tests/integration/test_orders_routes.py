"""Route integration tests for ``/api/v1/orders`` (error envelopes + CRUD)."""

from __future__ import annotations

import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
for p in (
    _repo,
    _repo / "apps" / "api",
    _repo / "packages" / "core",
    _repo / "packages" / "backtest",
    _repo / "packages" / "security" / "src",
    _repo / "packages",
):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.testclient import TestClient

from core.enums.direction import Direction, Offset
from core.enums.market import Exchange
from core.enums.order_status import OrderStatus
from core.models.order import Order

from tests.integration.route_harness import build_test_app


@pytest.fixture
def mock_execution_service() -> MagicMock:
    svc = MagicMock()
    svc.place_order = AsyncMock()
    svc.cancel_order = AsyncMock(return_value=True)
    svc.get_order = MagicMock(return_value=None)
    svc.get_all_orders = MagicMock(return_value=[])
    svc.get_active_orders = MagicMock(return_value=[])
    return svc


@pytest.fixture
def client(mock_execution_service: MagicMock) -> TestClient:
    app = build_test_app(routers=("orders",), execution_service=mock_execution_service)
    return TestClient(app, raise_server_exceptions=False)


def test_post_order_creates_order(client: TestClient, mock_execution_service: MagicMock) -> None:
    mock_execution_service.place_order.return_value = Order(
        order_id="ord-int-1",
        strategy_id="st-1",
        symbol="rb2505",
        exchange=Exchange.SHFE,
        direction=Direction.LONG,
        offset=Offset.OPEN,
        price=Decimal("3600"),
        volume=2,
        status=OrderStatus.SUBMITTED,
    )
    body = {
        "strategy_id": "st-1",
        "symbol": "rb2505",
        "exchange": "SHFE",
        "direction": "LONG",
        "offset": "OPEN",
        "price": "3600",
        "volume": 2,
    }
    r = client.post("/api/v1/orders", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["order_id"] == "ord-int-1"
    assert data["status"] == "SUBMITTED"


def test_get_order_returns_404_envelope_when_missing(
    client: TestClient, mock_execution_service: MagicMock
) -> None:
    mock_execution_service.get_order.return_value = None
    r = client.get("/api/v1/orders/missing-order")
    assert r.status_code == 404
    body = r.json()
    assert body["error"] == "ORDER_NOT_FOUND"
    assert "missing-order" in body["message"]


def test_delete_order_returns_400_when_cancel_fails(
    client: TestClient, mock_execution_service: MagicMock
) -> None:
    mock_execution_service.cancel_order.return_value = False
    r = client.delete("/api/v1/orders/ord-x")
    assert r.status_code == 400
    body = r.json()
    assert body["error"] == "ORDER_CANCEL_FAILED"


def test_get_orders_returns_list(client: TestClient, mock_execution_service: MagicMock) -> None:
    mock_execution_service.get_all_orders.return_value = [
        Order(
            order_id="a1",
            strategy_id="s1",
            symbol="rb2505",
            exchange=Exchange.SHFE,
            direction=Direction.LONG,
            offset=Offset.OPEN,
            price=Decimal("1"),
            volume=1,
            status=OrderStatus.SUBMITTED,
        )
    ]
    r = client.get("/api/v1/orders")
    assert r.status_code == 200
    rows = r.json()
    assert isinstance(rows, list)
    assert rows[0]["order_id"] == "a1"


def test_delete_order_success_payload(client: TestClient, mock_execution_service: MagicMock) -> None:
    mock_execution_service.cancel_order.return_value = True
    r = client.delete("/api/v1/orders/ord-ok")
    assert r.status_code == 200
    assert r.json() == {"order_id": "ord-ok", "status": "CANCELLED"}


def test_list_orders_active_only_filters(
    client: TestClient, mock_execution_service: MagicMock
) -> None:
    o1 = Order(
        order_id="o1",
        strategy_id="s1",
        symbol="rb2505",
        exchange=Exchange.SHFE,
        direction=Direction.LONG,
        offset=Offset.OPEN,
        price=Decimal("1"),
        volume=1,
        status=OrderStatus.SUBMITTED,
    )
    mock_execution_service.get_active_orders.return_value = [o1]
    r = client.get("/api/v1/orders?active_only=true")
    assert r.status_code == 200
    assert len(r.json()) == 1


def test_list_orders_filters_by_strategy_id(
    client: TestClient, mock_execution_service: MagicMock
) -> None:
    mock_execution_service.get_all_orders.return_value = [
        Order(
            order_id="x",
            strategy_id="keep",
            symbol="rb2505",
            exchange=Exchange.SHFE,
            direction=Direction.LONG,
            offset=Offset.OPEN,
            price=Decimal("1"),
            volume=1,
            status=OrderStatus.SUBMITTED,
        ),
        Order(
            order_id="y",
            strategy_id="drop",
            symbol="rb2505",
            exchange=Exchange.SHFE,
            direction=Direction.LONG,
            offset=Offset.OPEN,
            price=Decimal("2"),
            volume=1,
            status=OrderStatus.SUBMITTED,
        ),
    ]
    r = client.get("/api/v1/orders?strategy_id=keep")
    assert r.status_code == 200
    assert [row["order_id"] for row in r.json()] == ["x"]


def test_create_order_validation_error_on_non_positive_volume(
    client: TestClient, mock_execution_service: MagicMock
) -> None:
    body = {
        "strategy_id": "st",
        "symbol": "rb2505",
        "exchange": "SHFE",
        "direction": "LONG",
        "offset": "OPEN",
        "price": "1",
        "volume": 0,
    }
    r = client.post("/api/v1/orders", json=body)
    assert r.status_code == 422


def test_get_order_returns_payload_when_found(
    client: TestClient, mock_execution_service: MagicMock
) -> None:
    mock_execution_service.get_order.return_value = Order(
        order_id="found-1",
        strategy_id="s",
        symbol="rb2505",
        exchange=Exchange.SHFE,
        direction=Direction.LONG,
        offset=Offset.OPEN,
        price=Decimal("10"),
        volume=3,
        status=OrderStatus.SUBMITTED,
    )
    r = client.get("/api/v1/orders/found-1")
    assert r.status_code == 200
    assert r.json()["order_id"] == "found-1"


def test_create_order_rejected_includes_message_field(
    client: TestClient, mock_execution_service: MagicMock
) -> None:
    mock_execution_service.place_order.return_value = Order(
        order_id="rej-1",
        strategy_id="st",
        symbol="rb2505",
        exchange=Exchange.SHFE,
        direction=Direction.LONG,
        offset=Offset.OPEN,
        price=Decimal("1"),
        volume=1,
        status=OrderStatus.REJECTED,
    )
    body = {
        "strategy_id": "st",
        "symbol": "rb2505",
        "exchange": "SHFE",
        "direction": "LONG",
        "offset": "OPEN",
        "price": "1",
        "volume": 1,
    }
    r = client.post("/api/v1/orders", json=body)
    assert r.status_code == 200
    assert r.json()["message"] == "Risk rejected"


def test_list_orders_empty_returns_empty_array(
    client: TestClient, mock_execution_service: MagicMock
) -> None:
    mock_execution_service.get_all_orders.return_value = []
    r = client.get("/api/v1/orders")
    assert r.status_code == 200
    assert r.json() == []
