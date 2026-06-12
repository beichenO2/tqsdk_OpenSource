"""Integration tests for /api/v1/orders with mocked ExecutionService."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

from starlette.testclient import TestClient

from core.enums.direction import Direction, Offset
from core.enums.market import Exchange
from core.enums.order_status import OrderStatus
from core.models.order import Order


def test_create_order_returns_200_and_payload(
    test_client_with_mock_exec: TestClient,
    mock_execution_service: MagicMock,
) -> None:
    mock_execution_service.place_order.return_value = Order(
        order_id="ord-abc",
        strategy_id="st-1",
        symbol="rb2505",
        exchange=Exchange.SHFE,
        direction=Direction.LONG,
        offset=Offset.OPEN,
        price=Decimal("3500"),
        volume=3,
        status=OrderStatus.SUBMITTED,
    )
    body = {
        "strategy_id": "st-1",
        "symbol": "rb2505",
        "exchange": "SHFE",
        "direction": "LONG",
        "offset": "OPEN",
        "price": "3500",
        "volume": 3,
    }
    response = test_client_with_mock_exec.post("/api/v1/orders", json=body)
    assert response.status_code == 200
    data = response.json()
    assert data["order_id"] == "ord-abc"
    assert data["status"] == "SUBMITTED"
    assert data["symbol"] == "rb2505"
    assert data["volume"] == 3
    assert data["message"] == ""


def test_create_order_risk_rejected_sets_message(
    test_client_with_mock_exec: TestClient,
    mock_execution_service: MagicMock,
) -> None:
    mock_execution_service.place_order.return_value = Order(
        order_id="rej-1",
        strategy_id="st-1",
        symbol="rb2505",
        exchange=Exchange.SHFE,
        direction=Direction.LONG,
        offset=Offset.OPEN,
        price=Decimal("3500"),
        volume=99,
        status=OrderStatus.REJECTED,
    )
    body = {
        "strategy_id": "st-1",
        "symbol": "rb2505",
        "exchange": "SHFE",
        "direction": "LONG",
        "offset": "OPEN",
        "price": "3500",
        "volume": 99,
    }
    response = test_client_with_mock_exec.post("/api/v1/orders", json=body)
    assert response.status_code == 200
    assert response.json()["message"] == "Risk rejected"


def test_get_order_404_when_missing(
    test_client_with_mock_exec: TestClient,
    mock_execution_service: MagicMock,
) -> None:
    mock_execution_service.get_order.return_value = None
    response = test_client_with_mock_exec.get("/api/v1/orders/unknown-id")
    assert response.status_code == 404


def test_get_order_returns_model_when_present(
    test_client_with_mock_exec: TestClient,
    mock_execution_service: MagicMock,
) -> None:
    mock_execution_service.get_order.return_value = Order(
        order_id="ord-x",
        strategy_id="st-1",
        symbol="rb2505",
        exchange=Exchange.SHFE,
        direction=Direction.LONG,
        offset=Offset.OPEN,
        price=Decimal("1"),
        volume=1,
        status=OrderStatus.SUBMITTED,
    )
    response = test_client_with_mock_exec.get("/api/v1/orders/ord-x")
    assert response.status_code == 200
    assert response.json()["order_id"] == "ord-x"


def test_list_orders_returns_json_array(
    test_client_with_mock_exec: TestClient,
    mock_execution_service: MagicMock,
) -> None:
    mock_execution_service.get_all_orders.return_value = [
        Order(
            order_id="a",
            strategy_id="s",
            symbol="rb2505",
            exchange=Exchange.SHFE,
            direction=Direction.LONG,
            offset=Offset.OPEN,
            price=Decimal("1"),
            volume=1,
            status=OrderStatus.SUBMITTED,
        )
    ]
    response = test_client_with_mock_exec.get("/api/v1/orders")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == 1


def test_cancel_order_success(
    test_client_with_mock_exec: TestClient,
    mock_execution_service: MagicMock,
) -> None:
    mock_execution_service.cancel_order.return_value = True
    response = test_client_with_mock_exec.delete("/api/v1/orders/ord-1")
    assert response.status_code == 200
    assert response.json() == {"order_id": "ord-1", "status": "CANCELLED"}
