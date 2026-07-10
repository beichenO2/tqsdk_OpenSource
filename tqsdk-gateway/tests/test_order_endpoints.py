"""Gateway order query endpoints — list, single lookup, 404."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from main import app
from session import TqSdkSession


def _mock_order(**overrides):
    o = MagicMock()
    o.order_id = overrides.get("order_id", "order-1")
    o.instrument_id = overrides.get("instrument_id", "rb2510")
    o.exchange_id = overrides.get("exchange_id", "SHFE")
    o.direction = overrides.get("direction", "BUY")
    o.offset = overrides.get("offset", "OPEN")
    o.status = overrides.get("status", "ALIVE")
    o.volume_orign = overrides.get("volume_orign", 2)
    o.volume_left = overrides.get("volume_left", 2)
    o.limit_price = overrides.get("limit_price", 3500.0)
    o.trade_price = overrides.get("trade_price", 0.0)
    o.last_msg = overrides.get("last_msg", "")
    o.is_error = overrides.get("is_error", False)
    return o


def _connected_session(mock_api: MagicMock) -> TqSdkSession:
    session = TqSdkSession()
    session._api = mock_api
    session._connected = True
    return session


def test_list_orders_returns_items() -> None:
    mock_api = MagicMock()
    order = _mock_order(order_id="ord-100", volume_orign=3, volume_left=1, trade_price=3510.0)
    mock_api.get_order.return_value = {"ord-100": order}

    session = _connected_session(mock_api)
    with patch("main.get_session", return_value=session):
        client = TestClient(app)
        resp = client.get("/api/v1/orders")

    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert len(data["items"]) == 1
    item = data["items"][0]
    assert item["order_id"] == "ord-100"
    assert item["status"] == "ALIVE"
    assert item["volume"] == 3
    assert item["filled_volume"] == 2
    assert item["avg_price"] == 3510.0


def test_get_order_returns_single_order() -> None:
    mock_api = MagicMock()
    order = _mock_order(
        order_id="ord-200",
        status="FINISHED",
        volume_orign=2,
        volume_left=0,
        trade_price=3499.5,
    )
    mock_api.get_order.return_value = order

    session = _connected_session(mock_api)
    with patch("main.get_session", return_value=session):
        client = TestClient(app)
        resp = client.get("/api/v1/orders/ord-200")

    assert resp.status_code == 200
    item = resp.json()
    assert item["order_id"] == "ord-200"
    assert item["status"] == "FINISHED"
    assert item["filled_volume"] == 2
    assert item["volume_left"] == 0
    assert item["avg_price"] == 3499.5


def test_get_order_returns_404_when_missing() -> None:
    mock_api = MagicMock()
    mock_api.get_order.return_value = None

    session = _connected_session(mock_api)
    with patch("main.get_session", return_value=session):
        client = TestClient(app)
        resp = client.get("/api/v1/orders/missing-id")

    assert resp.status_code == 404


def test_list_orders_uses_locked_session() -> None:
    """Orders endpoints must acquire the timed lock like account/positions."""
    mock_api = MagicMock()
    mock_api.get_order.return_value = {}
    session = _connected_session(mock_api)

    lock_entered = []

    class TrackingLock:
        def __enter__(self):
            lock_entered.append(True)
            return session._lock.__enter__()

        def __exit__(self, *exc):
            return session._lock.__exit__(*exc)

    session._locked = lambda: TrackingLock()  # type: ignore[method-assign]

    with patch("main.get_session", return_value=session):
        client = TestClient(app)
        resp = client.get("/api/v1/orders")

    assert resp.status_code == 200
    assert lock_entered, "expected _locked() to be used for list orders"
