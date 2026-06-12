"""Shared pytest fixtures — async, DB session mocks, API TestClient."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from decimal import Decimal
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.testclient import TestClient

from core.enums.direction import Direction, Offset
from core.enums.market import Exchange
from core.enums.order_status import OrderStatus
from core.models.order import Order
from core.models.position import Position

from execution.broker_adapter import BrokerAdapter


class MockBrokerAdapter(BrokerAdapter):
    """In-memory broker for tests — no TqSdk or network."""

    def __init__(self) -> None:
        self._connected = False
        self._order_seq = 0
        self.submit_order_calls = 0

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
        self.submit_order_calls += 1
        self._order_seq += 1
        return Order(
            order_id=f"mock-{self._order_seq}",
            strategy_id=strategy_id or "test-strategy",
            symbol=symbol,
            exchange=Exchange.SHFE,
            direction=direction,
            offset=offset,
            price=price,
            volume=volume,
            status=OrderStatus.SUBMITTED,
        )

    async def cancel_order(self, order_id: str) -> bool:
        return True

    async def query_order(self, order_id: str) -> Optional[Order]:
        return None

    async def query_positions(self) -> list[Position]:
        return []

    async def get_account_info(self) -> dict[str, Any]:
        return {"balance": "0", "available": "0"}


@pytest.fixture
def mock_broker() -> MockBrokerAdapter:
    return MockBrokerAdapter()


@pytest_asyncio.fixture
async def mock_async_session() -> AsyncMock:
    """Async SQLAlchemy session double — no PostgreSQL."""
    session = AsyncMock(spec=AsyncSession)
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    session.refresh = AsyncMock()
    return session


@pytest_asyncio.fixture
async def mock_db_session_factory(
    mock_async_session: AsyncMock,
) -> Callable[[], AsyncIterator[AsyncMock]]:
    """Factory matching `async def get_session(): yield session` DI pattern."""

    async def _factory() -> AsyncIterator[AsyncMock]:
        yield mock_async_session

    return _factory


def build_test_app(**kwargs: Any) -> FastAPI:
    """FastAPI test app factory (no lifespan) — delegates to ``route_harness.build_test_app``."""
    from tests.integration.route_harness import build_test_app as _build_test_app

    return _build_test_app(**kwargs)


def build_minimal_api_app(
    *,
    execution_service: Any | None = None,
) -> FastAPI:
    """FastAPI app with health + orders routers, no lifespan (no TqSdk startup)."""
    from app.deps import get_execution_service
    from app.routers import health, orders
    from tests.integration.route_harness import register_platform_exception_handlers

    app = FastAPI()
    register_platform_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(orders.router, prefix="/api/v1")
    if execution_service is not None:
        app.dependency_overrides[get_execution_service] = lambda: execution_service
    return app


@pytest.fixture
def test_client_no_deps() -> TestClient:
    """Client for routes that do not need ExecutionService (e.g. /healthz)."""
    app = build_minimal_api_app()
    return TestClient(app)


@pytest.fixture
def mock_execution_service() -> MagicMock:
    """Default mock for order API tests — override methods per test as needed."""
    svc = MagicMock()
    svc.place_order = AsyncMock()
    svc.cancel_order = AsyncMock(return_value=True)
    svc.get_order = MagicMock(return_value=None)
    svc.get_all_orders = MagicMock(return_value=[])
    svc.get_active_orders = MagicMock(return_value=[])
    return svc


@pytest.fixture
def test_client_with_mock_exec(mock_execution_service: MagicMock) -> TestClient:
    app = build_minimal_api_app(execution_service=mock_execution_service)
    return TestClient(app)
