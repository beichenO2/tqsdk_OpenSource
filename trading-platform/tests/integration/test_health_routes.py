"""Integration tests for ``/healthz`` and ``/readyz``."""

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

from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

from tests.integration.route_harness import build_test_app


@pytest.fixture
def exec_service_state():
    import app.deps as deps

    prev = deps._execution_service
    yield deps
    deps.set_execution_service(prev)


@pytest.fixture
def health_client() -> TestClient:
    app = build_test_app(routers=("health",))
    return TestClient(app, raise_server_exceptions=False)


def test_healthz_returns_200(health_client: TestClient) -> None:
    r = health_client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_readyz_returns_503_when_not_ready(
    health_client: TestClient, exec_service_state: object
) -> None:
    deps = exec_service_state
    deps.set_execution_service(None)
    r = health_client.get("/readyz")
    assert r.status_code == 503
    body = r.json()
    assert body["error"] == "SERVICE_NOT_READY"
    assert "ExecutionService" in body["message"]


def test_readyz_503_includes_detail_envelope(health_client: TestClient, exec_service_state: object) -> None:
    deps = exec_service_state
    deps.set_execution_service(None)
    r = health_client.get("/readyz")
    assert r.status_code == 503
    detail = r.json().get("detail") or {}
    assert detail.get("execution") == "unavailable"


def test_readyz_returns_200_when_ready(
    health_client: TestClient, exec_service_state: object
) -> None:
    deps = exec_service_state
    deps.set_execution_service(MagicMock(name="ExecutionService"))
    r = health_client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_readyz_includes_execution_field_when_ready(
    health_client: TestClient, exec_service_state: object
) -> None:
    deps = exec_service_state
    deps.set_execution_service(MagicMock())
    r = health_client.get("/readyz")
    assert r.status_code == 200
    assert r.json().get("execution") == "ok"


def test_healthz_content_type_json(health_client: TestClient) -> None:
    r = health_client.get("/healthz")
    assert "application/json" in r.headers.get("content-type", "")
