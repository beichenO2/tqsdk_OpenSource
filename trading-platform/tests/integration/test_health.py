"""HTTP integration tests for health endpoints."""

from __future__ import annotations

from starlette.testclient import TestClient


def test_healthz_returns_200_and_ok_status(test_client_no_deps: TestClient) -> None:
    response = test_client_no_deps.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_healthz_content_type_json(test_client_no_deps: TestClient) -> None:
    response = test_client_no_deps.get("/healthz")
    assert "application/json" in response.headers.get("content-type", "")


def test_minimal_app_exposes_healthz_route(test_client_no_deps: TestClient) -> None:
    routes = {r.path for r in test_client_no_deps.app.routes}
    assert "/healthz" in routes
