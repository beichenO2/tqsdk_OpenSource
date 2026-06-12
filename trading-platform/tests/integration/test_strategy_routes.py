"""Integration tests for ``/api/v1/strategies`` routes."""

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

import pytest

pytest.importorskip("strategy.registry", reason="strategy package required")
from starlette.testclient import TestClient

from tests.integration.route_harness import build_test_app


@pytest.fixture
def client() -> TestClient:
    app = build_test_app(routers=("strategies",))
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _clean_strategy_instances() -> None:
    from strategy.registry import StrategyRegistry

    before = {c.strategy_id for c in StrategyRegistry.list_instances()}
    yield
    after = {c.strategy_id for c in StrategyRegistry.list_instances()}
    for sid in after - before:
        StrategyRegistry.delete_instance(sid)


def test_post_strategy_creates_and_returns_ids(client: TestClient) -> None:
    body = {
        "name": "integration-test-strat",
        "symbols": ["SHFE.rb2501"],
        "params": {"foo": 1},
        "enabled": True,
        "max_position": 5,
        "capital": 500_000.0,
    }
    r = client.post("/api/v1/strategies", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "created"
    assert "strategy_id" in data


def test_get_strategies_lists_created_strategy(client: TestClient) -> None:
    create = client.post(
        "/api/v1/strategies",
        json={"name": "list-me", "symbols": ["X"], "enabled": True},
    )
    sid = create.json()["strategy_id"]
    listed = client.get("/api/v1/strategies").json()
    ids = {row["strategy_id"] for row in listed}
    assert sid in ids


def test_get_strategy_unknown_returns_404(client: TestClient) -> None:
    r = client.get("/api/v1/strategies/does-not-exist-zzzz")
    assert r.status_code == 404
    assert r.json()["error"] == "STRATEGY_NOT_FOUND"


def test_put_toggle_unknown_returns_404(client: TestClient) -> None:
    r = client.put("/api/v1/strategies/unknown-999/toggle?enabled=false")
    assert r.status_code == 404


def test_pause_all_returns_ok_shape(client: TestClient) -> None:
    r = client.post("/api/v1/strategies/pause-all")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "paused" in body
    assert isinstance(body["paused"], int)


def test_get_strategy_by_id_returns_payload(client: TestClient) -> None:
    sid = client.post(
        "/api/v1/strategies",
        json={"name": "fetch-me", "symbols": ["A", "B"], "enabled": True},
    ).json()["strategy_id"]
    r = client.get(f"/api/v1/strategies/{sid}")
    assert r.status_code == 200
    row = r.json()
    assert row["strategy_id"] == sid
    assert row["name"] == "fetch-me"
    assert row["symbols"] == ["A", "B"]


def test_put_toggle_updates_enabled(client: TestClient) -> None:
    sid = client.post(
        "/api/v1/strategies",
        json={"name": "toggle-me", "symbols": ["S"], "enabled": True},
    ).json()["strategy_id"]
    r = client.put(f"/api/v1/strategies/{sid}/toggle?enabled=false")
    assert r.status_code == 200
    assert r.json()["enabled"] is False
    r2 = client.get(f"/api/v1/strategies/{sid}")
    assert r2.json()["enabled"] is False


def test_pause_all_pauses_enabled_strategies(client: TestClient) -> None:
    s1 = client.post(
        "/api/v1/strategies",
        json={"name": "p1", "symbols": ["S"], "enabled": True},
    ).json()["strategy_id"]
    s2 = client.post(
        "/api/v1/strategies",
        json={"name": "p2", "symbols": ["S"], "enabled": False},
    ).json()["strategy_id"]
    r = client.post("/api/v1/strategies/pause-all")
    assert r.status_code == 200
    assert r.json()["paused"] >= 1
    assert client.get(f"/api/v1/strategies/{s1}").json()["enabled"] is False
    assert client.get(f"/api/v1/strategies/{s2}").json()["enabled"] is False


def test_list_strategies_returns_json_array(client: TestClient) -> None:
    r = client.get("/api/v1/strategies")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_create_strategy_defaults_merge_params(client: TestClient) -> None:
    r = client.post(
        "/api/v1/strategies",
        json={"name": "defaults", "symbols": ["Q"], "params": {"x": 2}},
    )
    assert r.status_code == 200
    sid = r.json()["strategy_id"]
    cfg = client.get(f"/api/v1/strategies/{sid}").json()
    assert cfg["params"]["x"] == 2
    assert cfg["params"]["max_position"] == 10
    assert cfg["params"]["capital"] == 1_000_000.0


def test_toggle_with_enabled_query_defaults_true(client: TestClient) -> None:
    sid = client.post(
        "/api/v1/strategies",
        json={"name": "toggle-default", "symbols": ["S"], "enabled": False},
    ).json()["strategy_id"]
    r = client.put(f"/api/v1/strategies/{sid}/toggle")
    assert r.status_code == 200
    assert r.json()["enabled"] is True


def test_post_multiple_strategies_unique_ids(client: TestClient) -> None:
    a = client.post("/api/v1/strategies", json={"name": "m1", "symbols": ["S"]}).json()["strategy_id"]
    b = client.post("/api/v1/strategies", json={"name": "m2", "symbols": ["S"]}).json()["strategy_id"]
    assert a != b


def test_get_strategies_includes_enabled_field(client: TestClient) -> None:
    client.post("/api/v1/strategies", json={"name": "en", "symbols": ["S"], "enabled": False})
    rows = client.get("/api/v1/strategies").json()
    assert all("enabled" in row for row in rows)


def test_toggle_unknown_returns_strategy_not_found_envelope(client: TestClient) -> None:
    r = client.put("/api/v1/strategies/no-such-strategy/toggle")
    assert r.json().get("error") == "STRATEGY_NOT_FOUND"


def test_pause_all_idempotent_when_none_enabled(client: TestClient) -> None:
    s = client.post(
        "/api/v1/strategies",
        json={"name": "already-off", "symbols": ["S"], "enabled": False},
    ).json()["strategy_id"]
    r1 = client.post("/api/v1/strategies/pause-all")
    r2 = client.post("/api/v1/strategies/pause-all")
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert client.get(f"/api/v1/strategies/{s}").json()["enabled"] is False
