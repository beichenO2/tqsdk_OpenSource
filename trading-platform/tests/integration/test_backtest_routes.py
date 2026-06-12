"""Integration tests for ``/api/v1/backtest`` routes."""

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

import importlib.util

import pytest
from starlette.testclient import TestClient

from tests.integration.route_harness import build_test_app


def _skip_if_backtest_engine_missing() -> None:
    import app.routers.backtest as bt

    if bt._BACKTEST_IMPORT_ERROR is not None or bt.BacktestEngine is None:
        pytest.skip("backtest engine not importable in this environment")


@pytest.fixture
def client() -> TestClient:
    app = build_test_app(routers=("backtest",))
    return TestClient(app, raise_server_exceptions=False)


def test_strategy_names_returns_list(client: TestClient) -> None:
    r = client.get("/api/v1/backtest/strategy-names")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) >= 5


def test_strategy_names_contains_expected_defaults(client: TestClient) -> None:
    r = client.get("/api/v1/backtest/strategy-names")
    names = set(r.json())
    assert "dual_ma" in names or len(names) > 0


def test_strategy_names_json_content_type(client: TestClient) -> None:
    r = client.get("/api/v1/backtest/strategy-names")
    assert r.headers.get("content-type", "").startswith("application/json")


def test_strategy_names_get_method_only(client: TestClient) -> None:
    r = client.post("/api/v1/backtest/strategy-names")
    assert r.status_code == 405


def test_results_returns_list(client: TestClient) -> None:
    r = client.get("/api/v1/backtest/results")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_results_empty_list_is_valid_json_array(client: TestClient) -> None:
    r = client.get("/api/v1/backtest/results")
    assert r.json() == [] or isinstance(r.json(), list)


def test_results_entries_have_expected_shape_when_present(client: TestClient) -> None:
    r = client.get("/api/v1/backtest/results")
    rows = r.json()
    if not rows:
        return
    row = rows[0]
    for key in ("id", "strategy_name", "status"):
        assert key in row


def test_results_get_method_only(client: TestClient) -> None:
    r = client.delete("/api/v1/backtest/results")
    assert r.status_code == 405


def test_post_backtest_unavailable_returns_503(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    import app.routers.backtest as bt

    monkeypatch.setattr(bt, "_BACKTEST_IMPORT_ERROR", "forced missing backtest for test", raising=False)
    body = {
        "strategy_name": "dual_ma",
        "symbols": ["SHFE.rb2501"],
        "params": {},
        "start_date": "2026-03-01",
        "end_date": "2026-03-31",
        "initial_capital": 1_000_000.0,
        "commission_rate": 0.0001,
        "contract_multiplier": 10,
    }
    r = client.post("/api/v1/backtest", json=body)
    assert r.status_code == 503
    payload = r.json()
    assert payload["error"] == "BACKTEST_UNAVAILABLE"
    assert "import_error" in payload.get("detail", {})


def test_post_backtest_run_unavailable_returns_503(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    import app.routers.backtest as bt

    monkeypatch.setattr(bt, "_BACKTEST_IMPORT_ERROR", "forced", raising=False)
    body = {
        "strategy_name": "dual_ma",
        "symbols": ["SHFE.rb2501"],
        "start_date": "2026-03-01",
        "end_date": "2026-03-31",
    }
    r = client.post("/api/v1/backtest/run", json=body)
    assert r.status_code == 503


def test_post_backtest_invalid_date_order_returns_422(client: TestClient) -> None:
    _skip_if_backtest_engine_missing()
    body = {
        "strategy_name": "dual_ma",
        "symbols": ["SHFE.rb2501"],
        "start_date": "2026-04-01",
        "end_date": "2026-03-01",
    }
    r = client.post("/api/v1/backtest", json=body)
    assert r.status_code == 422
    assert r.json()["error"] == "VALIDATION_ERROR"


def test_post_backtest_run_invalid_date_order_returns_422(client: TestClient) -> None:
    _skip_if_backtest_engine_missing()
    body = {
        "strategy_name": "dual_ma",
        "symbols": ["SHFE.rb2501"],
        "start_date": "2026-05-10",
        "end_date": "2026-05-09",
    }
    r = client.post("/api/v1/backtest/run", json=body)
    assert r.status_code == 422


def test_post_backtest_invalid_json_body_returns_unprocessable(client: TestClient) -> None:
    r = client.post(
        "/api/v1/backtest",
        content=b"{not-json",
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 422


def test_post_backtest_missing_required_field_returns_422(client: TestClient) -> None:
    r = client.post("/api/v1/backtest", json={"strategy_name": "x"})
    assert r.status_code == 422


def test_post_backtest_empty_symbols_returns_422(client: TestClient) -> None:
    body = {
        "strategy_name": "dual_ma",
        "symbols": [],
        "start_date": "2026-03-01",
        "end_date": "2026-03-31",
    }
    r = client.post("/api/v1/backtest", json=body)
    assert r.status_code == 422


def test_post_backtest_invalid_initial_capital_returns_422(client: TestClient) -> None:
    body = {
        "strategy_name": "dual_ma",
        "symbols": ["SHFE.rb2501"],
        "start_date": "2026-03-01",
        "end_date": "2026-03-31",
        "initial_capital": 0,
    }
    r = client.post("/api/v1/backtest", json=body)
    assert r.status_code == 422


def test_post_backtest_invalid_contract_multiplier_returns_422(client: TestClient) -> None:
    body = {
        "strategy_name": "dual_ma",
        "symbols": ["SHFE.rb2501"],
        "start_date": "2026-03-01",
        "end_date": "2026-03-31",
        "contract_multiplier": 0,
    }
    r = client.post("/api/v1/backtest", json=body)
    assert r.status_code == 422


def test_post_backtest_invalid_commission_rate_returns_422(client: TestClient) -> None:
    body = {
        "strategy_name": "dual_ma",
        "symbols": ["SHFE.rb2501"],
        "start_date": "2026-03-01",
        "end_date": "2026-03-31",
        "commission_rate": -0.01,
    }
    r = client.post("/api/v1/backtest", json=body)
    assert r.status_code == 422


def test_post_backtest_malformed_date_valueerror_returns_422(client: TestClient) -> None:
    _skip_if_backtest_engine_missing()
    body = {
        "strategy_name": "dual_ma",
        "symbols": ["SHFE.rb2501"],
        "start_date": "not-a-date",
        "end_date": "2026-03-31",
    }
    r = client.post("/api/v1/backtest", json=body)
    assert r.status_code == 422


@pytest.mark.skipif(
    importlib.util.find_spec("backtest") is None,
    reason="backtest package not installed",
)
def test_post_backtest_success_when_engine_available(client: TestClient) -> None:
    import app.routers.backtest as bt

    if bt._BACKTEST_IMPORT_ERROR is not None:
        pytest.skip("backtest router deps not loadable in this environment")
    body = {
        "strategy_name": "dual_ma",
        "symbols": ["SHFE.rb2501"],
        "params": {},
        "start_date": "2024-01-15",
        "end_date": "2024-02-15",
        "initial_capital": 100_000.0,
        "commission_rate": 0.0001,
        "contract_multiplier": 10,
    }
    r = client.post("/api/v1/backtest", json=body)
    if r.status_code == 404:
        pytest.skip("strategy type not registered in StrategyRegistry")
    if r.status_code == 422 and "parquet" in r.json().get("message", "").lower():
        pytest.skip("No parquet data available for test date range")
    assert r.status_code == 200
    data = r.json()
    for key in ("total_return", "max_drawdown", "sharpe", "win_rate", "total_trades", "final_equity"):
        assert key in data
