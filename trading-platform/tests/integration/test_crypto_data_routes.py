"""Integration tests for ``/api/v1/crypto-data`` when providers are unavailable."""

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
from starlette.testclient import TestClient

from tests.integration.route_harness import build_test_app


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    import app.routers.crypto_data as cd

    monkeypatch.setattr(cd, "_IMPORT_ERROR", "forced unavailable providers", raising=False)
    app = build_test_app(routers=("crypto_data",))
    return TestClient(app, raise_server_exceptions=False)


def _assert_503_providers(r) -> None:
    assert r.status_code == 503
    body = r.json()
    assert body["error"] == "PROVIDERS_UNAVAILABLE"
    assert "import_error" in body.get("detail", {})


def test_news_returns_503_when_unavailable(client: TestClient) -> None:
    _assert_503_providers(client.get("/api/v1/crypto-data/news"))


def test_fund_flows_returns_503_when_unavailable(client: TestClient) -> None:
    _assert_503_providers(client.get("/api/v1/crypto-data/fund-flows/BTC"))


def test_macro_returns_503_when_unavailable(client: TestClient) -> None:
    _assert_503_providers(client.get("/api/v1/crypto-data/macro"))


def test_quotes_returns_503_when_unavailable(client: TestClient) -> None:
    _assert_503_providers(client.get("/api/v1/crypto-data/quotes"))


def test_global_metrics_returns_503_when_unavailable(client: TestClient) -> None:
    _assert_503_providers(client.get("/api/v1/crypto-data/global-metrics"))


def test_open_interest_returns_503_when_unavailable(client: TestClient) -> None:
    _assert_503_providers(client.get("/api/v1/crypto-data/open-interest/BTC"))


def test_funding_rates_returns_503_when_unavailable(client: TestClient) -> None:
    _assert_503_providers(client.get("/api/v1/crypto-data/funding-rates/BTC"))


def test_liquidations_returns_503_when_unavailable(client: TestClient) -> None:
    _assert_503_providers(client.get("/api/v1/crypto-data/liquidations/BTC"))


def test_long_short_ratio_returns_503_when_unavailable(client: TestClient) -> None:
    _assert_503_providers(client.get("/api/v1/crypto-data/long-short-ratio/BTC"))


def test_onchain_returns_503_when_unavailable(client: TestClient) -> None:
    _assert_503_providers(client.get("/api/v1/crypto-data/onchain/1"))


def test_onchain_cached_returns_503_when_unavailable(client: TestClient) -> None:
    _assert_503_providers(client.get("/api/v1/crypto-data/onchain/42/cached"))
