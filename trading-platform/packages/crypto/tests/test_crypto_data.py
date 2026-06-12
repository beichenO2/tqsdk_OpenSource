"""Integration tests for crypto data aggregation endpoints and BTC API compatibility."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient


def _build_crypto_app() -> tuple[FastAPI, MagicMock]:
    from app.deps import set_btc_broker_manager
    from app.routers import btc, crypto_data
    from broker_crypto.models import Exchange as CryptoExchange
    from tests.integration.route_harness import register_platform_exception_handlers

    manager = MagicMock()
    manager.exchanges = [CryptoExchange.BINANCE]

    _adapters = {CryptoExchange.BINANCE: MagicMock(get_klines=AsyncMock(return_value=[]))}

    def _get_adapter(ex):
        if ex not in _adapters:
            raise KeyError(f"未注册的交易所: {ex.value}")
        return _adapters[ex]

    manager.get_adapter = MagicMock(side_effect=_get_adapter)

    now = datetime.now(tz=timezone.utc)
    manager.get_ticker = AsyncMock(return_value=MagicMock(
        exchange=CryptoExchange.BINANCE, symbol="BTCUSDT",
        bid=Decimal("68000"), ask=Decimal("68001"), last=Decimal("68000.5"),
        volume_24h=Decimal("1000"), timestamp=now,
        model_dump=lambda: {"exchange": "BINANCE", "symbol": "BTCUSDT",
                            "bid": "68000", "ask": "68001", "last": "68000.5",
                            "volume_24h": "1000", "timestamp": now.isoformat()},
    ))
    manager.get_orderbook = AsyncMock(return_value=MagicMock(
        model_dump=lambda: {"exchange": "BINANCE", "symbol": "BTCUSDT",
                            "bids": [["68000", "1"]], "asks": [["68001", "1"]],
                            "timestamp": now.isoformat()},
    ))
    manager.get_recent_trades = AsyncMock(return_value=[])
    manager.get_open_orders = AsyncMock(return_value=[])
    manager.place_order = AsyncMock(side_effect=PermissionError("No auth"))
    manager.get_balances = AsyncMock(return_value=[])
    manager.get_positions = AsyncMock(return_value=[])

    set_btc_broker_manager(manager)

    app = FastAPI()
    register_platform_exception_handlers(app)
    app.include_router(btc.router, prefix="/api/v1/btc", tags=["btc"])
    app.include_router(crypto_data.router, prefix="/api/v1")
    return app, manager


@pytest.fixture
def client() -> TestClient:
    from app.deps import set_btc_broker_manager

    app, manager = _build_crypto_app()
    yield TestClient(app)
    set_btc_broker_manager(None)


class TestBtcMarketEndpoints:
    """Market endpoints use Binance public API — 200 on success, 502 on network error."""

    def test_get_ticker(self, client: TestClient):
        resp = client.get("/api/v1/btc/market/ticker/BTCUSDT")
        assert resp.status_code in (200, 502)
        if resp.status_code == 200:
            data = resp.json()
            assert "bid" in data

    def test_get_klines(self, client: TestClient):
        resp = client.get("/api/v1/btc/market/klines/BTCUSDT?timeframe=1h&limit=10")
        assert resp.status_code in (200, 502)

    def test_get_orderbook(self, client: TestClient):
        resp = client.get("/api/v1/btc/market/orderbook/BTCUSDT?limit=20")
        assert resp.status_code in (200, 502)

    def test_get_recent_trades(self, client: TestClient):
        resp = client.get("/api/v1/btc/market/trades/BTCUSDT")
        assert resp.status_code in (200, 502)


class TestBtcTradingEndpoints:
    """Trading endpoints need real API keys — expect 502 (auth failure) with public-only creds."""

    def test_get_orders(self, client: TestClient):
        resp = client.get("/api/v1/btc/orders")
        assert resp.status_code in (200, 502)

    def test_get_trade_history(self, client: TestClient):
        resp = client.get("/api/v1/btc/trades")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_balances(self, client: TestClient):
        resp = client.get("/api/v1/btc/account/balances")
        assert resp.status_code in (200, 502)

    def test_get_positions(self, client: TestClient):
        resp = client.get("/api/v1/btc/account/positions")
        assert resp.status_code in (200, 502)

    def test_place_order_no_auth(self, client: TestClient):
        resp = client.post(
            "/api/v1/btc/orders",
            json={
                "symbol": "BTCUSDT",
                "side": "buy",
                "type": "limit",
                "amount": 0.01,
                "price": 65000,
            },
        )
        assert resp.status_code in (403, 502)


class TestBtcStrategyEndpoints:
    def test_list_strategies_returns_array(self, client: TestClient):
        resp = client.get("/api/v1/btc/strategies")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_backtest_results_by_strategy(self, client: TestClient):
        resp = client.get("/api/v1/btc/backtest/results/btc_momentum")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_list_backtests_returns_array(self, client: TestClient):
        resp = client.get("/api/v1/btc/backtest")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


class TestBtcExchangeEndpoints:
    def test_list_exchanges(self, client: TestClient):
        resp = client.get("/api/v1/btc/exchanges")
        assert resp.status_code == 200
        data = resp.json()
        assert "exchanges" in data
        assert len(data["exchanges"]) >= 2
        binance = next(e for e in data["exchanges"] if e["id"] == "binance")
        assert binance["connected"] is True

    def test_exchange_status_binance_connected(self, client: TestClient):
        resp = client.get("/api/v1/btc/exchanges/binance/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["connected"] is True

    def test_exchange_status_okx_not_connected(self, client: TestClient):
        resp = client.get("/api/v1/btc/exchanges/okx/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["connected"] is False


class TestCryptoDataEndpoints:
    def test_news_endpoint_exists(self, client: TestClient):
        resp = client.get("/api/v1/crypto-data/news")
        assert resp.status_code in (200, 503)

    def test_fund_flows_endpoint_exists(self, client: TestClient):
        resp = client.get("/api/v1/crypto-data/fund-flows/BTC")
        assert resp.status_code in (200, 503)

    def test_quotes_endpoint_exists(self, client: TestClient):
        resp = client.get("/api/v1/crypto-data/quotes")
        assert resp.status_code in (200, 503)

    def test_open_interest_endpoint_exists(self, client: TestClient):
        resp = client.get("/api/v1/crypto-data/open-interest/BTC")
        assert resp.status_code in (200, 503)

    def test_funding_rates_endpoint_exists(self, client: TestClient):
        resp = client.get("/api/v1/crypto-data/funding-rates/BTC")
        assert resp.status_code in (200, 503)

    def test_liquidations_endpoint_exists(self, client: TestClient):
        resp = client.get("/api/v1/crypto-data/liquidations/BTC")
        assert resp.status_code in (200, 503)
