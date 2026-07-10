"""Unit tests for POST /research/runs/{id}/execute backtest wiring."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "packages"))
sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))


def _make_research_app() -> FastAPI:
    app = FastAPI()
    from app.routers import research

    app.include_router(research.router, prefix="/api/v1")
    return app


def _make_futures_ohlcv_df(count: int = 500, base_price: float = 3500.0) -> pd.DataFrame:
    """Synthetic OHLCV matching FuturesDataLoader bar schema."""
    rows: list[dict] = []
    price = base_price
    dt = datetime(2026, 1, 2, 9, 0, 0)
    for i in range(count):
        if i < count // 3:
            price += 5
        elif i < 2 * count // 3:
            price -= 8
        else:
            price += 6
        rows.append(
            {
                "datetime": dt,
                "open": round(price - 2, 1),
                "high": round(price + 6, 1),
                "low": round(price - 6, 1),
                "close": round(price, 1),
                "volume": 2000 + i * 10,
                "instrument": "rb2501",
                "open_interest": 10000,
                "turnover": 0.0,
            }
        )
        dt += timedelta(minutes=5)
    return pd.DataFrame(rows)


def _make_crypto_ohlcv_df(count: int = 500, base_price: float = 50000.0) -> pd.DataFrame:
    """Synthetic OHLCV matching CryptoDataLoader.load output."""
    rows: list[dict] = []
    price = base_price
    dt = datetime(2026, 1, 2, 0, 0, 0)
    for i in range(count):
        if i < count // 3:
            price += 120
        elif i < 2 * count // 3:
            price -= 200
        else:
            price += 150
        rows.append(
            {
                "open_time": dt,
                "open": round(price - 50, 2),
                "high": round(price + 200, 2),
                "low": round(price - 200, 2),
                "close": round(price, 2),
                "volume": 100.0 + i,
                "quote_volume": 1_000_000.0 + i * 100,
                "trades": 1000 + i,
                "taker_buy_volume": 50.0,
                "taker_buy_quote_volume": 500_000.0,
            }
        )
        dt += timedelta(hours=1)
    return pd.DataFrame(rows)


@pytest.fixture
def client(tmp_path, monkeypatch):
    from app.routers import research
    from experiment.research_run import RunStore

    monkeypatch.setattr(research, "RESEARCH_DIR", tmp_path)
    monkeypatch.setattr(research, "_store", RunStore(tmp_path))
    with TestClient(_make_research_app()) as tc:
        yield tc


class TestResearchExecute:
    def test_execute_futures_run(self, client, monkeypatch):
        import strategy.futures  # noqa: F401 — register futures strategies

        from datahub.futures_loader import FuturesDataLoader

        synthetic = _make_futures_ohlcv_df(500)

        def _fake_load_bars(self, instrument, timeframe="5m", *args, **kwargs):
            return synthetic.copy()

        monkeypatch.setattr(FuturesDataLoader, "load_bars", _fake_load_bars)

        create = client.post(
            "/api/v1/research/runs",
            json={
                "prompt": "dual ma rb test",
                "strategy_name": "futures_dual_ma",
                "symbols": ["rb"],
                "timeframe": "5m",
            },
        )
        assert create.status_code == 200
        run_id = create.json()["run_id"]

        exec_resp = client.post(f"/api/v1/research/runs/{run_id}/execute")
        assert exec_resp.status_code == 200
        assert exec_resp.json()["ok"] is True

        run_resp = client.get(f"/api/v1/research/runs/{run_id}")
        assert run_resp.status_code == 200
        data = run_resp.json()
        assert data["status"] == "completed"
        assert data["backtest_results"]
        assert "rb" in data["backtest_results"]
        assert "sharpe" in data["metrics"]

    def test_execute_crypto_run(self, client, monkeypatch):
        import strategy.btc  # noqa: F401 — register crypto strategies

        from datahub.crypto_loader import CryptoDataLoader

        synthetic = _make_crypto_ohlcv_df(500)

        def _fake_crypto_load(self, symbol="BTCUSDT", timeframe="1h", *args, **kwargs):
            return synthetic.copy()

        monkeypatch.setattr(CryptoDataLoader, "load", _fake_crypto_load)

        create = client.post(
            "/api/v1/research/runs",
            json={
                "prompt": "supertrend btc test",
                "strategy_name": "supertrend",
                "symbols": ["BTCUSDT"],
                "timeframe": "1h",
            },
        )
        assert create.status_code == 200
        run_id = create.json()["run_id"]

        exec_resp = client.post(f"/api/v1/research/runs/{run_id}/execute")
        assert exec_resp.status_code == 200

        run_resp = client.get(f"/api/v1/research/runs/{run_id}")
        assert run_resp.status_code == 200
        data = run_resp.json()
        assert data["status"] == "completed"
        assert data["backtest_results"]
        assert "BTCUSDT" in data["backtest_results"]
        assert "sharpe" in data["metrics"]

    def test_execute_unknown_strategy_returns_4xx(self, client):
        create = client.post(
            "/api/v1/research/runs",
            json={
                "prompt": "bad strategy",
                "strategy_name": "totally_unknown_strategy_xyz",
                "symbols": ["rb"],
            },
        )
        assert create.status_code == 200
        run_id = create.json()["run_id"]

        exec_resp = client.post(f"/api/v1/research/runs/{run_id}/execute")
        assert exec_resp.status_code in (400, 404)
        detail = exec_resp.json()["detail"].lower()
        assert "strategy" in detail
        assert "totally_unknown_strategy_xyz" in detail or "not found" in detail or "unknown" in detail
