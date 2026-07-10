"""Tests for multi-crypto cross-sectional factor pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "packages", ROOT / "packages" / "factor", ROOT / "packages" / "features"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

CRYPTO_COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades",
    "taker_buy_volume",
    "taker_buy_quote_volume",
]


def _write_crypto_parquet(data_dir: Path, symbol: str, n: int = 120, *, offset: float = 0.0) -> None:
    sym_dir = data_dir / symbol.lower()
    sym_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(hash(symbol) % 2**32)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = 100 + offset + np.cumsum(rng.normal(0, 0.3, n))
    df = pd.DataFrame(
        {
            "open_time": idx,
            "open": close + rng.normal(0, 0.05, n),
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": rng.integers(1000, 5000, n).astype(float),
            "quote_volume": rng.integers(1e5, 5e5, n).astype(float),
            "trades": rng.integers(100, 500, n),
            "taker_buy_volume": rng.integers(500, 2500, n).astype(float),
            "taker_buy_quote_volume": rng.integers(5e4, 2e5, n).astype(float),
        }
    )
    df.to_parquet(sym_dir / "1h.parquet", index=False)


@pytest.fixture
def crypto_data_dir(tmp_path: Path) -> Path:
    for i, sym in enumerate(["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]):
        _write_crypto_parquet(tmp_path, sym, n=120, offset=float(i))
    return tmp_path


def test_build_cs_panels_shape(crypto_data_dir: Path):
    from factor.cs_pipeline import build_cs_panels

    fpanel, cpanel = build_cs_panels(
        "a158_roc_5",
        ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"],
        timeframe="1h",
        limit=100,
        data_dir=crypto_data_dir,
    )
    assert fpanel.shape == cpanel.shape
    assert fpanel.shape[1] == 4
    assert len(fpanel) >= 50
    assert list(fpanel.columns) == list(cpanel.columns)


def test_run_cs_analysis_fields(crypto_data_dir: Path):
    from factor.cs_pipeline import run_cs_analysis

    result = run_cs_analysis(
        "a158_roc_5",
        ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"],
        timeframe="1h",
        limit=100,
        quantiles=5,
        data_dir=crypto_data_dir,
    )
    assert result["mode"] == "cross_sectional"
    assert "summary" in result
    assert result["summary"]["n"] is not None
    assert "quantile_returns" in result
    assert "Q1" in result["quantile_returns"]["mean_returns"]
    assert result["n_assets"] == 4


def test_build_cs_panels_from_expr(crypto_data_dir: Path):
    from factor.cs_pipeline import build_cs_panels_from_expr

    fpanel, cpanel = build_cs_panels_from_expr(
        "roc(close, 5)",
        ["BTCUSDT", "ETHUSDT"],
        timeframe="1h",
        limit=80,
        data_dir=crypto_data_dir,
    )
    assert fpanel.shape[1] == 2
    assert fpanel.notna().sum().min() > 20


def test_build_cs_panels_rejects_single_symbol(crypto_data_dir: Path):
    from factor.cs_pipeline import build_cs_panels

    with pytest.raises(ValueError, match="2"):
        build_cs_panels("a158_roc_5", ["BTCUSDT"], data_dir=crypto_data_dir)


def test_analyze_cs_crypto_api(crypto_data_dir: Path):
    from apps.api.app.routers import factors as factors_router

    app = FastAPI()
    app.include_router(factors_router.router)

    with patch.object(factors_router, "DEFAULT_CRYPTO_DATA_DIR", crypto_data_dir):
        client = TestClient(app)
        resp = client.post(
            "/factors/analyze-cs-crypto",
            json={
                "factor_name": "a158_roc_5",
                "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"],
                "timeframe": "1h",
                "limit": 100,
                "quantiles": 5,
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "cross_sectional"
    assert body["symbols_used"] == ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
