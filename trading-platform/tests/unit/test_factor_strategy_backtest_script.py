"""Lightweight tests for BTC factor→strategy backtest script (synthetic data)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
PACKAGES = ROOT / "packages"
for p in (PACKAGES, SCRIPTS):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


REQUIRED_METRIC_KEYS = {
    "total_return",
    "annual_return",
    "sharpe",
    "max_drawdown_pct",
    "win_rate",
    "profit_factor",
    "total_trades",
    "turnover",
    "cost_ratio",
}


def _synth_ohlcv(n: int = 2000, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 50000 + np.cumsum(rng.normal(0, 80, n))
    high = close + rng.uniform(10, 60, n)
    low = close - rng.uniform(10, 60, n)
    open_ = close + rng.normal(0, 20, n)
    volume = rng.integers(100, 5000, n).astype(float)
    dt = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "datetime": dt,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _fake_factors_json(path: Path) -> Path:
    """Three factors: two nearly identical (corr>~0.9), one orthogonal-ish."""
    payload = {
        "ts": "test",
        "timeframe": "1h",
        "factors": [
            {
                "expr": "roc(close, 10)",
                "train_ic": -0.08,
                "test_ic": -0.05,
                "test_ir": -0.4,
            },
            {
                "expr": "delta(close, 10)",
                "train_ic": -0.07,
                "test_ic": -0.04,
                "test_ir": -0.3,
            },
            {
                "expr": "ts_std(volume, 20)",
                "train_ic": 0.03,
                "test_ic": 0.01,
                "test_ir": 0.1,
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_greedy_dedup_keeps_one_of_highly_correlated():
    from run_btc_factor_strategy_backtest import (
        evaluate_factors_on_df,
        greedy_cluster_by_corr,
    )

    df = _synth_ohlcv(800)
    factors = [
        {"expr": "roc(close, 10)", "train_ic": -0.08, "test_ic": -0.05, "test_ir": -0.4},
        {"expr": "delta(close, 10)", "train_ic": -0.07, "test_ic": -0.04, "test_ir": -0.3},
        {"expr": "ts_std(volume, 20)", "train_ic": 0.03, "test_ic": 0.01, "test_ir": 0.1},
    ]
    frame = evaluate_factors_on_df(df, [f["expr"] for f in factors])
    # roc(close,10) and delta(close,10)/close are highly related on trending noise
    corr = frame.corr(method="spearman").abs()
    assert float(corr.iloc[0, 1]) >= 0.7

    clusters = greedy_cluster_by_corr(factors, frame, threshold=0.7)
    reps = [c["representative"]["expr"] for c in clusters]
    assert len(reps) < 3
    assert "roc(close, 10)" in reps  # higher |train_ic|
    assert "delta(close, 10)" not in reps
    assert any("ts_std" in r for r in reps)


def test_main_writes_report_with_metrics(tmp_path: Path):
    from run_btc_factor_strategy_backtest import main

    df = _synth_ohlcv(2000)
    factors_path = _fake_factors_json(tmp_path / "btc_surviving_factors.json")
    out_dir = tmp_path / "research"

    with patch("run_btc_factor_strategy_backtest.load_btc_ohlcv", return_value=df):
        result = main(
            [
                "--factors-json",
                str(factors_path),
                "--train-bars",
                "1200",
                "--test-bars",
                "800",
                "--dedup-eval-bars",
                "1200",
                "--out-dir",
                str(out_dir),
                "--corr-threshold",
                "0.7",
            ]
        )

    report = Path(result["report_path"])
    assert report.exists()
    assert report.name.startswith("btc_factor_strategy_report_")
    text = report.read_text(encoding="utf-8")
    assert "防泄漏" in text or "leakage" in text.lower() or "信息泄漏" in text
    assert "FactorStrategy" in text or "factor_strategy" in text

    assert "clusters" in result
    assert len(result["clusters"]) >= 1
    # Highly correlated pair should collapse
    assert len(result["representatives"]) < 3

    for row in result["metrics_table"]:
        missing = REQUIRED_METRIC_KEYS - set(row.keys())
        assert not missing, f"missing metrics {missing} in {row.get('name')}"

    names = {r["name"] for r in result["metrics_table"]}
    assert "factor_strategy" in names
    assert "composite_z" in names
    assert "buy_and_hold" in names
    assert "supertrend" in names
