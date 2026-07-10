"""Lightweight test for BTC factor mining script (synthetic bars, small MCTS)."""

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


def _synth_ohlcv(n: int = 1500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 20000 + np.cumsum(rng.normal(0, 50, n))
    high = close + rng.uniform(5, 40, n)
    low = close - rng.uniform(5, 40, n)
    open_ = close + rng.normal(0, 10, n)
    volume = rng.integers(100, 5000, n).astype(float)
    dt = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "datetime": dt,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def test_btc_mining_script_writes_report_and_json(tmp_path: Path):
    from run_btc_factor_mining import main, run_pipeline

    df = _synth_ohlcv(1500)
    out_dir = tmp_path / "research"

    with patch("run_btc_factor_mining.load_btc_ohlcv", return_value=df):
        result = main([
            "--iterations", "10",
            "--train-bars", "900",
            "--test-bars", "600",
            "--min-valid", "100",
            "--skip-cs",
            "--out-dir", str(out_dir),
        ])

    report = Path(result["report_path"])
    jpath = Path(result["json_path"])
    assert report.exists()
    assert jpath.exists()
    assert report.name.startswith("btc_mining_report_")
    assert report.suffix == ".md"

    text = report.read_text(encoding="utf-8")
    assert "## OOS Survival" in text
    assert "| expr | train IC |" in text
    assert "## Conclusions" in text

    payload = json.loads(jpath.read_text(encoding="utf-8"))
    assert "ts" in payload
    assert payload["timeframe"] == "1h"
    assert isinstance(payload["factors"], list)
    for f in payload["factors"]:
        assert "expr" in f
        assert "train_ic" in f
        assert "test_ic" in f
        assert "test_ir" in f

    assert "n_elite" in result
    assert "n_qualified" in result
    assert "n_survived" in result
    assert isinstance(result["oos_rows"], list)
    for row in result["oos_rows"]:
        assert set(row.keys()) >= {
            "expr", "train_ic", "train_ir", "test_ic", "test_ir", "n_valid", "status",
        }
        assert row["status"] in {
            "survived", "failed_ic", "failed_sign", "insufficient", "error",
        }


def test_run_pipeline_direct(tmp_path: Path):
    from run_btc_factor_mining import run_pipeline

    df = _synth_ohlcv(1500)
    result = run_pipeline(
        df,
        iterations=5,
        train_bars=900,
        test_bars=600,
        min_valid=80,
        skip_cs=True,
        out_dir=tmp_path,
        ts="testrun",
    )
    assert (tmp_path / "btc_mining_report_testrun.md").exists()
    assert (tmp_path / "btc_surviving_factors.json").exists()
    assert result["split_info"]["n_train"] == 900
    assert result["split_info"]["n_test"] == 600
