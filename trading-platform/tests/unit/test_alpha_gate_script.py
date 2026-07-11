"""Tests for run_alpha_gate_btc.py (mocked data loader)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
PACKAGES = ROOT / "packages"
for p in (PACKAGES, SCRIPTS):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


def _synth_ohlcv(n: int = 3000, freq: str = "4h", seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.005, n)
    close = 30_000 + np.cumsum(ret * 100)
    dt = pd.date_range("2020-01-01", periods=n, freq="4h" if freq == "4h" else "D", tz="UTC")
    return pd.DataFrame({
        "open_time": dt,
        "open": close,
        "high": close * 1.002,
        "low": close * 0.998,
        "close": close,
        "volume": rng.integers(100, 5000, n).astype(float),
    })


def test_run_alpha_gate_script_generates_report(tmp_path: Path):
    from run_alpha_gate_btc import main

    ohlcv_4h = _synth_ohlcv(3000, "4h")
    ohlcv_1d = _synth_ohlcv(500, "1d", seed=22)
    out_dir = tmp_path / "research"
    data_dir = tmp_path / "data"

    def fake_load(symbol, timeframe, start=None, end=None):
        if timeframe == "4h":
            return ohlcv_4h.copy()
        if timeframe == "1d":
            return ohlcv_1d.copy()
        return pd.DataFrame()

    with patch("run_alpha_gate_btc.CryptoDataLoader") as mock_loader_cls:
        mock_loader = mock_loader_cls.return_value
        mock_loader.load.side_effect = fake_load
        mock_loader.data_dir = data_dir
        result = main(["--out-dir", str(out_dir), "--data-dir", str(data_dir)])

    report = Path(result["report_path"])
    jpath = Path(result["json_path"])
    assert report.exists()
    assert jpath.exists()
    text = report.read_text(encoding="utf-8")
    assert "H1" in text
    assert "H2" in text
    assert "H3" in text
    assert "H4" in text
    assert "SKIPPED" in text  # no funding data in tmp_path

    payload = json.loads(jpath.read_text(encoding="utf-8"))
    assert len(payload["factors"]) == 4
    ids = {f["id"] for f in payload["factors"]}
    assert ids == {"H1", "H2", "H3", "H4"}
