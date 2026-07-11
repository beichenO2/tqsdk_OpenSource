"""Tests for run_h2_robustness_grid.py and robustness assessment."""

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


def _synth_ohlcv(n: int, freq: str, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.005, n)
    close = 30_000 + np.cumsum(ret * 100)
    dt = pd.date_range("2020-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({
        "open_time": dt,
        "open": close,
        "high": close * 1.002,
        "low": close * 0.998,
        "close": close,
        "volume": rng.integers(100, 5000, n).astype(float),
    })


def _grid_row(mom: int, band: float, net: float) -> dict:
    return {
        "timeframe": "1d",
        "momentum_days": mom,
        "band": band,
        "net_return_5bp": net,
        "net_sharpe": 0.1,
        "annual_turnover": 10.0,
        "trade_expectancy_bp": 5.0,
        "gates_passed": 2,
        "verdict": "REJECT",
    }


class TestH2RobustnessAssessment:
    def test_all_positive_passes(self):
        from run_h2_robustness_grid import assess_timeframe_robustness

        rows = [
            _grid_row(m, b, 0.05)
            for m in (5, 7, 10)
            for b in (0.2, 0.3, 0.4)
        ]
        result = assess_timeframe_robustness(rows)
        assert result["passed"]
        assert result["positive_count"] == 9

    def test_half_positive_fails(self):
        from run_h2_robustness_grid import assess_timeframe_robustness

        rows = [
            _grid_row(m, b, 0.05 if i < 4 else -0.01)
            for i, (m, b) in enumerate(
                (mom, band)
                for mom in (5, 7, 10)
                for band in (0.2, 0.3, 0.4)
            )
        ]
        result = assess_timeframe_robustness(rows)
        assert not result["passed"]
        assert result["positive_count"] == 4

    def test_center_isolated_peak_fails(self):
        from run_h2_robustness_grid import assess_timeframe_robustness

        rows = []
        for m in (5, 7, 10):
            for b in (0.2, 0.3, 0.4):
                if m == 7 and b == 0.3:
                    net = 0.30
                else:
                    net = 0.02
                rows.append(_grid_row(m, b, net))
        result = assess_timeframe_robustness(rows)
        assert not result["passed"]
        assert result["positive_count"] == 9
        assert result["peak_ratio"] > 3.0


class TestH2RobustnessGridScript:
    def test_grid_nine_rows_and_report_generation(self, tmp_path: Path):
        from run_h2_robustness_grid import run_pipeline

        frames = {
            "1d": _synth_ohlcv(800, "D", seed=11),
            "4h": _synth_ohlcv(3000, "4h", seed=22),
        }
        out_dir = tmp_path / "research"
        data_dir = tmp_path / "data"

        def fake_load(symbol, timeframe, start=None, end=None):
            return frames.get(timeframe, pd.DataFrame()).copy()

        with patch("run_h2_robustness_grid.CryptoDataLoader") as mock_cls:
            mock_loader = mock_cls.return_value
            mock_loader.load.side_effect = fake_load
            mock_loader.data_dir = data_dir
            result = run_pipeline(
                symbol="BTCUSDT",
                data_dir=data_dir,
                out_dir=out_dir,
            )

        assert len(result["rows"]) == 18  # 9 per timeframe × 2
        for tf in ("1d", "4h"):
            tf_rows = [r for r in result["rows"] if r["timeframe"] == tf]
            assert len(tf_rows) == 9

        md_path = Path(result["report_path"])
        json_path = Path(result["json_path"])
        assert md_path.exists()
        assert json_path.exists()

        text = md_path.read_text(encoding="utf-8")
        assert "H2 Vol-Adj Momentum Robustness Grid" in text
        assert "robustness verdict" in text

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert len(payload["rows"]) == 18
        assert "robustness" in payload
        assert set(payload["robustness"].keys()) == {"1d", "4h"}
