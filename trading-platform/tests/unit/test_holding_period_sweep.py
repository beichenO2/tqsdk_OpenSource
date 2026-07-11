"""Tests for run_holding_period_sweep.py (mocked data loader)."""

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


def _fake_available_timeframes(symbol: str) -> list[str]:
    return ["1h", "4h"]


def _make_fake_load(frames: dict[str, pd.DataFrame]):
    def fake_load(symbol, timeframe, start=None, end=None):
        return frames.get(timeframe, pd.DataFrame()).copy()

    return fake_load


class TestHoldingPeriodSweep:
    def test_matrix_row_count_and_report_generation(self, tmp_path: Path):
        from run_holding_period_sweep import run_pipeline

        frames = {
            "1h": _synth_ohlcv(5000, "1h", seed=11),
            "4h": _synth_ohlcv(3000, "4h", seed=22),
        }
        out_dir = tmp_path / "research"
        data_dir = tmp_path / "data"

        with patch("run_holding_period_sweep.CryptoDataLoader") as mock_cls:
            mock_loader = mock_cls.return_value
            mock_loader.load.side_effect = _make_fake_load(frames)
            mock_loader.available_timeframes.side_effect = _fake_available_timeframes
            mock_loader.data_dir = data_dir
            result = run_pipeline(
                symbol="BTCUSDT",
                data_dir=data_dir,
                out_dir=out_dir,
            )

        # 2 periods × 2 factors × 3 suppressions = 12 rows
        assert len(result["rows"]) == 12

        md_path = Path(result["report_path"])
        json_path = Path(result["json_path"])
        assert md_path.exists()
        assert json_path.exists()

        text = md_path.read_text(encoding="utf-8")
        assert "Holding Period" in text or "holding_period" in text.lower()
        assert "vol_adj_momentum" in text
        assert "short_high_momentum" in text
        assert "Auto Conclusions" in text or "自动结论" in text

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert len(payload["rows"]) == 12
        assert set(payload["available_timeframes"]) == {"1h", "4h"}

    def test_band_suppression_lowers_turnover(self, tmp_path: Path):
        from run_holding_period_sweep import run_pipeline

        frames = {
            "4h": _synth_ohlcv(3000, "4h", seed=33),
        }
        out_dir = tmp_path / "research"
        data_dir = tmp_path / "data"

        with patch("run_holding_period_sweep.CryptoDataLoader") as mock_cls:
            mock_loader = mock_cls.return_value
            mock_loader.load.side_effect = _make_fake_load(frames)
            mock_loader.available_timeframes.return_value = ["4h"]
            mock_loader.data_dir = data_dir
            result = run_pipeline(
                symbol="BTCUSDT",
                data_dir=data_dir,
                out_dir=out_dir,
            )

        rows = result["rows"]
        for factor in ("H2_vol_adj_momentum", "H4_short_high_momentum"):
            none_row = next(
                r for r in rows
                if r["timeframe"] == "4h" and r["factor"] == factor and r["suppression"] == "none"
            )
            band_row = next(
                r for r in rows
                if r["timeframe"] == "4h" and r["factor"] == factor and r["suppression"] == "band_0.3"
            )
            assert band_row["annual_turnover"] < none_row["annual_turnover"], (
                f"{factor}: band turnover {band_row['annual_turnover']} "
                f"should be < none {none_row['annual_turnover']}"
            )
