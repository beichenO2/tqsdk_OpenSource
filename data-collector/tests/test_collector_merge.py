"""Tests for incremental crypto kline merging in data-collector.

Guards against the regression where collect_crypto_klines overwrote full
history parquet files with a 500-bar window.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collector import (  # noqa: E402
    _atomic_write_parquet,
    _merge_incremental,
    _normalize_binance_klines,
)


def _make_bars(start: str, periods: int, freq: str = "1h") -> pd.DataFrame:
    times = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    return pd.DataFrame({
        "open_time": times,
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 10.0,
        "close_time": times + pd.Timedelta(minutes=59),
        "quote_volume": 1000.0,
        "trades": 5,
        "taker_buy_volume": 4.0,
        "taker_buy_quote_volume": 400.0,
    })


class TestNormalize:
    def test_renames_taker_columns_and_drops_ignore(self):
        raw = pd.DataFrame([[
            1700000000000, "100", "101", "99", "100.5", "10",
            1700003599999, "1000", 5, "4", "400", "0",
        ]], columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore",
        ])
        out = _normalize_binance_klines(raw)
        assert "taker_buy_volume" in out.columns
        assert "taker_buy_quote_volume" in out.columns
        assert "ignore" not in out.columns
        assert str(out["open_time"].dtype).startswith("datetime64")
        assert out["open_time"].dt.tz is not None
        assert out["close"].dtype == float


class TestMergeIncremental:
    def test_extends_history_without_truncation(self):
        existing = _make_bars("2020-01-01", 5000)
        new = _make_bars("2020-07-27 07:00:00", 10)  # overlaps last bar
        merged = _merge_incremental(existing, new)
        assert len(merged) >= 5000  # history preserved
        assert merged["open_time"].is_monotonic_increasing
        assert not merged["open_time"].duplicated().any()

    def test_overlap_keeps_newest_row(self):
        existing = _make_bars("2024-01-01", 3)
        new = _make_bars("2024-01-01 02:00:00", 2)
        new.loc[0, "close"] = 999.0  # revised bar
        merged = _merge_incremental(existing, new)
        assert len(merged) == 4
        row = merged[merged["open_time"] == pd.Timestamp("2024-01-01 02:00:00", tz="UTC")]
        assert row["close"].iloc[0] == 999.0

    def test_empty_existing(self):
        new = _make_bars("2024-01-01", 3)
        merged = _merge_incremental(None, new)
        assert len(merged) == 3


class TestAtomicWriteFollowsSymlink:
    def test_write_through_symlink_updates_target(self, tmp_path):
        target_dir = tmp_path / "downloads"
        cache_dir = tmp_path / "cache"
        target_dir.mkdir()
        cache_dir.mkdir()

        target = target_dir / "1h.parquet"
        _make_bars("2020-01-01", 100).to_parquet(target, index=False)

        link = cache_dir / "1h.parquet"
        os.symlink(target, link)

        merged = _merge_incremental(
            pd.read_parquet(link), _make_bars("2020-01-05 04:00:00", 10)
        )
        real = _atomic_write_parquet(merged, link)

        assert real == target
        assert link.is_symlink()  # link not replaced by a regular file
        assert len(pd.read_parquet(target)) >= 100


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
