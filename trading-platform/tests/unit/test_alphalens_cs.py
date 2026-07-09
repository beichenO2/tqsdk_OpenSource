"""Tests for Alphalens-style cross-sectional IC + alpha_sets registration."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "packages", ROOT / "packages" / "factor", ROOT / "packages" / "features"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


def test_cross_sectional_ic_and_quantile():
    from factor.alphalens_cs import analyze_cross_section, cross_sectional_ic, quantile_returns

    rng = np.random.default_rng(0)
    n_t, n_a = 80, 8
    idx = pd.date_range("2024-01-01", periods=n_t, freq="h")
    assets = [f"s{i}" for i in range(n_a)]

    # factor positively predicts next return cross-sectionally
    close = pd.DataFrame(100 + rng.normal(0, 1, (n_t, n_a)).cumsum(axis=0), index=idx, columns=assets)
    fwd = close.shift(-1) / close - 1
    noise = pd.DataFrame(rng.normal(0, 0.01, (n_t, n_a)), index=idx, columns=assets)
    factor = fwd.shift(1).fillna(0) + noise  # lag so it's known at t

    ic = cross_sectional_ic(factor, close, horizon=1, min_assets=3)
    assert len(ic.dropna()) > 10

    q = quantile_returns(factor, close, horizon=1, quantiles=5, min_assets=5)
    assert q["n_periods"] > 5
    assert "Q1" in q["mean_returns"]
    assert q["long_short"] is not None

    full = analyze_cross_section(factor, close, horizon=1, quantiles=5)
    assert full["mode"] == "cross_sectional"
    assert full["summary"]["n"] and full["summary"]["n"] > 0


def test_alpha158_wq101_registered():
    from factor.registry import list_factor_metas

    metas = list_factor_metas()
    names = {m["name"] for m in metas}
    cats = {m["category"] for m in metas}
    assert "alpha158" in cats
    assert "wq101" in cats
    assert "a158_roc_20" in names
    assert "wq101" in names
    assert "wq006" in names
    assert len([m for m in metas if m["category"] == "alpha158"]) >= 10
    assert len([m for m in metas if m["category"] == "wq101"]) >= 8


def test_alpha_sets_compute_on_ohlcv():
    from factor.registry import compute_factor_frame

    n = 100
    df = pd.DataFrame({
        "open": np.linspace(100, 110, n),
        "high": np.linspace(101, 111, n),
        "low": np.linspace(99, 109, n),
        "close": np.linspace(100, 110, n) + np.sin(np.linspace(0, 6, n)),
        "volume": np.linspace(1000, 2000, n),
    })
    out = compute_factor_frame(df, ["a158_roc_5", "wq101", "wq012"])
    assert "a158_roc_5" in out.columns
    assert "wq101" in out.columns
    assert out["wq101"].notna().sum() > 50
