"""TDD: AlphaForge-style dynamic time-varying factor weights."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "packages", ROOT / "packages" / "factor", ROOT / "packages" / "features", ROOT):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


def _make_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="h")


def test_no_lookahead_weights_ignore_future_ic():
    """Factor only predictive in second half → early weights must not use late IC."""
    from factor.combine import rolling_ic_weights

    rng = np.random.default_rng(0)
    n = 400
    idx = _make_index(n)
    # Forward returns: random walk noise
    fwd = pd.Series(rng.normal(0, 0.01, n), index=idx)
    # Factor is pure noise in first half, strongly tracks fwd in second half
    factor = pd.Series(rng.normal(0, 1, n), index=idx)
    half = n // 2
    factor.iloc[half:] = fwd.iloc[half:].to_numpy() * 50 + rng.normal(0, 0.01, n - half)

    factor_df = pd.DataFrame({"late_only": factor})
    weights = rolling_ic_weights(
        factor_df,
        fwd,
        window=60,
        min_periods=40,
        smoothing_halflife=1,  # almost no smoothing
        weight_floor=0.0,
    )

    # Early window (before half + window): weight should be near-zero / NaN / unstable low |IC|
    # Compare mid-first-half vs late-second-half absolute signed weight magnitude via raw IC path:
    # At t < half, historical IC window is entirely in noise → |weight| contribution weak.
    early = weights["late_only"].iloc[80:120].dropna()
    late = weights["late_only"].iloc[-40:].dropna()
    assert len(early) > 10 and len(late) > 10
    # Late period should have higher |effective signed weight| mean abs after regime kicks in
    # More importantly: early weights must equal recomputation on truncated history (no future leak)
    cut = half
    w_trunc = rolling_ic_weights(
        factor_df.iloc[:cut],
        fwd.iloc[:cut],
        window=60,
        min_periods=40,
        smoothing_halflife=1,
        weight_floor=0.0,
    )
    common = early.index.intersection(w_trunc.index)
    assert len(common) > 5
    full_early = weights.loc[common, "late_only"]
    trunc_early = w_trunc.loc[common, "late_only"]
    # Truncated history must match full-series early weights (proves no future leak)
    np.testing.assert_allclose(
        full_early.to_numpy(),
        trunc_early.to_numpy(),
        rtol=1e-9,
        atol=1e-12,
        equal_nan=True,
    )


def test_weights_follow_regime_switch():
    """A strong then weak; B weak then strong → weight curves cross."""
    from factor.combine import rolling_ic_weights

    rng = np.random.default_rng(1)
    n = 500
    idx = _make_index(n)
    fwd = pd.Series(rng.normal(0, 0.01, n), index=idx)
    half = n // 2

    a = pd.Series(rng.normal(0, 1, n), index=idx)
    b = pd.Series(rng.normal(0, 1, n), index=idx)
    a.iloc[:half] = fwd.iloc[:half].to_numpy() * 40 + rng.normal(0, 0.01, half)
    b.iloc[half:] = fwd.iloc[half:].to_numpy() * 40 + rng.normal(0, 0.01, n - half)

    factor_df = pd.DataFrame({"A": a, "B": b})
    w = rolling_ic_weights(
        factor_df,
        fwd,
        window=80,
        min_periods=50,
        smoothing_halflife=5,
        weight_floor=0.0,
    )

    # Use abs of signed weights for magnitude comparison
    early_a = w["A"].iloc[120:180].abs().mean()
    early_b = w["B"].iloc[120:180].abs().mean()
    late_a = w["A"].iloc[-60:].abs().mean()
    late_b = w["B"].iloc[-60:].abs().mean()

    assert early_a > early_b, f"early: A={early_a:.3f} B={early_b:.3f}"
    assert late_b > late_a, f"late: A={late_a:.3f} B={late_b:.3f}"
    assert late_a < early_a  # A declines
    assert late_b > early_b  # B rises


def test_negative_ic_factor_is_flipped():
    """Negative-IC factor gets negative weight → contribution aligns with returns."""
    from factor.combine import dynamic_combine, rolling_ic_weights

    rng = np.random.default_rng(2)
    n = 300
    idx = _make_index(n)
    fwd = pd.Series(rng.normal(0, 0.01, n), index=idx)
    # Factor anti-correlated with forward returns
    factor = (-fwd * 40 + rng.normal(0, 0.01, n)).astype(float)
    factor = pd.Series(factor, index=idx)
    factor_df = pd.DataFrame({"neg": factor})

    w = rolling_ic_weights(
        factor_df,
        fwd,
        window=60,
        min_periods=40,
        smoothing_halflife=5,
        weight_floor=0.0,
    )
    # Signed weights should be predominantly negative
    tail = w["neg"].dropna().iloc[-80:]
    assert (tail < 0).mean() > 0.7

    combined = dynamic_combine(
        factor_df,
        fwd,
        window=60,
        min_periods=40,
        smoothing_halflife=5,
    )
    # Combined (after flip) should positively correlate with fwd
    aligned = pd.concat([combined.rename("c"), fwd.rename("r")], axis=1).dropna()
    # Use causal subset after warmup
    aligned = aligned.iloc[100:]
    spearman = aligned["c"].corr(aligned["r"], method="spearman")
    assert spearman > 0.3


def test_ema_smoothing_reduces_variance():
    from factor.combine import rolling_ic_weights

    rng = np.random.default_rng(3)
    n = 400
    idx = _make_index(n)
    fwd = pd.Series(rng.normal(0, 0.01, n), index=idx)
    f1 = pd.Series(fwd.to_numpy() * 10 + rng.normal(0, 0.5, n), index=idx)
    f2 = pd.Series(rng.normal(0, 1, n), index=idx)
    factor_df = pd.DataFrame({"f1": f1, "f2": f2})

    w_fast = rolling_ic_weights(
        factor_df, fwd, window=60, min_periods=40, smoothing_halflife=2, weight_floor=0.0
    )
    w_slow = rolling_ic_weights(
        factor_df, fwd, window=60, min_periods=40, smoothing_halflife=40, weight_floor=0.0
    )
    var_fast = w_fast["f1"].dropna().iloc[100:].var()
    var_slow = w_slow["f1"].dropna().iloc[100:].var()
    assert var_slow < var_fast


def test_dynamic_beats_static_under_regime_switch():
    from factor.combine import compare_static_vs_dynamic

    rng = np.random.default_rng(4)
    n = 600
    idx = _make_index(n)
    # Build close from returns so forward returns are well-defined
    rets = rng.normal(0, 0.01, n)
    close = pd.Series(100 * np.cumprod(1 + rets), index=idx)
    fwd = close.shift(-1) / close - 1.0
    half = n // 2

    a = pd.Series(rng.normal(0, 1, n), index=idx)
    b = pd.Series(rng.normal(0, 1, n), index=idx)
    a.iloc[:half] = fwd.iloc[:half].fillna(0).to_numpy() * 50 + rng.normal(0, 0.02, half)
    b.iloc[half:] = fwd.iloc[half:].fillna(0).to_numpy() * 50 + rng.normal(0, 0.02, n - half)
    # Avoid leaking last NaN fwd into factor
    a = a.fillna(0)
    b = b.fillna(0)

    factor_df = pd.DataFrame({"A": a, "B": b})
    result = compare_static_vs_dynamic(
        factor_df,
        close,
        horizon=1,
        window=80,
        min_periods=50,
        smoothing_halflife=10,
    )
    assert "static" in result and "dynamic" in result
    assert "ic" in result["static"] and "ir" in result["static"]
    assert "ic" in result["dynamic"] and "ir" in result["dynamic"]
    assert "weights_last" in result
    assert result["dynamic"]["ic"] >= result["static"]["ic"]


def test_api_combine_dynamic_method():
    from apps.api.app.routers import factors as factors_router

    app = FastAPI()
    app.include_router(factors_router.router)

    n = 250
    idx = _make_index(n)
    rng = np.random.default_rng(5)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=idx)
    fwd = close.shift(-1) / close - 1
    f_a = (fwd.fillna(0) * 20 + rng.normal(0, 0.1, n)).astype(float)
    f_b = rng.normal(0, 1, n)

    ohlcv = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1.0,
            "rsi": f_a,
            "roc": f_b,
        },
        index=idx,
    )

    def fake_meta(name: str):
        return {"name": name, "category": "tech", "output_columns": [name]}

    with patch.object(factors_router, "_load_ohlcv", return_value=ohlcv[["open", "high", "low", "close", "volume"]]):
        with patch(
            "factor.registry.compute_factor_frame",
            return_value=ohlcv,
        ):
            with patch("factor.registry.get_factor_meta", side_effect=fake_meta):
                client = TestClient(app)
                resp = client.post(
                    "/factors/combine",
                    json={
                        "symbol": "rb",
                        "factor_names": ["rsi", "roc"],
                        "method": "dynamic",
                        "limit": 250,
                        "window": 60,
                        "halflife": 10,
                    },
                )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["method"] == "dynamic"
    assert "compare" in body
    assert "static" in body["compare"]
    assert "dynamic" in body["compare"]
    assert body["combined"]["last"] is not None
