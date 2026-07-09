"""Unit tests for packages/factor analysis + combine."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "packages", ROOT / "packages" / "factor", ROOT / "packages" / "features"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


def test_factor_ic_and_summary():
    from factor.analysis import factor_ic, summarize_ic

    rng = np.random.default_rng(42)
    n = 200
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)))
    # positively predictive noisy factor
    fwd = close.shift(-1) / close - 1
    factor = fwd.shift(1).fillna(0) + rng.normal(0, 0.01, n)
    factor = pd.Series(factor.to_numpy(), index=close.index)

    ic = factor_ic(factor, close, horizon=1)
    summary = summarize_ic(ic)
    assert summary["n"] and summary["n"] > 10
    assert summary["ic_mean"] is not None


def test_dedupe_and_combine():
    from factor.analysis import deduplicate_factors
    from factor.combine import combine_equal_weight, orthogonalize

    idx = pd.RangeIndex(100)
    a = pd.Series(np.linspace(0, 1, 100), index=idx, name="a")
    b = a * 0.999 + 1e-6  # nearly identical
    c = pd.Series(np.sin(np.linspace(0, 6, 100)), index=idx, name="c")
    df = pd.DataFrame({"a": a, "b": b, "c": c})

    dedupe = deduplicate_factors(df, threshold=0.99)
    assert "a" in dedupe["kept"]
    assert "b" in dedupe["dropped"]

    combined = combine_equal_weight(df[["a", "c"]])
    assert len(combined.dropna()) > 50

    orth = orthogonalize(df[["a", "c"]])
    assert list(orth.columns) == ["a_orth", "c_orth"]


def test_registry_lists_builtin_factors():
    from factor.registry import list_factor_metas

    metas = list_factor_metas()
    names = {m["name"] for m in metas}
    assert "rsi" in names
    assert "macd" in names
    assert len(metas) >= 10
