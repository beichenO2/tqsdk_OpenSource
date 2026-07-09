"""Tests for factor expression evolution + bandit."""

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


def _ohlcv(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame({
        "open": close + rng.normal(0, 0.1, n),
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": rng.integers(1000, 5000, n).astype(float),
    })


def test_validate_and_eval_expr():
    from factor.evolution import _validate_expr, evaluate_expression

    assert _validate_expr("roc(close, 5)") is None
    assert _validate_expr("import os") is not None
    assert _validate_expr("__import__('os')") is not None
    df = _ohlcv()
    s = evaluate_expression("roc(close, 5)", df)
    assert s.notna().sum() > 50


def test_bandit_and_round_without_llm():
    from factor.evolution import FactorBandit, run_evolution_round

    b = FactorBandit(epsilon=0.0)
    for _ in range(20):
        arm = b.select()
        b.update(arm, reward=0.3 if arm == "factor_mining" else 0.1)
    snap = b.snapshot()
    assert snap["factor_mining"]["pulls"] + snap["model_tune"]["pulls"] == 20

    df = _ohlcv(250)
    result = run_evolution_round(df, n_proposals=4, use_llm=False, bandit=b)
    assert result["n_valid"] >= 1
    assert result["best"] is not None
    assert "arm" in result
    assert Path(ROOT / result["path"]).exists() or (ROOT / "output" / "factor_evolution" / "latest.json").exists()


def test_dedupe_penalizes_clone():
    from factor.evolution import evaluate_expression, score_candidate

    df = _ohlcv()
    base = evaluate_expression("roc(close, 5)", df)
    existing = pd.DataFrame({"e0": base})
    clone = score_candidate("roc(close, 5)", df, existing_factors=existing)
    assert clone.dedupe_ok is False or (clone.max_corr is not None and clone.max_corr >= 0.99)
