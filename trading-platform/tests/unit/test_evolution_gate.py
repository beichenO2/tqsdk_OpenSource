"""Tests for CogAlpha-style dual-threshold evolution gating + /evolve auto-register."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "packages", ROOT / "packages" / "factor", ROOT / "packages" / "features"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


def _cand(
    expr: str,
    *,
    ic_mean: float | None = 0.02,
    ir: float | None = 0.4,
    dedupe_ok: bool = True,
    error: str | None = None,
    score: float | None = 0.5,
) -> dict[str, Any]:
    return {
        "expr": expr,
        "source": "mutate",
        "score": score,
        "ic_mean": ic_mean,
        "ir": ir,
        "dedupe_ok": dedupe_ok,
        "max_corr": None,
        "error": error,
        "meta": {},
    }


def test_classify_elite_qualified_rejected():
    from factor.evolution_registry import classify_candidates

    payload = {
        "candidates": [
            _cand("roc(close, 5)", ic_mean=0.04, ir=0.6),  # elite
            _cand("roc(close, 10)", ic_mean=0.02, ir=0.35),  # qualified only
            _cand("roc(close, 20)", ic_mean=0.01, ir=0.4),  # rejected: low IC
            _cand("roc(close, 30)", ic_mean=0.02, ir=0.2),  # rejected: low IR
            _cand("roc(close, 40)", ic_mean=-0.04, ir=-0.55),  # elite (negative ok)
            _cand("roc(close, 50)", ic_mean=0.025, ir=0.45, dedupe_ok=False),  # rejected
            _cand("bad()", ic_mean=0.05, ir=0.6, error="syntax"),  # rejected
        ]
    }
    out = classify_candidates(payload)
    elite_exprs = {c["expr"] for c in out["elite"]}
    qual_exprs = {c["expr"] for c in out["qualified"]}
    rej_exprs = {c["expr"] for c in out["rejected"]}

    assert "roc(close, 5)" in elite_exprs
    assert "roc(close, 40)" in elite_exprs
    assert "roc(close, 10)" in qual_exprs
    assert "roc(close, 5)" not in qual_exprs  # elite not duplicated in qualified
    assert "roc(close, 20)" in rej_exprs
    assert "roc(close, 30)" in rej_exprs
    assert "roc(close, 50)" in rej_exprs
    assert "bad()" in rej_exprs


def test_classify_threshold_boundary():
    from factor.evolution_registry import classify_candidates

    payload = {
        "candidates": [
            _cand("a", ic_mean=0.015, ir=0.3),  # qualified boundary
            _cand("b", ic_mean=0.03, ir=0.5),  # elite boundary
            _cand("c", ic_mean=0.0149, ir=0.35),  # rejected
            _cand("d", ic_mean=0.02, ir=0.299),  # rejected
            _cand("e", ic_mean=0.029, ir=0.55),  # qualified not elite
        ]
    }
    out = classify_candidates(payload)
    qual_exprs = {c["expr"] for c in out["qualified"]}
    elite_exprs = {c["expr"] for c in out["elite"]}
    assert "a" in qual_exprs
    assert "b" in elite_exprs
    assert "e" in qual_exprs and "e" not in elite_exprs
    assert "c" not in qual_exprs
    assert "d" not in qual_exprs


def test_register_evolved_factors_only_elite(tmp_path: Path):
    from factor.evolution_registry import register_evolved_factors
    from factor.registry import get_registry

    payload = {
        "candidates": [
            _cand("roc(close, 5)", ic_mean=0.04, ir=0.6),
            _cand("roc(close, 10)", ic_mean=0.02, ir=0.35),
            _cand("delta(close, 1)", ic_mean=0.01, ir=0.1),
        ]
    }
    path = tmp_path / "latest.json"
    path.write_text(json.dumps(payload))

    names = register_evolved_factors(path)
    assert names == ["evolved_0"]

    reg = get_registry()
    df = pd.DataFrame(
        {
            "open": np.linspace(100, 110, 40),
            "high": np.linspace(101, 111, 40),
            "low": np.linspace(99, 109, 40),
            "close": np.linspace(100, 110, 40),
            "volume": np.full(40, 1000.0),
        }
    )
    out = reg.compute("evolved_0", df)
    assert "evolved_0" in out.columns


def test_evolve_api_response_structure():
    from apps.api.app.routers import factors as factors_router

    app = FastAPI()
    app.include_router(factors_router.router)

    mock_result = {
        "ts": 1.0,
        "arm": "factor_mining",
        "bandit": {},
        "candidates": [
            _cand("roc(close, 5)", ic_mean=0.04, ir=0.6),
            _cand("roc(close, 10)", ic_mean=0.02, ir=0.35),
        ],
        "best": _cand("roc(close, 5)", ic_mean=0.04, ir=0.6),
        "n_valid": 2,
        "path": "output/factor_evolution/round_1.json",
    }

    fake_df = pd.DataFrame(
        {
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1.0],
        }
    )

    with patch.object(factors_router, "run_evolution_round", return_value=mock_result) as mock_run:
        with patch.object(factors_router, "_load_evolution_ohlcv", return_value=fake_df):
            client = TestClient(app)
            resp = client.post(
                "/factors/evolve",
                json={"symbol": "BTCUSDT", "n_proposals": 2, "limit": 200, "use_llm": False},
            )

    assert resp.status_code == 200
    body = resp.json()
    assert "elite" in body
    assert "qualified" in body
    assert "registered" in body
    assert len(body["elite"]) == 1
    assert len(body["qualified"]) == 1
    assert body["registered"] == ["evolved_0"]
    mock_run.assert_called_once()


def test_evolve_crypto_symbol_uses_crypto_loader():
    from apps.api.app.routers import factors as factors_router

    assert factors_router._is_crypto_usdt("BTCUSDT") is True
    assert factors_router._is_crypto_usdt("btcusdt") is False
    assert factors_router._is_crypto_usdt("rb") is False
