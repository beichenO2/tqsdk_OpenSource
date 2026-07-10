"""Tests for MCTS factor expression search + subtree avoidance + experience store."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

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


def _ohlcv(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame({
        "open": close + rng.normal(0, 0.1, n),
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": rng.integers(1000, 5000, n).astype(float),
    })


def test_ucb1_prefers_high_reward_low_visits():
    from factor.mcts_search import MCTSNode, select_ucb1

    root = MCTSNode(expr="root", parent=None)
    root.visits = 10
    root.total_reward = 0.0

    high_reward = MCTSNode(expr="a", parent=root)
    high_reward.visits = 4
    high_reward.total_reward = 3.2  # mean=0.8

    low_visits = MCTSNode(expr="b", parent=root)
    low_visits.visits = 1
    low_visits.total_reward = 0.3  # mean=0.3 but underexplored

    mediocre = MCTSNode(expr="c", parent=root)
    mediocre.visits = 5
    mediocre.total_reward = 1.0  # mean=0.2

    root.children = [high_reward, low_visits, mediocre]

    # UCB1: low_visits gets large exploration bonus → selected at c=1.4
    # high: 0.8 + 1.4*sqrt(ln(10)/4) ≈ 0.8+1.06=1.86
    # low:  0.3 + 1.4*sqrt(ln(10)/1) ≈ 0.3+2.12=2.42
    chosen = select_ucb1(root, c=1.4)
    assert chosen is low_visits

    # With c=0, pure exploitation → high_reward
    chosen_exploit = select_ucb1(root, c=0.0)
    assert chosen_exploit is high_reward


def test_expand_rejects_invalid_expr_into_experience(tmp_path: Path):
    from factor.mcts_search import ExperienceStore, MCTSNode, expand_node

    store = ExperienceStore(tmp_path / "experience.jsonl")
    parent = MCTSNode(expr="roc(close, 5)", parent=None)
    parent.visits = 1

    with patch("factor.mcts_search.mutate_expression", return_value="import os"):
        children = expand_node(
            parent,
            k=1,
            use_llm=False,
            experience=store,
            subtree_counts={},
            subtree_threshold=5,
        )

    assert children == []
    assert parent.children == []
    records = store.load_all()
    assert len(records) >= 1
    assert records[-1]["reason"] == "validation_error"
    assert records[-1]["expr"] == "import os"


def test_subtree_frequency_penalty(tmp_path: Path):
    from factor.mcts_search import (
        ExperienceStore,
        apply_subtree_penalty,
        extract_subtrees,
    )

    store = ExperienceStore(tmp_path / "experience.jsonl")
    expr = "ts_rank(delta(close, 1), 5)"
    subs = extract_subtrees(expr)
    assert len(subs) >= 1

    # Force one subtree over threshold
    counts: dict[str, int] = {}
    for s in subs:
        counts[s] = 6  # > default threshold 5

    raw_reward = 1.0
    penalized, hit = apply_subtree_penalty(
        expr,
        raw_reward,
        counts,
        threshold=5,
        penalty=0.5,
        experience=store,
    )
    assert hit is True
    assert penalized == pytest.approx(0.5)
    records = store.load_all()
    assert any(r["reason"] == "overused_subtree" for r in records)


def test_experience_store_roundtrip(tmp_path: Path):
    from factor.mcts_search import ExperienceStore

    path = tmp_path / "experience.jsonl"
    store = ExperienceStore(path)
    store.record("roc(close, 5)", "low_ic", ic_mean=0.001, ir=0.05)
    store.record("bad()", "validation_error")
    store.record("roc(close, 5)", "low_ic", ic_mean=0.002, ir=0.1)
    store.record("dup", "duplicate")

    reloaded = ExperienceStore(path)
    all_recs = reloaded.load_all()
    assert len(all_recs) == 4
    assert all_recs[0]["expr"] == "roc(close, 5)"
    assert all_recs[0]["reason"] == "low_ic"

    patterns = reloaded.top_failure_patterns(2)
    assert len(patterns) >= 1
    assert patterns[0]["reason"] == "low_ic"
    assert patterns[0]["count"] == 2
    assert "expr" in patterns[0]


def test_run_mcts_search_e2e(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from factor import mcts_search as ms

    # Redirect output dirs into tmp
    monkeypatch.setattr(ms, "OUT_DIR", tmp_path / "factor_evolution")
    ms.OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = _ohlcv(500)
    result = ms.run_mcts_search(
        df,
        n_iterations=20,
        use_llm=False,
        experience_path=tmp_path / "experience.jsonl",
        seed_exprs=["roc(close, 5)", "roc(close, 20)", "-ts_std(close, 20)"],
    )

    assert "candidates" in result
    assert len(result["candidates"]) >= 1
    assert "elite" in result
    assert "qualified" in result
    assert "tree_stats" in result
    stats = result["tree_stats"]
    assert "nodes" in stats
    assert "max_depth" in stats
    assert "subtree_penalties" in stats
    assert "path" in result

    cand0 = result["candidates"][0]
    for key in ("expr", "score", "ic_mean", "ir", "dedupe_ok", "max_corr", "error", "meta"):
        assert key in cand0
    assert cand0["meta"].get("source") == "mcts"
    assert "depth" in cand0["meta"]
    assert "visits" in cand0["meta"]

    out_path = Path(result["path"])
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert "candidates" in payload


def test_evolve_mcts_api_endpoint():
    from apps.api.app.routers import factors as factors_router

    app = FastAPI()
    app.include_router(factors_router.router)

    mock_result = {
        "ts": 1.0,
        "candidates": [
            {
                "expr": "roc(close, 5)",
                "source": "mcts",
                "score": 0.5,
                "ic_mean": 0.04,
                "ir": 0.6,
                "dedupe_ok": True,
                "max_corr": None,
                "error": None,
                "meta": {"source": "mcts", "depth": 1, "visits": 3},
            },
            {
                "expr": "roc(close, 10)",
                "source": "mcts",
                "score": 0.3,
                "ic_mean": 0.02,
                "ir": 0.35,
                "dedupe_ok": True,
                "max_corr": None,
                "error": None,
                "meta": {"source": "mcts", "depth": 2, "visits": 1},
            },
        ],
        "elite": [],
        "qualified": [],
        "tree_stats": {"nodes": 10, "max_depth": 3, "subtree_penalties": 0},
        "path": "output/factor_evolution/mcts_round_1.json",
    }
    # classify will fill elite/qualified from candidates
    fake_df = pd.DataFrame({
        "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0],
    })

    with patch.object(factors_router, "run_mcts_search", return_value=mock_result):
        with patch.object(factors_router, "_load_crypto_ohlcv", return_value=fake_df):
            client = TestClient(app)
            resp = client.post(
                "/factors/evolve-mcts",
                json={
                    "symbol": "BTCUSDT",
                    "n_iterations": 10,
                    "use_llm": False,
                    "timeframe": "1h",
                    "limit": 1000,
                },
            )

    assert resp.status_code == 200
    body = resp.json()
    assert "elite" in body
    assert "qualified" in body
    assert "registered" in body
    assert "tree_stats" in body
    assert len(body["elite"]) == 1
    assert len(body["qualified"]) == 1
    assert body["registered"] == ["evolved_0"]
