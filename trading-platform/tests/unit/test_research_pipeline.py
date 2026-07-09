"""Tests for ResearchRun pipeline derivation and promote gates."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))

from experiment.research_run import ResearchRun, RunStatus


def test_pipeline_idea_only():
    run = ResearchRun(prompt="test dual ma on rb", strategy_name="dual_ma")
    p = run.derive_pipeline()
    assert p["total"] == 8
    assert p["steps"][0]["status"] == "done"
    assert p["steps"][1]["status"] == "active"
    assert p["promotion"] == "research"


def test_pipeline_with_factor_and_gate():
    run = ResearchRun(prompt="x", strategy_name="s")
    run.factor_snapshot = {"ic": {"ic_mean": 0.05}, "dedupe": {"kept": ["a"]}}
    run.model_snapshot = {"algo": "lgbm"}
    run.metrics = {"sharpe": 1.2}
    run.status = RunStatus.COMPLETED
    run.add_validation("oos", True, {"sharpe": 1.1})
    p = run.derive_pipeline()
    assert p["steps"][0]["done"] is True
    assert p["steps"][1]["done"] is True
    assert p["steps"][2]["done"] is True
    assert p["gate_passed"] is True


def test_promote_gates():
    run = ResearchRun(prompt="p", strategy_name="s")
    ok, _ = run.can_promote_to("backtest")
    assert ok
    run.promotion = "backtest"
    ok, msg = run.can_promote_to("paper")
    assert not ok
    run.status = RunStatus.COMPLETED
    run.metrics = {"sharpe": 1.0}
    ok, _ = run.can_promote_to("paper")
    assert ok
    run.promotion = "paper"
    ok, _ = run.can_promote_to("live")
    assert ok


def test_wq_expanded_count():
    sys.path.insert(0, str(ROOT / "packages" / "features"))
    from factor.registry import list_factor_metas

    wq = [m for m in list_factor_metas() if m["category"] == "wq101"]
    assert len(wq) >= 20
