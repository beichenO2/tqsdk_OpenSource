#!/usr/bin/env python3
"""Deploy volbar champion params to standard runtime locations (Phase A)."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAMPIONS = ROOT / "champions" / "volbar"
DEFAULT_SNAPSHOT = "volbar_r0773_0.9633_20260430_182057"


def _latest_champion_dir() -> Path:
    leaderboard = CHAMPIONS / "leaderboard.json"
    if leaderboard.exists():
        entries = json.loads(leaderboard.read_text())
        if entries:
            snap = entries[0]["snapshot"]
            candidate = CHAMPIONS / snap
            if candidate.exists():
                return candidate
    dirs = sorted(CHAMPIONS.glob("volbar_r*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not dirs:
        raise FileNotFoundError(f"No champion snapshots under {CHAMPIONS}")
    return dirs[0]


def deploy(snapshot: str | None = None) -> dict:
    champ_dir = CHAMPIONS / snapshot if snapshot else _latest_champion_dir()
    params_path = champ_dir / "params.json"
    metrics_path = champ_dir / "metrics.json"
    champion_path = champ_dir / "champion.json"

    if not params_path.exists():
        raise FileNotFoundError(params_path)

    params = json.loads(params_path.read_text())
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    meta = json.loads(champion_path.read_text()) if champion_path.exists() else {}

    model_path = ROOT / "eternal-optimizer" / "models" / "strategy_volbar_best.json"
    if model_path.exists():
        base_params = json.loads(model_path.read_text()).get("parameters", {})
        merged = {**base_params, **params}
        params = merged

    deployed_dir = ROOT / "data" / "deployed_params"
    deployed_dir.mkdir(parents=True, exist_ok=True)

    strategy_name = "volbar_v4_blend"
    deploy_record = {
        "strategy_name": strategy_name,
        "variant": "volbar",
        "source_snapshot": champ_dir.name,
        "deployed_at": datetime.now(timezone.utc).isoformat(),
        "params": params,
        "metrics_summary": {
            "best_score": meta.get("score") or metrics.get("best_score"),
            "portfolio_return_pct": metrics.get("portfolio_return_pct"),
            "oos_gate_passed": metrics.get("oos_gate", {}).get("passed"),
            "oos_gate": metrics.get("oos_gate"),
        },
        "symbols_focus": ["SOLUSDT", "BTCUSDT"],
        "symbols_reduce_weight": ["ETHUSDT"],
        "mode": "paper",
    }

    (deployed_dir / f"{strategy_name}.json").write_text(
        json.dumps(params, indent=2, ensure_ascii=False)
    )

    active_path = ROOT / "data" / "active_deployment.json"
    active_path.write_text(json.dumps(deploy_record, indent=2, ensure_ascii=False))

    model_out = ROOT / "eternal-optimizer" / "models" / "strategy_volbar_best.json"
    model_payload = {
        "strategy_name": "V4 Volbar Breakout (Champion Deploy)",
        "version": champ_dir.name,
        "exported_at": deploy_record["deployed_at"],
        "parameters": params,
        "performance": metrics,
        "deployment_notes": [
            "Auto-deployed from champion_archive snapshot",
            "Paper mode — SOL/BTC focus per OOS per_asset metrics",
        ],
    }
    model_out.write_text(json.dumps(model_payload, indent=2, ensure_ascii=False))

    status_path = ROOT / "eternal-optimizer" / "STATUS.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    status["volbar"] = {
        "round": meta.get("round", 773),
        "generation": status.get("volbar", {}).get("generation", 0),
        "best_ever_score": meta.get("score") or metrics.get("best_score"),
        "total_trials": meta.get("total_trials", metrics.get("total_trials", 0)),
        "last_improvement_round": meta.get("round", 773),
        "deployed_snapshot": champ_dir.name,
        "deployed_at": deploy_record["deployed_at"],
        "paper_mode": True,
        "timestamp": datetime.now().isoformat(),
    }
    status["_updated"] = datetime.now().isoformat()
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False))

    archive_copy = deployed_dir / f"{strategy_name}_manifest.json"
    archive_copy.write_text(json.dumps(deploy_record, indent=2, ensure_ascii=False))

    return deploy_record


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy volbar champion to paper runtime paths")
    parser.add_argument("--snapshot", default=None, help=f"Champion snapshot dir name (default: latest or {DEFAULT_SNAPSHOT})")
    args = parser.parse_args()
    snapshot = args.snapshot or None
    if snapshot is None and (CHAMPIONS / DEFAULT_SNAPSHOT).exists():
        snapshot = DEFAULT_SNAPSHOT
    result = deploy(snapshot)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Auto-run overfit validation after every deploy — WF fail triggers robust search
    py = sys.executable
    venv_py = ROOT / ".venv" / "bin" / "python"
    if venv_py.exists():
        py = str(venv_py)

    probe = subprocess.run(
        [py, str(ROOT / "scripts" / "run_overfit_validation.py")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    print(probe.stdout[-2000:] if probe.stdout else "")
    if probe.returncode != 0:
        print("WARN: overfit gates not all passed — running WF robust search...")
        search = subprocess.run(
            [py, str(ROOT / "scripts" / "run_wf_robust_search.py"), "--trials", "120", "--apply"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        print(search.stdout[-3000:] if search.stdout else "")
        if search.stderr:
            print(search.stderr[-1000:])
        if search.returncode != 0:
            print("FAIL: could not find params passing OOS+WF+MC gates")
            return 1
        probe2 = subprocess.run(
            [py, str(ROOT / "scripts" / "run_overfit_validation.py")],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        print(probe2.stdout[-2000:] if probe2.stdout else "")
        if probe2.returncode != 0:
            print("FAIL: overfit validation still failing after robust search")
            return 1
    print("PASS: all overfit gates (OOS + WF + MC)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
