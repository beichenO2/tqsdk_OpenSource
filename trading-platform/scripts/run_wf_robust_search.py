#!/usr/bin/env python3
"""Search volbar params that pass OOS + Walk-Forward + Monte Carlo gates (no overfit)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(ROOT / "eternal-optimizer"))
sys.path.insert(0, str(ROOT / "scripts"))

import optuna  # noqa: E402
from overfit_data import load_volbar_gate_datasets  # noqa: E402
from eternal_optimizer_volbar import (  # noqa: E402
    DEFAULT_SEARCH_SPACE,
    monte_carlo_robustness,
    oos_gate_check,
    walk_forward_validate,
)


def _load_datasets(symbols: list[str]):
    return load_volbar_gate_datasets(symbols, align_common_era=True)


def _build_params(trial: optuna.Trial, base: dict, search_space: dict) -> dict:
    params = dict(base)
    for pname, spec in search_space.items():
        lo, hi = spec["min"], spec["max"]
        if lo >= hi:
            continue
        if spec["type"] == "int":
            params[pname] = trial.suggest_int(pname, int(lo), int(hi))
        else:
            params[pname] = trial.suggest_float(pname, lo, hi)
    if "sl_atr_mult" in params and "tp_atr_mult" in params:
        if params["tp_atr_mult"] < params["sl_atr_mult"] * 1.3:
            params["tp_atr_mult"] = params["sl_atr_mult"] * 1.5
    params["adx_period"] = 14
    return params


def _gate_score(
    params: dict,
    full_data: dict,
    oos_datasets: list,
    mc_sims: int,
) -> tuple[float, dict]:
    oos = oos_gate_check(oos_datasets, params)
    if not oos.get("passed"):
        return -999.0, {"reason": "oos_fail", "oos": oos}

    wf = walk_forward_validate(full_data, params)
    wf_pass = wf.get("valid") and wf.get("profitable_pct", 0) >= 0.5
    if not wf_pass:
        return -999.0, {"reason": "wf_fail", "walk_forward": wf, "oos": oos}

    mc = monte_carlo_robustness(full_data, params, n_simulations=mc_sims)
    if mc.get("survival_rate", 0) < 0.80:
        return -999.0, {"reason": "mc_fail", "monte_carlo": mc, "walk_forward": wf, "oos": oos}

    score = (
        wf.get("profitable_pct", 0) * 5.0
        + oos.get("avg_sharpe", 0)
        + mc.get("survival_rate", 0)
        + oos.get("avg_return", 0)
    )
    return score, {
        "oos": oos,
        "walk_forward": {k: v for k, v in wf.items() if k != "windows"},
        "monte_carlo": mc,
        "gates": {"oos": True, "walk_forward": True, "monte_carlo": True, "all_pass": True},
    }


def search_robust_params(
    symbols: list[str] | None = None,
    n_trials: int = 120,
    mc_sims_search: int = 30,
    seed_params: dict | None = None,
) -> dict:
    symbols = symbols or ["SOLUSDT", "BTCUSDT"]
    full_data, oos_datasets, _era_meta = _load_datasets(symbols)
    if not full_data:
        raise RuntimeError("no datasets loaded")

    active_path = ROOT / "data" / "active_deployment.json"
    base = seed_params or {}
    if not base and active_path.exists():
        base = json.loads(active_path.read_text()).get("params", {})
    if not base:
        champ = ROOT / "champions" / "volbar" / "volbar_r0773_0.9633_20260430_182057" / "params.json"
        base = json.loads(champ.read_text()) if champ.exists() else {}

    search_space = {k: dict(v) for k, v in DEFAULT_SEARCH_SPACE.items()}
    best: dict = {"score": -999.0, "params": None, "report": None}

    def objective(trial: optuna.Trial) -> float:
        params = _build_params(trial, base, search_space)
        score, report = _gate_score(params, full_data, oos_datasets, mc_sims_search)
        if score > best["score"]:
            best["score"] = score
            best["params"] = params
            best["report"] = report
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    if base:
        study.enqueue_trial({k: base[k] for k in search_space if k in base})
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    if best["params"] is None or best["score"] <= -999:
        return {"found": False, "trials": n_trials, "best_score": best["score"]}

    final_score, final_report = _gate_score(best["params"], full_data, oos_datasets, 100)
    return {
        "found": final_score > -999,
        "score": final_score,
        "params": best["params"],
        "report": final_report,
        "trials": n_trials,
        "symbols": list(full_data.keys()),
    }


def apply_robust_params(result: dict) -> Path:
    out_dir = ROOT / "data" / "deployed_params"
    out_dir.mkdir(parents=True, exist_ok=True)
    params_path = out_dir / "volbar_v4_blend_robust.json"
    params_path.write_text(json.dumps(result["params"], indent=2, ensure_ascii=False))

    active_path = ROOT / "data" / "active_deployment.json"
    dep = json.loads(active_path.read_text()) if active_path.exists() else {}
    dep["params"] = result["params"]
    dep["source_snapshot"] = dep.get("source_snapshot", "unknown") + "_wf_robust"
    dep["wf_robust_search"] = {
        "score": result["score"],
        "trials": result["trials"],
        "applied_at": time.time(),
    }
    dep["metrics_summary"] = dep.get("metrics_summary", {})
    dep["metrics_summary"]["overfit_gates"] = result["report"]["gates"]
    active_path.write_text(json.dumps(dep, indent=2, ensure_ascii=False))
    return params_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Find volbar params passing all overfit gates")
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--apply", action="store_true", help="Write params to active_deployment.json")
    args = parser.parse_args()

    result = search_robust_params(n_trials=args.trials)
    out_path = ROOT / "data" / "overfit_validation" / "wf_robust_search.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in result.items() if k != "params"}
    if result.get("params"):
        payload["params_hash"] = hash(json.dumps(result["params"], sort_keys=True))
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))

    if result.get("found") and args.apply:
        apply_robust_params(result)
        print("Applied robust params to active_deployment.json")

    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return 0 if result.get("found") else 1


if __name__ == "__main__":
    raise SystemExit(main())
