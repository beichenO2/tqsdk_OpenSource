#!/usr/bin/env python3
"""Overfit validation gate for deployed volbar champion (OOS + WF + MC)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(ROOT / "eternal-optimizer"))
sys.path.insert(0, str(ROOT / "scripts"))

from overfit_data import load_volbar_gate_datasets  # noqa: E402
from eternal_optimizer_volbar import (  # noqa: E402
    monte_carlo_robustness,
    oos_gate_check,
    walk_forward_validate,
)


def main() -> int:
    active_path = ROOT / "data" / "active_deployment.json"
    if not active_path.exists():
        print("FAIL: run deploy_volbar_champion.py first")
        return 1

    deployment = json.loads(active_path.read_text())
    params = deployment["params"]
    symbols = deployment.get("symbols_focus", ["SOLUSDT", "BTCUSDT"])

    full_data, oos_datasets, era_meta = load_volbar_gate_datasets(symbols)
    if not full_data:
        print("FAIL: no volbar datasets loaded")
        return 1

    oos = oos_gate_check(oos_datasets, params)
    wf = walk_forward_validate(full_data, params)
    mc = monte_carlo_robustness(full_data, params, n_simulations=100)

    wf_pass = wf.get("valid") and wf.get("profitable_pct", 0) >= 0.5
    mc_pass = mc.get("survival_rate", 0) >= 0.80
    all_pass = bool(oos.get("passed")) and wf_pass and mc_pass

    report = {
        "snapshot": deployment.get("source_snapshot"),
        "symbols": list(full_data.keys()),
        "validation_era": era_meta,
        "oos_gate": oos,
        "walk_forward": {k: v for k, v in wf.items() if k != "windows"},
        "monte_carlo": mc,
        "gates": {
            "oos": bool(oos.get("passed")),
            "walk_forward": wf_pass,
            "monte_carlo": mc_pass,
            "all_pass": all_pass,
        },
        "thresholds": {
            "oos": "avg_sharpe>0.5, avg_return>0, avg_dd<0.40, no liq",
            "walk_forward": "profitable_pct>=0.50 (common-era aligned multi-asset windows)",
            "monte_carlo": "survival_rate>=0.80",
        },
    }

    out_dir = ROOT / "data" / "overfit_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "volbar_champion_gates.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if not all_pass:
        print("FAIL: overfit validation gates not all passed")
        print("HINT: run scripts/run_wf_robust_search.py --apply")
        return 1
    print("PASS: OOS + Walk-Forward + Monte Carlo gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
