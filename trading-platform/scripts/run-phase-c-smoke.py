#!/usr/bin/env python3
"""Phase C smoke — agentic Idea→Factor→Eval loop."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EO = ROOT / "eternal-optimizer"


def main() -> int:
    sys.path.insert(0, str(EO))
    from agentic_loop import _eval_gates, _factor_from_deployment, _idea_from_digist

    steps = []

    idea = _idea_from_digist(["agentic trading"])
    steps.append(("idea_hypothesis", bool(idea.get("hypothesis"))))

    factor = _factor_from_deployment()
    steps.append(("factor_strategy", bool(factor.get("strategy"))))
    steps.append(("factor_symbols", len(factor.get("symbols", [])) >= 1))

    gates_path = ROOT / "data" / "overfit_validation" / "volbar_champion_gates.json"
    steps.append(("overfit_report_exists", gates_path.exists()))
    if gates_path.exists():
        gates = json.loads(gates_path.read_text()).get("gates", {})
        steps.append(("oos_gate_pass", bool(gates.get("oos"))))
        steps.append(("wf_gate_pass", bool(gates.get("walk_forward"))))
        steps.append(("all_gates_pass", bool(gates.get("all_pass"))))

    print(json.dumps({k: v for k, v in steps}, indent=2))
    ok = all(v for _, v in steps)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
