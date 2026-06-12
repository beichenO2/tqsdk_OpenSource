#!/usr/bin/env python3
"""Phase B smoke — strategy adapter + optimizer policy."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EO = ROOT / "eternal-optimizer"


def main() -> int:
    sys.path.insert(0, str(EO))
    sys.path.insert(0, str(ROOT / "packages"))
    from strategy_adapter import (  # noqa: E402
        DEFAULT_STRATEGY,
        backtest_dispatch,
        suggest_strategy_switch,
    )

    steps = []

    nxt = suggest_strategy_switch(DEFAULT_STRATEGY, [DEFAULT_STRATEGY], "crypto")
    steps.append(("suggest_switch", nxt != DEFAULT_STRATEGY))

    policy = json.loads((EO / "optimizer_policy.json").read_text())
    steps.append(("policy_paused_1h_5min", "1h" in policy.get("paused", []) and "5min" in policy.get("paused", [])))
    steps.append(("policy_enabled_volbar", "volbar" in policy.get("enabled", [])))

    proc = subprocess.run(
        [sys.executable, str(EO / "eternal_supervisor.py"), "--dry-run"],
        cwd=str(EO),
        capture_output=True,
        text=True,
    )
    dry = proc.stdout
    steps.append(("supervisor_skips_1h", "1h:" not in dry or "eternal_optimizer.py" not in dry.split("1h:")[0] if "1h:" in dry else True))
    steps.append(("supervisor_includes_volbar", "volbar" in dry))

    print(json.dumps({k: v for k, v in steps}, indent=2))
    ok = all(v for _, v in steps)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
