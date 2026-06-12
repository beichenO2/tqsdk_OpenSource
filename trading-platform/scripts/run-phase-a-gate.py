#!/usr/bin/env python3
"""Phase A gate: deploy champion → paper probe → smoke tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out


def main() -> int:
    steps: list[dict] = []

    code, out = _run([sys.executable, "scripts/deploy_volbar_champion.py"])
    steps.append({"name": "deploy_volbar_champion", "passed": code == 0, "detail": out.strip()[-500:]})
    if code != 0:
        print(json.dumps({"phase": "A", "passed": False, "steps": steps}, indent=2))
        return 1

    code, out = _run([sys.executable, "scripts/run_volbar_paper_probe.py"])
    steps.append({"name": "volbar_paper_probe", "passed": code == 0, "detail": out.strip()[-800:]})
    if code != 0:
        print(json.dumps({"phase": "A", "passed": False, "steps": steps}, indent=2))
        return 1

    code, out = _run([sys.executable, "scripts/run_overfit_validation.py"])
    steps.append({"name": "overfit_validation", "passed": code == 0, "detail": out.strip()[-800:]})
    if code != 0:
        print(json.dumps({"phase": "A", "passed": False, "steps": steps}, indent=2))
        return 1

    code, out = _run([sys.executable, "scripts/run_smoke.py"])
    steps.append({"name": "smoke", "passed": code == 0, "detail": out.strip()[-500:]})

    passed = all(s["passed"] for s in steps)
    print(json.dumps({"phase": "A", "passed": passed, "steps": [s["name"] + ":" + ("PASS" if s["passed"] else "FAIL") for s in steps]}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
