#!/usr/bin/env python3
"""Run Idea → Factor → Eval agentic loop (Phase C)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eternal-optimizer"))

from agentic_loop import run_agentic_cycle  # noqa: E402


def main() -> int:
    result = run_agentic_cycle()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    gates = result.get("evaluation", {}).get("gates") or {}
    return 0 if gates.get("all_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
