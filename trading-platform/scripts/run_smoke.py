#!/usr/bin/env python3
"""Run tqsdk authorized smoke tests (Lobster adapter contract)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lobster_adapter import run_smoke_tests  # noqa: E402


def main() -> int:
    result = run_smoke_tests()
    print(json.dumps(result, indent=2))
    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
