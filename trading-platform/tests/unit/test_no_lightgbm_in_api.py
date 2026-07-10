"""CI guard: API / trading process must never import LightGBM (OpenMP isolation)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]


def _api_pythonpath() -> str:
    return os.pathsep.join(
        [
            str(_repo),
            str(_repo / "apps" / "api"),
            str(_repo / "packages" / "core"),
            str(_repo / "packages" / "backtest"),
            str(_repo / "packages" / "broker_tqsdk"),
            str(_repo / "packages" / "broker_crypto" / "src"),
            str(_repo / "packages" / "risk"),
            str(_repo / "packages" / "factor"),
            str(_repo / "packages" / "features"),
            str(_repo / "packages" / "sim_live"),
            str(_repo / "packages" / "security" / "src"),
            str(_repo / "packages"),
        ]
    )


def test_api_app_import_does_not_load_lightgbm() -> None:
    """Importing the FastAPI app in a clean subprocess must not pull in lightgbm."""
    env = os.environ.copy()
    env["PYTHONPATH"] = _api_pythonpath()
    script = """
import sys
from app.main import app  # noqa: F401 — side-effect import check
assert "lightgbm" not in sys.modules, (
    "lightgbm must not be loaded in API process; loaded modules: "
    + ", ".join(sorted(m for m in sys.modules if "light" in m.lower() or "lgb" in m.lower()))
)
print("OK_NO_LIGHTGBM_IN_API")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(_repo),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"stderr={result.stderr!r}\nstdout={result.stdout!r}"
    assert "OK_NO_LIGHTGBM_IN_API" in result.stdout
