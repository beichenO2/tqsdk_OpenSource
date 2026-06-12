"""tqsdk Lobster SDK Adapter — event emission + status/health/test for PolarClaw Pilot Runtime.

Writes structured events to lobster-events.jsonl (consumed by PolarClaw Pilot).
Exposes status/health/test adapters for Pilot Runtime health checks.

This adapter uses the PolarClaw Project SDK schema for events. If the SDK package
is not yet installed, it falls back to direct JSONL writing with the same schema.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("lobster-adapter")

PROJ_ROOT = Path(__file__).resolve().parent.parent
LOBSTER_DIR = PROJ_ROOT / "lobster"
TARGETS_DIR = LOBSTER_DIR / "targets"
EVENTS_FILE = LOBSTER_DIR / "lobster-events.jsonl"


def _ensure_dirs() -> None:
    LOBSTER_DIR.mkdir(exist_ok=True)
    TARGETS_DIR.mkdir(exist_ok=True)


def emit_event(
    event_type: str,
    source: str,
    detail: dict[str, Any],
    severity: str = "error",
) -> None:
    """Write a structured event to lobster-events.jsonl.

    Event types: backtest_error, submit_failure, optimizer_crash,
                 contract_test_red, build_failure, gate_rejection.
    Severity: info, warning, error, critical.
    """
    _ensure_dirs()
    event = {
        "timestamp": time.time(),
        "iso_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "event_type": event_type,
        "source": source,
        "severity": severity,
        "detail": detail,
        "project": "tqsdk",
    }
    try:
        with open(EVENTS_FILE, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")
        logger.info("[lobster] event emitted: %s from %s", event_type, source)
    except Exception as e:
        logger.warning("[lobster] failed to write event: %s", e)


def get_status() -> dict[str, Any]:
    """Return tqsdk project status for Pilot Runtime health checks."""
    status: dict[str, Any] = {
        "project": "tqsdk",
        "timestamp": time.time(),
        "components": {},
    }

    optimizer_status = _read_optimizer_status()
    if optimizer_status:
        status["components"]["eternal_optimizer"] = optimizer_status

    api_healthy = _check_api_health()
    status["components"]["api"] = {"healthy": api_healthy}

    recent_events = _read_recent_events(limit=10)
    status["recent_errors"] = [e for e in recent_events if e.get("severity") in ("error", "critical")]
    status["error_count_24h"] = _count_events_since(time.time() - 86400)

    return status


def get_health() -> dict[str, Any]:
    """Quick health check — returns pass/fail with component breakdown."""
    status = get_status()
    components = status.get("components", {})

    checks = {}
    overall = True

    if "eternal_optimizer" in components:
        opt = components["eternal_optimizer"]
        opt_ok = not opt.get("all_stopped", False)
        checks["optimizer"] = {"healthy": opt_ok, "detail": opt}
        if not opt_ok:
            overall = False

    api_ok = components.get("api", {}).get("healthy", False)
    checks["api"] = {"healthy": api_ok}

    error_rate = status.get("error_count_24h", 0)
    checks["error_rate"] = {"healthy": error_rate < 50, "count_24h": error_rate}
    if error_rate >= 50:
        overall = False

    return {
        "project": "tqsdk",
        "healthy": overall,
        "checks": checks,
        "timestamp": time.time(),
    }


def run_smoke_tests() -> dict[str, Any]:
    """Run authorized smoke/contract tests and return results."""
    results: list[dict] = []

    results.append(_test_data_availability())
    results.append(_test_backtest_import())
    results.append(_test_optimizer_modules())
    results.append(_test_api_import())
    results.append(_test_rl_import())

    passed = sum(1 for r in results if r["passed"])
    return {
        "project": "tqsdk",
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "tests": results,
        "timestamp": time.time(),
    }


def _read_optimizer_status() -> dict | None:
    status_path = PROJ_ROOT / "trading-platform" / "eternal-optimizer" / "STATUS.json"
    try:
        if status_path.exists():
            return json.loads(status_path.read_text())
    except Exception:
        pass
    return None


def _check_api_health() -> bool:
    try:
        from urllib.request import urlopen
        api_port = os.environ.get("TQSDK_API_PORT", "8000")
        with urlopen(f"http://127.0.0.1:{api_port}/health", timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def _read_recent_events(limit: int = 10) -> list[dict]:
    if not EVENTS_FILE.exists():
        return []
    try:
        lines = EVENTS_FILE.read_text().strip().split("\n")
        events = []
        for line in lines[-limit:]:
            if line.strip():
                events.append(json.loads(line))
        return events
    except Exception:
        return []


def _count_events_since(since_ts: float) -> int:
    if not EVENTS_FILE.exists():
        return 0
    count = 0
    try:
        with open(EVENTS_FILE) as f:
            for line in f:
                if line.strip():
                    evt = json.loads(line)
                    if evt.get("timestamp", 0) >= since_ts:
                        count += 1
    except Exception:
        pass
    return count


def _test_data_availability() -> dict:
    """Check that parquet data files exist."""
    data_dir = PROJ_ROOT / "trading-platform" / "data"
    try:
        parquets = list(data_dir.rglob("*.parquet")) if data_dir.exists() else []
        return {"name": "data_availability", "passed": len(parquets) > 0,
                "detail": f"{len(parquets)} parquet files found"}
    except Exception as e:
        return {"name": "data_availability", "passed": False, "detail": str(e)}


def _test_backtest_import() -> dict:
    """Check that backtest engine can be imported."""
    try:
        import importlib
        importlib.import_module("packages.backtest")
        return {"name": "backtest_import", "passed": True, "detail": "import ok"}
    except Exception as e:
        return {"name": "backtest_import", "passed": False, "detail": str(e)}


def _test_optimizer_modules() -> dict:
    """Check that optimizer core modules can be imported."""
    try:
        import importlib
        import sys

        opt_dir = PROJ_ROOT / "trading-platform" / "eternal-optimizer"
        opt_path = str(opt_dir)
        if opt_path not in sys.path:
            sys.path.insert(0, opt_path)
        for mod in ("stop_conditions", "gate_report", "adaptive_search"):
            importlib.import_module(mod)
        return {"name": "optimizer_modules", "passed": True, "detail": "all imports ok"}
    except Exception as e:
        return {"name": "optimizer_modules", "passed": False, "detail": str(e)}


def _test_api_import() -> dict:
    """Check that FastAPI app and core routers can be imported."""
    try:
        import importlib
        import sys

        api_root = PROJ_ROOT / "trading-platform"
        for path in reversed((
            api_root,
            api_root / "apps" / "api",
            api_root / "packages" / "core",
            api_root / "packages" / "backtest",
            api_root / "packages" / "broker_tqsdk",
            api_root / "packages" / "broker_crypto" / "src",
            api_root / "packages" / "risk",
            api_root / "packages" / "sim_live",
            api_root / "packages" / "security" / "src",
            api_root / "packages",
        )):
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)

        for mod in ("app.routers.health", "app.routers.research", "app.routers.live_trading"):
            importlib.import_module(mod)
        return {"name": "api_import", "passed": True, "detail": "health/research/live_trading routers ok"}
    except Exception as e:
        return {"name": "api_import", "passed": False, "detail": str(e)}


def _test_rl_import() -> dict:
    """Check that RL PPO + MambaFormer modules can be imported."""
    try:
        import importlib
        import sys

        tp_root = PROJ_ROOT / "trading-platform"
        tp_path = str(tp_root)
        if tp_path not in sys.path:
            sys.path.insert(0, tp_path)

        importlib.import_module("packages.rl.mambaformer_extractor")
        importlib.import_module("packages.rl.trading_env")
        return {"name": "rl_import", "passed": True, "detail": "MambaFormer + trading_env import ok"}
    except Exception as e:
        return {"name": "rl_import", "passed": False, "detail": str(e)}
