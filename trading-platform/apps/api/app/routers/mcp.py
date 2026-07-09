"""MCP (Model Context Protocol) server router.

Exposes trading-platform capabilities as MCP tools so that LLM agents
can discover and invoke them via the standard MCP JSON-RPC protocol.

Reference: contracts.md §7 — MCP 暴露层
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "packages"))

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mcp", tags=["mcp"])

# ---------------------------------------------------------------------------
# MCP-specific auth — separate from the global auth middleware.
# An MCP client (e.g. Claude Desktop, Cursor) authenticates with
# X-MCP-Key header. Set MCP_API_KEY env var to enable; if unset,
# MCP endpoints are open (dev mode).
# ---------------------------------------------------------------------------

_MCP_KEY: str | None = os.getenv("MCP_API_KEY")


async def _verify_mcp_key(request: Request) -> None:
    if _MCP_KEY is None:
        return
    provided = request.headers.get("x-mcp-key", "")
    if not provided:
        raise HTTPException(401, "MCP authentication required (X-MCP-Key header)")
    if not hmac.compare_digest(provided, _MCP_KEY):
        raise HTTPException(403, "Invalid MCP API key")

# ---------------------------------------------------------------------------
# Tool registry — canonical definitions live in packages/research/mcp_tools.py
# ---------------------------------------------------------------------------

try:
    from research.mcp_tools import TOOL_DEFINITIONS as _TOOL_DEFINITIONS
except ImportError:
    from importlib import import_module as _im
    _TOOL_DEFINITIONS = _im("research.mcp_tools").TOOL_DEFINITIONS


# ---------------------------------------------------------------------------
# MCP protocol endpoints
# ---------------------------------------------------------------------------

@router.get("/tools", dependencies=[Depends(_verify_mcp_key)])
async def list_tools():
    """Return all available MCP tool definitions (discovery endpoint)."""
    return {"tools": _TOOL_DEFINITIONS}


class ToolCallRequest(BaseModel):
    name: str
    arguments: dict[str, Any] = {}


@router.post("/tools/call", dependencies=[Depends(_verify_mcp_key)])
async def call_tool(req: ToolCallRequest):
    """Invoke an MCP tool by name with the given arguments."""
    handler = _TOOL_HANDLERS.get(req.name)
    if handler is None:
        raise HTTPException(404, f"Unknown tool: {req.name}")
    try:
        result = await handler(req.arguments)
        return {"content": [{"type": "text", "text": json.dumps(result, default=str)}]}
    except Exception as e:
        logger.exception("MCP tool %s failed", req.name)
        return JSONResponse(
            status_code=500,
            content={
                "content": [{"type": "text", "text": f"Error: {type(e).__name__}: {e}"}],
                "isError": True,
            },
        )


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

async def _handle_list_strategies(args: dict[str, Any]) -> Any:
    market_filter = args.get("market", "all")
    enabled_only = args.get("enabled_only", False)

    strategies: list[dict] = []

    try:
        from strategy.registry import StrategyRegistry
        for cfg in StrategyRegistry.list_instances():
            entry = cfg.model_dump(mode="json")
            if enabled_only and not entry.get("enabled", True):
                continue
            strategies.append(entry)
    except (ImportError, Exception) as exc:
        logger.debug("StrategyRegistry unavailable (%s), falling back to filesystem scan", exc)

    if not strategies:
        strategies = _scan_strategy_files(market_filter)
    elif market_filter != "all":
        strategies = [s for s in strategies if _matches_market(s, market_filter)]

    return {
        "count": len(strategies),
        "market_filter": market_filter,
        "strategies": strategies,
    }


def _scan_strategy_files(market_filter: str) -> list[dict]:
    """Fallback: scan packages/strategy/ for .py files when no registry is loaded."""
    base = REPO_ROOT / "packages" / "strategy"
    results = []

    scan_dirs: list[tuple[str, Path]] = []
    if market_filter in ("all", "futures"):
        scan_dirs.append(("futures", base / "futures"))
    if market_filter in ("all", "btc"):
        scan_dirs.append(("btc", base / "btc"))

    for market, directory in scan_dirs:
        if not directory.is_dir():
            continue
        for f in sorted(directory.glob("*.py")):
            if f.name.startswith("_") or f.name == "__init__.py":
                continue
            name = f.stem
            meta_path = f.with_suffix("").with_name(f"{name}_meta.json")
            meta = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                except Exception:
                    pass
            results.append({
                "name": name,
                "market": market,
                "file": str(f.relative_to(REPO_ROOT)),
                "has_meta": meta_path.exists(),
                **({k: meta[k] for k in ("description", "symbols", "timeframe") if k in meta}),
            })

    return results


def _matches_market(strategy: dict, market: str) -> bool:
    name = strategy.get("name", "")
    syms = strategy.get("symbols", [])
    if market == "futures":
        return not any(s.upper() in ("BTCUSDT", "ETHUSDT") for s in syms)
    if market == "btc":
        return any(s.upper() in ("BTCUSDT", "ETHUSDT") for s in syms) or "btc" in name.lower()
    return True


async def _handle_run_backtest(args: dict[str, Any]) -> Any:
    strategy_name = args.get("strategy_name")
    if not strategy_name:
        return {"error": "strategy_name is required"}

    symbols = args.get("symbols", ["rb"])
    timeframe = args.get("timeframe", "5m")
    params = args.get("params", {})

    try:
        from experiment.research_run import ResearchRun, RunStore

        research_dir = REPO_ROOT / "output" / "research"
        store = RunStore(research_dir)

        run = ResearchRun(
            prompt=f"MCP backtest: {strategy_name} on {symbols}",
            strategy_name=strategy_name,
            symbols=symbols,
            timeframe=timeframe,
            config=params,
            tags=["mcp-initiated"],
        )
        store.save(run)

        asyncio.get_event_loop().create_task(_execute_backtest_bg(run.run_id, store))

        return {
            "run_id": run.run_id,
            "status": "running",
            "strategy_name": strategy_name,
            "symbols": symbols,
            "timeframe": timeframe,
        }
    except ImportError as exc:
        return {"error": f"Research module unavailable: {exc}"}


async def _execute_backtest_bg(run_id: str, store: Any) -> None:
    """Background backtest execution — mirrors research.py _execute_backtest."""
    import traceback

    run = store.load(run_id)
    if not run:
        return
    try:
        from experiment.research_run import RunStatus
        run.status = RunStatus.RUNNING
        store.save(run)

        from backtest.engine import BacktestEngine
        engine = BacktestEngine()
        symbols = run.symbols or ["rb"]
        results = {}
        for sym in symbols:
            result = engine.run(
                strategy_name=run.strategy_name,
                symbol=sym,
                timeframe=run.timeframe,
                params=run.config,
            )
            results[sym] = result

        agg = {}
        sharpes, rets, dds = [], [], []
        for r in results.values():
            if isinstance(r, dict):
                if "sharpe" in r:
                    sharpes.append(r["sharpe"])
                if "total_return" in r:
                    rets.append(r["total_return"])
                if "max_dd" in r:
                    dds.append(r["max_dd"])
        if sharpes:
            agg["avg_sharpe"] = sum(sharpes) / len(sharpes)
        if rets:
            agg["avg_return"] = sum(rets) / len(rets)
        if dds:
            agg["avg_max_dd"] = sum(dds) / len(dds)
        agg["symbols_tested"] = len(results)

        run.metrics = agg
        run.status = RunStatus.COMPLETED

    except ImportError as exc:
        from experiment.research_run import RunStatus
        run.add_diagnostic("runtime", "import_error", str(exc), severity="error")
        run.status = RunStatus.FAILED
    except Exception as exc:
        from experiment.research_run import RunStatus
        run.add_diagnostic("runtime", "execution_error",
                           f"{type(exc).__name__}: {exc}", severity="error")
        run.status = RunStatus.FAILED
    store.save(run)


async def _handle_get_run_status(args: dict[str, Any]) -> Any:
    run_id = args.get("run_id")
    if not run_id:
        return {"error": "run_id is required"}

    try:
        from experiment.research_run import RunStore

        research_dir = REPO_ROOT / "output" / "research"
        store = RunStore(research_dir)
        run = store.load(run_id)
        if not run:
            return {"error": f"Run {run_id} not found"}
        return run.to_dict()
    except ImportError as exc:
        return {"error": f"Research module unavailable: {exc}"}


async def _handle_get_metrics(args: dict[str, Any]) -> Any:
    strategy_name = args.get("strategy_name")
    include_leaderboard = args.get("include_leaderboard", False)

    result: dict[str, Any] = {}

    if strategy_name:
        latest_path = REPO_ROOT / "results" / f"{strategy_name}_latest.json"
        if latest_path.exists():
            try:
                result["latest"] = json.loads(latest_path.read_text())
            except Exception as exc:
                result["latest_error"] = str(exc)
        else:
            candidates = sorted(REPO_ROOT.glob(f"results/{strategy_name}*.json"))
            if candidates:
                try:
                    result["latest"] = json.loads(candidates[-1].read_text())
                    result["latest_file"] = candidates[-1].name
                except Exception as exc:
                    result["latest_error"] = str(exc)
            else:
                result["latest"] = None
                result["note"] = f"No results file for '{strategy_name}'"
    else:
        all_latest = sorted(REPO_ROOT.glob("results/*_latest.json"))
        result["available"] = [f.stem.replace("_latest", "") for f in all_latest]

    if include_leaderboard:
        lb_path = REPO_ROOT / ".coordination" / "leaderboard.md"
        if not lb_path.exists():
            lb_path = REPO_ROOT.parent.parent / "tqsdk" / ".coordination" / "leaderboard.md"
        if lb_path.exists():
            result["leaderboard_raw"] = lb_path.read_text()
        else:
            result["leaderboard"] = "leaderboard.md not found"

    return result


async def _handle_compare_strategies(args: dict[str, Any]) -> Any:
    name_a = args.get("strategy_a", "")
    name_b = args.get("strategy_b", "")
    requested_metrics = args.get("metrics")

    if not name_a or not name_b:
        return {"error": "Both strategy_a and strategy_b are required"}

    def _load_metrics(name: str) -> dict | None:
        path = REPO_ROOT / "results" / f"{name}_latest.json"
        if path.exists():
            return json.loads(path.read_text())
        candidates = sorted(REPO_ROOT.glob(f"results/{name}*.json"))
        if candidates:
            return json.loads(candidates[-1].read_text())
        return None

    data_a = _load_metrics(name_a)
    data_b = _load_metrics(name_b)

    if data_a is None and data_b is None:
        return {"error": f"No results found for either '{name_a}' or '{name_b}'"}

    _COMPARE_KEYS = ["sharpe", "total_return", "max_dd", "win_rate", "profit_factor", "total_trades",
                     "avg_sharpe", "avg_return", "avg_max_dd"]
    if requested_metrics:
        compare_keys = requested_metrics
    else:
        compare_keys = _COMPARE_KEYS

    def _extract(data: dict | None, key: str) -> float | None:
        if data is None:
            return None
        if key in data:
            return data[key]
        for section in ("train", "oos", "metrics"):
            if isinstance(data.get(section), dict) and key in data[section]:
                return data[section][key]
        return None

    comparison = []
    for key in compare_keys:
        val_a = _extract(data_a, key)
        val_b = _extract(data_b, key)
        winner = None
        if val_a is not None and val_b is not None:
            higher_better = key not in ("max_dd", "avg_max_dd")
            if higher_better:
                winner = name_a if val_a > val_b else name_b
            else:
                winner = name_a if val_a < val_b else name_b
        comparison.append({
            "metric": key,
            name_a: val_a,
            name_b: val_b,
            "winner": winner,
        })

    wins_a = sum(1 for c in comparison if c["winner"] == name_a)
    wins_b = sum(1 for c in comparison if c["winner"] == name_b)

    return {
        "strategy_a": name_a,
        "strategy_b": name_b,
        "comparison": comparison,
        "summary": {
            "wins": {name_a: wins_a, name_b: wins_b},
            "overall_winner": name_a if wins_a > wins_b else (name_b if wins_b > wins_a else "tie"),
        },
    }


# ---------------------------------------------------------------------------
# Handler dispatch table
# ---------------------------------------------------------------------------

_TOOL_HANDLERS = {
    "list_strategies": _handle_list_strategies,
    "run_backtest": _handle_run_backtest,
    "get_run_status": _handle_get_run_status,
    "get_metrics": _handle_get_metrics,
    "compare_strategies": _handle_compare_strategies,
}
