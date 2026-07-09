"""MCP tool definitions — canonical source of truth for tool schemas.

Imported by apps/api/app/routers/mcp.py. Keeping definitions here means
the research package owns its own MCP contract independent of the router.
"""

from __future__ import annotations

from typing import Any

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "list_strategies",
        "description": (
            "List all registered strategy instances. Returns name, symbols, "
            "enabled status, and config for each strategy in the registry."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "market": {
                    "type": "string",
                    "enum": ["all", "futures", "btc"],
                    "description": "Filter by market. Defaults to 'all'.",
                },
                "enabled_only": {
                    "type": "boolean",
                    "description": "If true, only return enabled strategies.",
                },
            },
        },
    },
    {
        "name": "run_backtest",
        "description": (
            "Create a research run and immediately execute a backtest for the "
            "given strategy and symbols. Returns a run_id that can be polled "
            "with get_run_status."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "strategy_name": {
                    "type": "string",
                    "description": "Name of the strategy to backtest.",
                },
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of symbols (e.g. ['rb', 'au']).",
                },
                "timeframe": {
                    "type": "string",
                    "description": "Bar timeframe (e.g. '5m', '1h', 'daily').",
                    "default": "5m",
                },
                "params": {
                    "type": "object",
                    "description": "Strategy parameter overrides.",
                    "additionalProperties": True,
                },
            },
            "required": ["strategy_name"],
        },
    },
    {
        "name": "get_run_status",
        "description": (
            "Get the current status and results of a research run. Includes "
            "metrics, validation gates, and diagnostics when available."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "The research run ID returned by run_backtest.",
                },
            },
            "required": ["run_id"],
        },
    },
    {
        "name": "get_metrics",
        "description": (
            "Get strategy performance metrics from results files and "
            "leaderboard. Returns latest backtest results and gate status."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "strategy_name": {
                    "type": "string",
                    "description": "Strategy name to look up (matches *_latest.json filename).",
                },
                "include_leaderboard": {
                    "type": "boolean",
                    "description": "If true, include current leaderboard data.",
                    "default": False,
                },
            },
        },
    },
    {
        "name": "compare_strategies",
        "description": (
            "Compare two strategies side-by-side on key metrics. "
            "Returns a structured comparison with winner per dimension."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "strategy_a": {
                    "type": "string",
                    "description": "First strategy name.",
                },
                "strategy_b": {
                    "type": "string",
                    "description": "Second strategy name.",
                },
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific metrics to compare. If omitted, compares all available.",
                },
            },
            "required": ["strategy_a", "strategy_b"],
        },
    },
]
