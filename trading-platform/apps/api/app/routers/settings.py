"""Research workbench settings — local-only access.

Reference: Vibe-Trading /settings/* pattern — only accessible from localhost
to prevent remote configuration changes on a local research platform.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[4]
SETTINGS_PATH = REPO_ROOT / "output" / "research" / "settings.json"

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/settings", tags=["settings"])

_LOCAL_ADDRS = {"127.0.0.1", "::1", "localhost", "0.0.0.0"}


def _require_local(request: Request) -> None:
    """Reject non-localhost requests to protect settings endpoints."""
    client_host = request.client.host if request.client else None
    if client_host not in _LOCAL_ADDRS:
        raise HTTPException(
            403,
            f"Settings are local-only. Your address: {client_host}",
        )


def _load_settings() -> dict[str, Any]:
    if SETTINGS_PATH.exists():
        try:
            return json.loads(SETTINGS_PATH.read_text())
        except Exception:
            logger.warning("Corrupt settings file, returning defaults")
    return _default_settings()


def _save_settings(data: dict[str, Any]) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def _default_settings() -> dict[str, Any]:
    return {
        "research": {
            "default_timeframe": "5m",
            "default_symbols": ["rb", "au", "cu"],
            "max_concurrent_runs": 3,
            "auto_validate": True,
            "validation_gates": ["OOS", "WF", "MC-CI", "X-Asset"],
        },
        "mcp": {
            "enabled": True,
            "tools_exposed": ["list_strategies", "run_backtest", "get_run_status"],
        },
        "upload": {
            "max_size_mb": 50,
            "allowed_extensions": [".csv", ".json", ".parquet", ".py", ".yaml", ".yml", ".txt", ".md"],
        },
        "notifications": {
            "sse_heartbeat_interval_s": 30,
        },
    }


class SettingsPatch(BaseModel):
    path: str
    value: Any


@router.get("")
async def get_all_settings(request: Request):
    """Return all current settings. Local-only."""
    _require_local(request)
    return _load_settings()


@router.get("/{section}")
async def get_section(section: str, request: Request):
    """Return settings for a specific section. Local-only."""
    _require_local(request)
    settings = _load_settings()
    if section not in settings:
        raise HTTPException(404, f"Unknown section: {section}. Available: {list(settings.keys())}")
    return {section: settings[section]}


@router.put("/{section}")
async def replace_section(section: str, request: Request):
    """Replace an entire settings section. Local-only."""
    _require_local(request)
    body = await request.json()
    settings = _load_settings()
    settings[section] = body
    _save_settings(settings)
    logger.info("Settings section '%s' replaced", section)
    return {"ok": True, "section": section}


@router.patch("")
async def patch_setting(req: SettingsPatch, request: Request):
    """Update a single setting by dotted path (e.g. 'research.default_timeframe'). Local-only."""
    _require_local(request)
    settings = _load_settings()

    parts = req.path.split(".")
    target = settings
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            raise HTTPException(400, f"Invalid path: {req.path}")
        target = target[part]

    old_value = target.get(parts[-1])
    target[parts[-1]] = req.value
    _save_settings(settings)

    logger.info("Setting '%s' changed: %s → %s", req.path, old_value, req.value)
    return {"ok": True, "path": req.path, "old_value": old_value, "new_value": req.value}


@router.post("/reset")
async def reset_settings(request: Request):
    """Reset all settings to defaults. Local-only."""
    _require_local(request)
    defaults = _default_settings()
    _save_settings(defaults)
    logger.info("Settings reset to defaults")
    return {"ok": True, "settings": defaults}
