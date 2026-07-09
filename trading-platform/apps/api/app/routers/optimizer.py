"""优化器 / 冠军 / gate 报告只读 API。

端点:
- GET /optimizer/champions          — 各变体冠军榜
- GET /optimizer/gates              — results/*_gate.json 摘要
- GET /optimizer/gates/{name}       — 单个 gate 详情
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/optimizer", tags=["optimizer"])

REPO = Path(__file__).resolve().parents[4]  # trading-platform/
CHAMPIONS_DIR = REPO / "champions"
RESULTS_DIR = REPO / "results"


@router.get("/champions")
async def list_champions(
    variant: str | None = Query(None),
    top_n: int = Query(20, ge=1, le=100),
) -> dict[str, Any]:
    """读取 champions/{variant}/leaderboard.json。"""
    if not CHAMPIONS_DIR.exists():
        return {"variants": [], "entries": []}

    variants = sorted(p.name for p in CHAMPIONS_DIR.iterdir() if p.is_dir())
    if variant:
        variants = [v for v in variants if v == variant]
        if not variants:
            raise HTTPException(status_code=404, detail=f"Variant not found: {variant}")

    entries: list[dict[str, Any]] = []
    for v in variants:
        lb = CHAMPIONS_DIR / v / "leaderboard.json"
        if not lb.exists():
            continue
        try:
            rows = json.loads(lb.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(rows, list):
            continue
        for row in rows[:top_n]:
            entries.append({
                "variant": v,
                "snapshot": row.get("snapshot"),
                "score": row.get("score"),
                "round": row.get("round"),
                "saved_at": row.get("saved_at"),
            })

    entries.sort(key=lambda x: float(x.get("score") or 0), reverse=True)
    return {"variants": variants, "entries": entries[:top_n], "total": len(entries)}


@router.get("/gates")
async def list_gates() -> dict[str, Any]:
    """扫描 results/*_gate.json 与 results/gate_rerun/*_gate.json。"""
    if not RESULTS_DIR.exists():
        return {"gates": []}

    files = list(RESULTS_DIR.glob("*_gate.json")) + list((RESULTS_DIR / "gate_rerun").glob("*_gate.json"))
    gates: list[dict[str, Any]] = []
    for f in sorted(files):
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        gate = data.get("gate") or {}
        checks = gate.get("checks") or []
        passed = sum(1 for c in checks if c.get("pass"))
        gates.append({
            "name": data.get("strategy") or f.stem,
            "file": str(f.relative_to(REPO)),
            "outcome": gate.get("outcome", "unknown"),
            "passed": passed,
            "total": len(checks),
            "primary_symbol": data.get("primary_symbol"),
            "symbols": data.get("symbols") or [],
            "generated_at": (data.get("_meta") or {}).get("generated_at"),
        })
    return {"gates": gates, "count": len(gates)}


@router.get("/gates/{name}")
async def get_gate(name: str) -> dict[str, Any]:
    """按策略名取 gate 详情（优先 results/{name}_gate.json）。"""
    candidates = [
        RESULTS_DIR / f"{name}_gate.json",
        RESULTS_DIR / "gate_rerun" / f"{name}_gate.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=500, detail=f"Invalid JSON: {e}") from e
    raise HTTPException(status_code=404, detail=f"Gate report not found: {name}")
