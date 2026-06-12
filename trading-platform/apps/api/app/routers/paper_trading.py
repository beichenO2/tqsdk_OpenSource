"""模拟实盘 API — 查看200个模拟账号的状态、排行榜、净值曲线。

端点:
- GET  /paper-trading/summary     — 总览（加密/期货分类统计）
- GET  /paper-trading/leaderboard — 排行榜
- GET  /paper-trading/accounts    — 全部账号列表
- GET  /paper-trading/accounts/{id} — 单账号详情
- GET  /paper-trading/catalog     — 策略目录（200个SOTA策略说明）
- GET  /paper-trading/equity/{id} — 单账号净值曲线
- POST /paper-trading/run         — 触发模拟回放（后台运行）
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/paper-trading", tags=["paper-trading"])

REPORT_DIR = Path(__file__).resolve().parents[4] / "output" / "paper_trading"


def _load_json(filename: str) -> Any:
    path = REPORT_DIR / filename
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Report file not found: {filename}. Run paper trading first.",
        )
    with open(path) as f:
        return json.load(f)


@router.get("/summary")
async def get_summary() -> dict[str, Any]:
    """模拟实盘总览。"""
    return _load_json("summary.json")


@router.get("/leaderboard")
async def get_leaderboard(
    market: str | None = Query(None, description="Filter by 'crypto' or 'futures'"),
    top_n: int = Query(20, ge=1, le=200),
) -> list[dict[str, Any]]:
    """排行榜 — 按收益率排序。"""
    accounts = _load_json("accounts.json")
    items = list(accounts.values())
    if market:
        items = [a for a in items if a.get("market") == market]
    items.sort(key=lambda a: a.get("total_return_pct", 0), reverse=True)
    return items[:top_n]


@router.get("/accounts")
async def get_all_accounts(
    market: str | None = Query(None),
) -> list[dict[str, Any]]:
    """全部账号列表。"""
    accounts = _load_json("accounts.json")
    items = list(accounts.values())
    if market:
        items = [a for a in items if a.get("market") == market]
    return items


@router.get("/accounts/{account_id}")
async def get_account(account_id: int) -> dict[str, Any]:
    """单账号详情。"""
    accounts = _load_json("accounts.json")
    key = str(account_id)
    if key not in accounts:
        raise HTTPException(status_code=404, detail=f"Account {account_id} not found")
    return accounts[key]


@router.get("/equity/{account_id}")
async def get_equity_curve(account_id: int) -> list[list]:
    """单账号净值曲线 — [[timestamp, equity], ...]。"""
    curves = _load_json("equity_curves.json")
    key = str(account_id)
    if key not in curves:
        raise HTTPException(status_code=404, detail=f"No equity data for account {account_id}")
    return curves[key]


@router.get("/catalog")
async def get_strategy_catalog(
    market: str | None = Query(None),
) -> list[dict[str, Any]]:
    """200个SOTA策略的完整目录。"""
    from sim_live.strategy_catalog import get_catalog
    catalog = get_catalog(market)
    return [
        {
            "account_id": e["account_id"],
            "name": e["name"],
            "class_path": e["class_path"],
            "symbols": e["symbols"],
            "description": e.get("description", ""),
        }
        for e in catalog
    ]


@router.get("/category-stats")
async def get_category_stats() -> dict[str, dict[str, Any]]:
    """按策略家族分类统计。"""
    from sim_live.strategy_catalog import FULL_CATALOG
    accounts = _load_json("accounts.json")

    family_stats: dict[str, list[float]] = {}
    for entry in FULL_CATALOG:
        aid = str(entry["account_id"])
        acct = accounts.get(aid)
        if acct is None:
            continue
        family = entry["class_path"].rsplit(".", 1)[0].rsplit(".", 1)[-1]
        if family not in family_stats:
            family_stats[family] = []
        family_stats[family].append(acct.get("total_return_pct", 0))

    result: dict[str, dict[str, Any]] = {}
    for family, returns in sorted(family_stats.items()):
        result[family] = {
            "count": len(returns),
            "avg_return": round(sum(returns) / len(returns), 2),
            "best": round(max(returns), 2),
            "worst": round(min(returns), 2),
            "profitable": sum(1 for r in returns if r > 0),
        }
    return result
