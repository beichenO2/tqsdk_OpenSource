"""实盘交易 API — 启动/停止策略、切换 paper/live 模式、查看运行状态。

端点:
- POST /live-trading/start          — 启动实盘/模拟交易
- POST /live-trading/stop           — 停止交易
- POST /live-trading/switch-mode    — 切换 paper/live 模式
- GET  /live-trading/status         — 查看运行状态
- GET  /live-trading/strategies     — 运行中策略列表
- POST /live-trading/strategies/{id}/toggle — 启用/禁用单个策略
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/live-trading", tags=["live-trading"])

_engine_state: dict[str, Any] = {
    "scheduler": None,
    "feed": None,
    "task": None,
    "mode": "paper",
    "running": False,
}


class StartRequest(BaseModel):
    mode: str = Field("paper", pattern="^(paper|live)$")
    market: str = Field("crypto", pattern="^(crypto|futures|both)$")
    symbols: list[str] | None = None
    interval: str = "1m"
    strategy_count: int = Field(100, ge=1, le=200)


class SwitchModeRequest(BaseModel):
    mode: str = Field(..., pattern="^(paper|live)$")


@router.post("/start")
async def start_live_trading(req: StartRequest) -> dict[str, Any]:
    """启动实盘/模拟交易。"""
    if _engine_state["running"]:
        raise HTTPException(status_code=409, detail="Trading already running. Stop first.")

    try:
        from sim_live.live_scheduler import LiveScheduler, TradingMode
        from sim_live.account_manager import AccountManager
        from sim_live.strategy_factory import create_all_strategies
        from sim_live.strategy_catalog import get_catalog

        crypto_count = req.strategy_count if req.market in ("crypto", "both") else 0
        futures_count = req.strategy_count if req.market in ("futures", "both") else 0

        accounts = AccountManager(crypto_count=crypto_count, futures_count=futures_count)
        strategies = create_all_strategies(
            market=None if req.market == "both" else req.market
        )

        for entry in get_catalog(None if req.market == "both" else req.market):
            accounts.assign_strategy(entry["account_id"], entry["name"])

        execution_svc = None
        if req.mode == "live":
            from app.deps import get_execution_service
            try:
                execution_svc = get_execution_service()
            except Exception:
                raise HTTPException(
                    status_code=503,
                    detail="ExecutionService not available. Start in paper mode or configure broker.",
                )

        trading_mode = TradingMode.LIVE if req.mode == "live" else TradingMode.PAPER
        scheduler = LiveScheduler(
            accounts=accounts,
            strategies=strategies,
            mode=trading_mode,
            execution_service=execution_svc,
        )

        if req.market in ("crypto", "both"):
            from sim_live.realtime_feed import BinanceKlineFeed
            default_crypto = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
            symbols = req.symbols or default_crypto
            feed = BinanceKlineFeed(
                symbols=symbols,
                interval=req.interval,
                on_bar=lambda sym, bar: asyncio.create_task(
                    _on_bar_wrapper(scheduler, sym, bar)
                ),
            )
            _engine_state["feed"] = feed
            _engine_state["task"] = asyncio.create_task(feed.start())
        else:
            _engine_state["feed"] = None
            _engine_state["task"] = None

        _engine_state["scheduler"] = scheduler
        _engine_state["mode"] = req.mode
        _engine_state["running"] = True

        return {
            "status": "started",
            "mode": req.mode,
            "market": req.market,
            "strategy_count": len(strategies),
            "symbols": req.symbols or ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to start live trading")
        raise HTTPException(status_code=500, detail=str(e))


async def _on_bar_wrapper(scheduler, symbol: str, bar: dict[str, Any]) -> None:
    """Bar 回调包装器 — 驱动 scheduler.run_bar。"""
    try:
        market_data = {symbol: bar}
        await scheduler.run_bar(bar.get("timestamp", ""), market_data)
    except Exception as e:
        logger.error("run_bar error for %s: %s", symbol, e)


@router.post("/stop")
async def stop_live_trading() -> dict[str, Any]:
    """停止实盘/模拟交易。"""
    if not _engine_state["running"]:
        raise HTTPException(status_code=409, detail="Trading not running.")

    scheduler = _engine_state["scheduler"]
    feed = _engine_state["feed"]
    task = _engine_state["task"]

    if feed:
        await feed.stop()
    if scheduler:
        await scheduler.shutdown()
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    summary = None
    if scheduler:
        summary = scheduler.accounts.summary()

    _engine_state.update({
        "scheduler": None, "feed": None, "task": None,
        "mode": "paper", "running": False,
    })

    return {"status": "stopped", "summary": summary}


@router.post("/switch-mode")
async def switch_mode(req: SwitchModeRequest) -> dict[str, Any]:
    """切换 paper/live 模式。"""
    scheduler = _engine_state.get("scheduler")
    if scheduler is None:
        raise HTTPException(status_code=409, detail="Trading not running.")

    from sim_live.live_scheduler import TradingMode

    new_mode = TradingMode.LIVE if req.mode == "live" else TradingMode.PAPER

    if new_mode == TradingMode.LIVE:
        from app.deps import get_execution_service
        try:
            svc = get_execution_service()
            scheduler._execution_service = svc
        except Exception:
            raise HTTPException(
                status_code=503,
                detail="Cannot switch to LIVE: ExecutionService not available.",
            )

    try:
        scheduler.switch_mode(new_mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    _engine_state["mode"] = req.mode
    return {"status": "mode_switched", "mode": req.mode}


@router.get("/status")
async def get_status() -> dict[str, Any]:
    """查看运行状态。"""
    scheduler = _engine_state.get("scheduler")
    if scheduler is None:
        return {"running": False, "mode": "paper"}

    status = scheduler.get_status()
    status["accounts_summary"] = scheduler.accounts.summary()
    return status


@router.get("/strategies")
async def list_running_strategies() -> list[dict[str, Any]]:
    """运行中策略列表。"""
    scheduler = _engine_state.get("scheduler")
    if scheduler is None:
        return []

    result = []
    for account_id, strategy in scheduler.strategies.items():
        account = scheduler.accounts.get(account_id)
        result.append({
            "account_id": account_id,
            "name": strategy.name,
            "state": strategy.state.value,
            "symbols": strategy.config.symbols,
            "enabled": strategy.config.enabled,
            "return_pct": round(account.total_return_pct, 2) if account else 0,
            "total_trades": account.total_trades if account else 0,
        })

    return result


@router.post("/strategies/{account_id}/toggle")
async def toggle_strategy(account_id: int) -> dict[str, Any]:
    """启用/禁用单个策略。"""
    scheduler = _engine_state.get("scheduler")
    if scheduler is None:
        raise HTTPException(status_code=409, detail="Trading not running.")

    account = scheduler.accounts.get(account_id)
    if account is None:
        raise HTTPException(status_code=404, detail=f"Account {account_id} not found")

    account.is_active = not account.is_active
    return {
        "account_id": account_id,
        "is_active": account.is_active,
        "strategy": account.strategy_name,
    }


@router.get("/leaderboard")
async def get_leaderboard(
    market: str | None = None,
    top_n: int = 20,
) -> list[dict[str, Any]]:
    """实时排行榜。"""
    scheduler = _engine_state.get("scheduler")
    if scheduler is None:
        return []
    return scheduler.accounts.leaderboard(market=market, top_n=top_n)
