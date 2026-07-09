"""实盘交易 API — 启动/停止策略、切换 paper/live 模式、下单、风控探测。

端点:
- POST /live-trading/start          — 启动实盘/模拟交易
- POST /live-trading/stop           — 停止交易
- POST /live-trading/switch-mode    — 切换 paper/live 模式
- GET  /live-trading/status         — 查看运行状态
- GET  /live-trading/strategies     — 运行中策略列表
- POST /live-trading/strategies/{id}/toggle — 启用/禁用单个策略
- POST /live-trading/order          — 手动下单（经 RiskGate）
- POST /live-trading/risk-probe     — 风控探测（不下单，只返回闸结果）
- GET  /live-trading/risk-status    — RiskGate 状态
"""

from __future__ import annotations

import asyncio
import enum
import logging
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from risk.gate import live_trading_enabled, verify_live_confirm_token

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/live-trading", tags=["live-trading"])


class EngineMode(str, enum.Enum):
    PAPER = "paper"
    LIVE = "live"


@dataclass
class EngineState:
    scheduler: Any = None
    feed: Any = None
    task: asyncio.Task | None = None  # type: ignore[type-arg]
    mode: EngineMode = EngineMode.PAPER
    running: bool = False
    started_at: float = 0.0

    def start(self, scheduler: Any, feed: Any, task: asyncio.Task | None, mode: EngineMode) -> None:
        if self.running:
            raise RuntimeError("Engine already running")
        self.scheduler = scheduler
        self.feed = feed
        self.task = task
        self.mode = mode
        self.running = True
        self.started_at = time.time()

    def stop(self) -> None:
        self.scheduler = None
        self.feed = None
        self.task = None
        self.mode = EngineMode.PAPER
        self.running = False
        self.started_at = 0.0

    @property
    def uptime_seconds(self) -> float:
        return (time.time() - self.started_at) if self.running else 0.0


_engine = EngineState()


def _require_live_authorization(x_live_confirm: Optional[str]) -> None:
    """Double gate for live mode: env kill-switch + confirm token."""
    if not live_trading_enabled():
        raise HTTPException(
            status_code=403,
            detail=(
                "LIVE trading is disabled. Set LIVE_TRADING_ENABLED=true to unlock, "
                "then pass X-Live-Confirm header."
            ),
        )
    if not verify_live_confirm_token(x_live_confirm):
        raise HTTPException(
            status_code=403,
            detail=(
                "LIVE trading requires X-Live-Confirm header "
                "(token must match LIVE_CONFIRM_TOKEN, or I_UNDERSTAND_LIVE_RISK in dev)."
            ),
        )


class StartRequest(BaseModel):
    mode: str = Field("paper", pattern="^(paper|live)$")
    market: str = Field("crypto", pattern="^(crypto|futures|both)$")
    symbols: list[str] | None = None
    interval: str = "1m"
    strategy_count: int = Field(100, ge=1, le=200)


class SwitchModeRequest(BaseModel):
    mode: str = Field(..., pattern="^(paper|live)$")


class LiveOrderRequest(BaseModel):
    symbol: str
    exchange: str = "SHFE"
    direction: str = Field(..., pattern="^(LONG|SHORT|BUY|SELL)$")
    offset: str = Field("OPEN", pattern="^(OPEN|CLOSE|CLOSE_TODAY)$")
    price: Decimal
    volume: int = Field(..., gt=0)
    strategy_id: str = "manual"


class RiskProbeRequest(BaseModel):
    symbol: str
    exchange: str = "SHFE"
    direction: str = Field("LONG", pattern="^(LONG|SHORT|BUY|SELL)$")
    offset: str = Field("OPEN", pattern="^(OPEN|CLOSE|CLOSE_TODAY)$")
    price: Decimal = Decimal("0")
    volume: int = Field(1, gt=0)


def _to_order_request(req: LiveOrderRequest | RiskProbeRequest):
    from core.enums.direction import Direction, Offset
    from execution.order_manager import OrderRequest

    direction_map = {
        "LONG": Direction.LONG,
        "BUY": Direction.LONG,
        "SHORT": Direction.SHORT,
        "SELL": Direction.SHORT,
    }
    return OrderRequest(
        symbol=req.symbol,
        exchange=req.exchange,
        direction=direction_map[req.direction.upper()],
        offset=Offset(req.offset.upper()),
        price=req.price,
        volume=req.volume,
        strategy_id=getattr(req, "strategy_id", "probe"),
    )


@router.post("/start")
async def start_live_trading(
    req: StartRequest,
    x_live_confirm: Optional[str] = Header(default=None, alias="X-Live-Confirm"),
) -> dict[str, Any]:
    """启动实盘/模拟交易。"""
    if _engine.running:
        raise HTTPException(status_code=409, detail="Trading already running. Stop first.")

    if req.mode == "live":
        _require_live_authorization(x_live_confirm)

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

        trading_mode = TradingMode.LIVE if req.mode == "live" else TradingMode.PAPER
        scheduler = LiveScheduler(
            accounts=accounts,
            strategies=strategies,
            mode=trading_mode,
            execution_service=None,
        )

        feed = None
        task = None
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
            task = asyncio.create_task(feed.start())

        engine_mode = EngineMode.LIVE if req.mode == "live" else EngineMode.PAPER
        _engine.start(scheduler=scheduler, feed=feed, task=task, mode=engine_mode)

        return {
            "status": "started",
            "mode": req.mode,
            "market": req.market,
            "strategy_count": len(strategies),
            "symbols": req.symbols or ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"],
            "live_enabled": live_trading_enabled(),
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
    if not _engine.running:
        raise HTTPException(status_code=409, detail="Trading not running.")

    summary = None
    if _engine.feed:
        await _engine.feed.stop()
    if _engine.scheduler:
        summary = _engine.scheduler.accounts.summary()
        await _engine.scheduler.shutdown()
    if _engine.task:
        _engine.task.cancel()
        try:
            await _engine.task
        except asyncio.CancelledError:
            pass

    uptime = _engine.uptime_seconds
    _engine.stop()

    return {"status": "stopped", "uptime_seconds": round(uptime, 1), "summary": summary}


@router.post("/switch-mode")
async def switch_mode(
    req: SwitchModeRequest,
    x_live_confirm: Optional[str] = Header(default=None, alias="X-Live-Confirm"),
) -> dict[str, Any]:
    """切换 paper/live 模式。"""
    if not _engine.running or _engine.scheduler is None:
        raise HTTPException(status_code=409, detail="Trading not running.")

    from sim_live.live_scheduler import TradingMode

    new_mode = TradingMode.LIVE if req.mode == "live" else TradingMode.PAPER

    if new_mode == TradingMode.LIVE:
        _require_live_authorization(x_live_confirm)

    try:
        _engine.scheduler.switch_mode(new_mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    _engine.mode = EngineMode(req.mode)
    return {"status": "mode_switched", "mode": req.mode, "live_enabled": live_trading_enabled()}


@router.get("/status")
async def get_status() -> dict[str, Any]:
    """查看运行状态。"""
    base = {
        "running": False,
        "mode": "paper",
        "live_enabled": live_trading_enabled(),
    }
    if not _engine.running or _engine.scheduler is None:
        return base

    status = _engine.scheduler.get_status()
    status["accounts_summary"] = _engine.scheduler.accounts.summary()
    status["uptime_seconds"] = round(_engine.uptime_seconds, 1)
    status["live_enabled"] = live_trading_enabled()
    return status


@router.get("/strategies")
async def list_running_strategies() -> list[dict[str, Any]]:
    """运行中策略列表。"""
    if not _engine.running or _engine.scheduler is None:
        return []

    result = []
    for account_id, strategy in _engine.scheduler.strategies.items():
        account = _engine.scheduler.accounts.get(account_id)
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
    if not _engine.running or _engine.scheduler is None:
        raise HTTPException(status_code=409, detail="Trading not running.")

    account = _engine.scheduler.accounts.get(account_id)
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
    if not _engine.running or _engine.scheduler is None:
        return []
    return _engine.scheduler.accounts.leaderboard(market=market, top_n=top_n)


@router.post("/order")
async def place_live_order(
    req: LiveOrderRequest,
    x_live_confirm: Optional[str] = Header(default=None, alias="X-Live-Confirm"),
) -> dict[str, Any]:
    """手动下单 — 经 RiskGate 前置；live 模式需二次确认。

    paper 模式：只跑风控探测 + 记录（不触达交易所）。
    live 模式：需 LIVE_TRADING_ENABLED + X-Live-Confirm，再经 ExecutionService。
    """
    from app.deps import get_execution_service

    order_req = _to_order_request(req)

    # Always run RiskGate first
    try:
        svc = get_execution_service()
        gate = svc.risk_gate
    except Exception:
        from risk.gate import RiskGate
        gate = RiskGate()

    verdict = gate.check(order_req)
    if not verdict.allowed:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "risk_rejected",
                "limit": verdict.limit_name,
                "reason": verdict.reason,
                "gate": verdict.to_dict(),
            },
        )

    is_live = _engine.mode == EngineMode.LIVE and _engine.running
    if is_live:
        _require_live_authorization(x_live_confirm)
        try:
            svc = get_execution_service()
            from core.enums.direction import Direction, Offset

            direction_map = {
                "LONG": Direction.LONG, "BUY": Direction.LONG,
                "SHORT": Direction.SHORT, "SELL": Direction.SHORT,
            }
            order = await svc.place_order(
                strategy_id=req.strategy_id,
                symbol=req.symbol,
                exchange=req.exchange,
                direction=direction_map[req.direction.upper()],
                offset=Offset(req.offset.upper()),
                price=req.price,
                volume=req.volume,
            )
            return {
                "status": order.status.value,
                "order_id": order.order_id,
                "mode": "live",
                "gate": verdict.to_dict(),
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("live order failed")
            raise HTTPException(status_code=500, detail=str(e))

    # Paper / dry-run: gate passed, no exchange submit
    return {
        "status": "ACCEPTED_PAPER",
        "order_id": f"paper-{int(time.time() * 1000)}",
        "mode": "paper",
        "symbol": req.symbol,
        "direction": req.direction,
        "offset": req.offset,
        "price": str(req.price),
        "volume": req.volume,
        "gate": verdict.to_dict(),
        "message": "RiskGate passed; paper mode — order not sent to exchange",
    }


@router.post("/risk-probe")
async def risk_probe(req: RiskProbeRequest) -> dict[str, Any]:
    """风控探测：只跑 RiskGate，不下单。用于演示拦截。"""
    from app.deps import get_execution_service

    order_req = _to_order_request(req)
    try:
        svc = get_execution_service()
        gate = svc.risk_gate
    except Exception:
        from risk.gate import RiskGate
        gate = RiskGate()

    # Seed last price for LimitUpDown / PriceBand if missing
    if req.price > 0 and req.symbol not in gate.engine._last_prices:
        gate.engine.update_prices({req.symbol: req.price})

    verdict = gate.check(order_req)
    return {
        "probe": True,
        "allowed": verdict.allowed,
        "gate": verdict.to_dict(),
        "live_enabled": live_trading_enabled(),
    }


@router.get("/risk-status")
async def risk_status() -> dict[str, Any]:
    """RiskGate / RiskEngine 状态。"""
    from app.deps import get_execution_service

    try:
        svc = get_execution_service()
        status = svc.risk_gate.get_status()
    except Exception:
        status = {"gate": "unavailable", "limits": []}
    status["live_enabled"] = live_trading_enabled()
    status["engine_mode"] = _engine.mode.value
    status["engine_running"] = _engine.running
    return status
