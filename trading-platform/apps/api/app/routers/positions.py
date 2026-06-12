"""持仓查询路由."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.deps import get_execution_service
from execution.service import ExecutionService

router = APIRouter(prefix="/positions", tags=["positions"])


class PositionResponse(BaseModel):
    symbol: str
    exchange: str
    direction: str
    volume: int
    available: int
    avg_price: str
    margin: str
    float_pnl: str
    close_pnl: str


@router.get("", response_model=list[PositionResponse])
async def list_positions(
    svc: ExecutionService = Depends(get_execution_service),
) -> list[PositionResponse]:
    """查询当前所有持仓."""
    positions = svc.get_positions()
    return [
        PositionResponse(
            symbol=p.symbol,
            exchange=p.exchange,
            direction=p.direction.value,
            volume=p.volume,
            available=p.available,
            avg_price=str(p.avg_price),
            margin=str(p.margin),
            float_pnl=str(p.float_pnl),
            close_pnl=str(p.close_pnl),
        )
        for p in positions
    ]


@router.get("/{symbol}")
async def get_position(
    symbol: str,
    svc: ExecutionService = Depends(get_execution_service),
) -> dict:
    """查询指定合约持仓（多空）."""
    from core.enums.direction import Direction

    long = svc.get_position(symbol, Direction.LONG)
    short = svc.get_position(symbol, Direction.SHORT)

    def pos_dict(p):
        if p is None:
            return {"volume": 0, "available": 0, "avg_price": "0", "float_pnl": "0"}
        return {
            "volume": p.volume,
            "available": p.available,
            "avg_price": str(p.avg_price),
            "float_pnl": str(p.float_pnl),
        }

    return {"symbol": symbol, "long": pos_dict(long), "short": pos_dict(short)}


@router.get("/risk/status")
async def risk_status(
    svc: ExecutionService = Depends(get_execution_service),
) -> dict:
    """查询风控引擎状态."""
    return svc.get_risk_status()


@router.post("/close-all")
async def close_all_positions(
    svc: ExecutionService = Depends(get_execution_service),
) -> dict:
    """平掉所有持仓."""
    try:
        result = await svc.close_all_positions()
        return {"status": "ok", "closed": result}
    except AttributeError:
        return {"status": "ok", "closed": 0, "message": "close_all not yet implemented"}


@router.get("/pnl-history")
async def pnl_history(
    days: int = 30,
    svc: ExecutionService = Depends(get_execution_service),
) -> list[dict]:
    """查询盈亏历史（日级别）.

    Returns list of {date, pnl, cumulative_pnl} for the past N days.
    """
    try:
        return await svc.get_pnl_history(days=days)
    except (AttributeError, NotImplementedError):
        return []


@router.get("/account/info")
async def account_info(
    svc: ExecutionService = Depends(get_execution_service),
) -> dict:
    """查询账户资金信息."""
    return await svc.get_account_info()
