"""行情数据路由."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.deps import get_market_service
from app.services.market import MarketService
from core.models.bar import Bar
from core.models.tick import Tick

router = APIRouter(prefix="/market", tags=["market"])


def _tick_to_response(tick: Tick) -> dict:
    return tick.model_dump(mode="json")


def _bars_to_list(bars: list[Bar]) -> list[dict]:
    return [b.model_dump(mode="json") for b in bars]


@router.get("/quote/{symbol}")
async def get_quote(
    symbol: str,
    market: MarketService = Depends(get_market_service),
) -> dict:
    """获取最新行情快照."""
    tick = await market.get_quote(symbol)
    if tick is None:
        return {
            "symbol": symbol,
            "last_price": None,
            "message": "no_quote",
        }
    data = _tick_to_response(tick)
    data["message"] = "ok"
    return data


@router.get("/klines/{symbol}")
async def get_klines(
    symbol: str,
    duration: int = Query(60, description="K线周期(秒)"),
    limit: int = Query(200, ge=1, le=8000),
    market: MarketService = Depends(get_market_service),
) -> list[dict]:
    """获取 K 线数据."""
    bars = await market.get_klines(
        symbol, duration_seconds=duration, data_length=limit
    )
    return _bars_to_list(bars)


@router.get("/instruments")
async def list_instruments(
    exchange: str | None = None,
    market: MarketService = Depends(get_market_service),
) -> list[dict]:
    """获取合约列表."""
    return await market.list_instruments(exchange_id=exchange)
