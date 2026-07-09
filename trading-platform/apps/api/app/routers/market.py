"""行情数据路由."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, Query

from app.deps import get_market_service
from app.services.market import MarketService
from core.models.bar import Bar
from core.models.tick import Tick

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/market", tags=["market"])

_REPO = Path(__file__).resolve().parents[4]
_FUTURES_CACHE = _REPO / "data" / "futures_cache"
_QUOTE_TIMEOUT_S = 5.0


def _tick_to_response(tick: Tick) -> dict:
    return tick.model_dump(mode="json")


def _bars_to_list(bars: list[Bar]) -> list[dict]:
    return [b.model_dump(mode="json") for b in bars]


_close_cache: dict[str, tuple[float, dict]] = {}
_CLOSE_CACHE_TTL_S = 300.0


def _cached_last_close(symbol: str) -> dict | None:
    """闭市回退：从 parquet 缓存取该品种最新 bar 的 close（内存缓存 5min）。"""
    import time as _time

    hit = _close_cache.get(symbol)
    if hit and _time.time() - hit[0] < _CLOSE_CACHE_TTL_S:
        return dict(hit[1])

    raw = symbol.split(".")[-1] if "." in symbol else symbol
    base = "".join(c for c in raw if not c.isdigit()) or raw
    if not _FUTURES_CACHE.exists():
        return None
    files: list[Path] = []
    for cand in (base, base.lower(), base.upper()):
        files = sorted(_FUTURES_CACHE.glob(f"KQ_m_*_{cand}_5m.parquet"))
        if files:
            break
    if not files:
        return None
    try:
        import pandas as pd
        df = pd.read_parquet(files[0])
        ts_col = next(
            (c for c in ("datetime", "timestamp", "dt", "date") if c in df.columns), None
        )
        row = df.sort_values(ts_col).iloc[-1] if ts_col else df.iloc[-1]
        ts_val = row[ts_col] if ts_col else None
        if ts_val is not None:
            # cache stores ns-epoch floats; normalize to ISO string
            try:
                ts_num = float(ts_val)
                if ts_num > 1e17:
                    ts_val = pd.Timestamp(int(ts_num), unit="ns").isoformat()
                elif ts_num > 1e11:
                    ts_val = pd.Timestamp(int(ts_num), unit="ms").isoformat()
                elif ts_num > 1e8:
                    ts_val = pd.Timestamp(int(ts_num), unit="s").isoformat()
                else:
                    ts_val = str(ts_val)
            except (TypeError, ValueError):
                ts_val = str(ts_val)
        result = {
            "symbol": symbol,
            "last_price": float(row["close"]),
            "datetime": ts_val,
            "message": "closed_market_cache",
            "source": "futures_cache",
        }
        import time as _time
        _close_cache[symbol] = (_time.time(), dict(result))
        return result
    except Exception as e:
        logger.debug("cache fallback failed for %s: %s", symbol, e)
        return None


@router.get("/quote/{symbol}")
async def get_quote(
    symbol: str,
    market: MarketService = Depends(get_market_service),
) -> dict:
    """获取最新行情快照；闭市/超时回退最近缓存 close."""
    tick: Tick | None = None
    try:
        tick = await asyncio.wait_for(market.get_quote(symbol), timeout=_QUOTE_TIMEOUT_S)
    except (asyncio.TimeoutError, Exception) as e:  # noqa: BLE001 — gateway may hang when closed
        logger.debug("quote timeout/error for %s: %s", symbol, e)

    if tick is None:
        cached = _cached_last_close(symbol)
        if cached:
            return cached
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


def _cached_instruments(exchange: str | None = None) -> list[dict]:
    """闭市回退：从缓存文件名合成主连合约列表。"""
    out: list[dict] = []
    if not _FUTURES_CACHE.exists():
        return out
    seen: set[str] = set()
    for p in sorted(_FUTURES_CACHE.glob("KQ_m_*_*_5m.parquet")):
        parts = p.stem.split("_")  # KQ m SHFE rb 5m
        if len(parts) < 5:
            continue
        exch, sym = parts[2], parts[3]
        if exchange and exch != exchange:
            continue
        symbol = f"KQ.m@{exch}.{sym}"
        if symbol in seen:
            continue
        seen.add(symbol)
        out.append({
            "symbol": symbol,
            "exchange": exch,
            "name": f"{sym} 主连",
            "source": "futures_cache",
        })
    return out


@router.get("/instruments")
async def list_instruments(
    exchange: str | None = None,
    market: MarketService = Depends(get_market_service),
) -> list[dict]:
    """获取合约列表；闭市/超时回退缓存主连列表."""
    try:
        items = await asyncio.wait_for(
            market.list_instruments(exchange_id=exchange), timeout=_QUOTE_TIMEOUT_S
        )
        if items:
            return items
    except (asyncio.TimeoutError, Exception) as e:  # noqa: BLE001
        logger.debug("instruments timeout/error: %s", e)
    return _cached_instruments(exchange)
