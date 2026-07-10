#!/usr/bin/env python3
"""TqSdk Gateway — sole process holding TqSdk credentials and TqApi session.

Trading-platform and data-collector call this HTTP API; they never receive
plaintext futures passwords from PolarPrivate.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from credentials import load_credentials
from session import SessionBusyError, get_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tqsdk-gateway")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    session = get_session()
    try:
        creds = load_credentials()
        session.connect(creds)
        logger.info("Gateway ready (mode=%s)", session.account_mode)
    except Exception:
        logger.exception("TqSdk gateway failed to connect — endpoints return 503")
    try:
        yield
    finally:
        session.disconnect()


app = FastAPI(title="TqSdk Gateway", version="1.0.0", lifespan=lifespan)


class OrderRequest(BaseModel):
    symbol: str
    direction: str = Field(description="BUY or SELL")
    offset: str = Field(description="OPEN, CLOSE, CLOSETODAY, etc.")
    price: float
    volume: int = Field(ge=1)


def _require_session():
    session = get_session()
    if not session.connected:
        raise HTTPException(status_code=503, detail="TqSdk session not connected")
    return session


@app.get("/health")
def health() -> dict:
    session = get_session()
    return {
        "status": "ok" if session.connected else "degraded",
        "connected": session.connected,
        "account_mode": session.account_mode,
    }


@app.get("/api/v1/account")
def account() -> dict:
    session = _require_session()
    try:
        return session.get_account_info()
    except SessionBusyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/v1/positions")
def positions() -> dict:
    session = _require_session()
    try:
        return {"items": session.get_positions()}
    except SessionBusyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/v1/orders")
def place_order(body: OrderRequest) -> dict:
    session = _require_session()
    direction = body.direction.upper()
    if direction not in ("BUY", "SELL"):
        raise HTTPException(status_code=422, detail="direction must be BUY or SELL")
    order_id = session.place_order(
        symbol=body.symbol,
        direction=direction,
        offset=body.offset.upper(),
        price=body.price,
        volume=body.volume,
    )
    return {"order_id": order_id}


@app.delete("/api/v1/orders/{order_id}")
def cancel_order(order_id: str) -> dict:
    session = _require_session()
    ok = session.cancel_order(order_id)
    return {"cancelled": ok, "order_id": order_id}


@app.get("/api/v1/orders")
def list_orders() -> dict:
    session = _require_session()
    try:
        return {"items": session.get_orders()}
    except SessionBusyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/v1/orders/{order_id}")
def get_order(order_id: str) -> dict:
    session = _require_session()
    try:
        order = session.get_order(order_id)
    except SessionBusyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if order is None:
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found")
    return order


@app.get("/api/v1/trades")
def list_trades() -> dict:
    session = _require_session()
    try:
        return {"items": session.get_trades()}
    except SessionBusyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/v1/market/quote/{symbol}")
def quote(symbol: str) -> dict:
    session = _require_session()
    try:
        return session.get_quote(symbol)
    except SessionBusyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/v1/market/klines/{symbol}")
def klines(symbol: str, duration: int = 300, length: int = 200) -> dict:
    session = _require_session()
    try:
        rows = session.get_klines(symbol, duration, max(1, min(length, 8000)))
        return {"symbol": symbol, "duration": duration, "items": rows}
    except SessionBusyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/v1/market/instruments")
def instruments(exchange_id: str | None = None, ins_class: str = "FUTURE") -> dict:
    session = _require_session()
    try:
        symbols = session.list_instruments(exchange_id=exchange_id, ins_class=ins_class)
        return {"items": [{"symbol": s} for s in symbols]}
    except SessionBusyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("TQSDK_GATEWAY_HOST", "127.0.0.1")
    port = int(os.getenv("TQSDK_GATEWAY_PORT", "12890"))
    uvicorn.run("main:app", host=host, port=port, reload=False)
