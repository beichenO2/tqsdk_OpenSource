"""WEEX 交易所适配器实现 — V3(BETA) REST + WebSocket."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import AsyncIterator
from urllib.parse import urlencode

import aiohttp

from .base import ExchangeAdapter
from .models import (
    Balance,
    Exchange,
    ExchangeCredentials,
    OHLCV,
    OrderBook,
    OrderRequest,
    OrderResponse,
    OrderStatus,
    OrderType,
    Position,
    Side,
    Ticker,
    Trade,
)

logger = logging.getLogger(__name__)

_WEEX_REST = "https://api-spot.weex.com"
_WEEX_WS_PUBLIC = "wss://ws-spot.weex.com/v3/ws/public"
_WEEX_WS_PRIVATE = "wss://ws-spot.weex.com/v3/ws/private"

_ORDER_STATUS_MAP: dict[str, OrderStatus] = {
    "NEW": OrderStatus.OPEN,
    "new": OrderStatus.OPEN,
    "PARTIALLY_FILLED": OrderStatus.PARTIAL_FILLED,
    "partial_fill": OrderStatus.PARTIAL_FILLED,
    "FILLED": OrderStatus.FILLED,
    "full_fill": OrderStatus.FILLED,
    "CANCELED": OrderStatus.CANCELLED,
    "cancelled": OrderStatus.CANCELLED,
    "REJECTED": OrderStatus.REJECTED,
    "EXPIRED": OrderStatus.CANCELLED,
}

_INTERVAL_MAP: dict[str, str] = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h",
    "8h": "8h", "12h": "12h", "1d": "1d", "1w": "1w", "1M": "1M",
}


class WEEXAdapter(ExchangeAdapter):
    """WEEX spot REST + WebSocket adapter (API V3)."""

    def __init__(self, credentials: ExchangeCredentials) -> None:
        super().__init__(credentials)
        self._session: aiohttp.ClientSession | None = None

    @property
    def exchange(self) -> Exchange:
        return Exchange.WEEX

    # ── Auth helpers ────────────────────────────────────────────────

    def _sign(self, timestamp: str, method: str, path: str,
              query_string: str = "", body: str = "") -> str:
        if query_string:
            prehash = f"{timestamp}{method}{path}?{query_string}{body}"
        else:
            prehash = f"{timestamp}{method}{path}{body}"
        signature = hmac.new(
            self._credentials.api_secret.encode(),
            prehash.encode(),
            hashlib.sha256,
        ).digest()
        return base64.b64encode(signature).decode()

    def _headers(self, method: str, path: str,
                 query_string: str = "", body: str = "") -> dict[str, str]:
        ts = str(int(time.time() * 1000))
        sig = self._sign(ts, method, path, query_string, body)
        return {
            "ACCESS-KEY": self._credentials.api_key,
            "ACCESS-SIGN": sig,
            "ACCESS-TIMESTAMP": ts,
            "ACCESS-PASSPHRASE": self._credentials.passphrase or "",
            "Content-Type": "application/json",
        }

    async def _get(self, path: str, params: dict | None = None,
                   signed: bool = False) -> dict:
        assert self._session is not None
        params = params or {}
        qs = urlencode(params) if params else ""
        headers = self._headers("GET", path, qs) if signed else {}
        url = f"{_WEEX_REST}{path}"
        async with self._session.get(url, params=params, headers=headers) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def _post(self, path: str, body: dict) -> dict:
        assert self._session is not None
        body_str = json.dumps(body)
        headers = self._headers("POST", path, body=body_str)
        url = f"{_WEEX_REST}{path}"
        async with self._session.post(url, data=body_str, headers=headers) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def _delete(self, path: str, params: dict) -> dict:
        assert self._session is not None
        qs = urlencode(params) if params else ""
        headers = self._headers("DELETE", path, qs)
        url = f"{_WEEX_REST}{path}"
        async with self._session.delete(url, params=params, headers=headers) as resp:
            resp.raise_for_status()
            return await resp.json()

    @staticmethod
    def _ts(ms: int) -> datetime:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)

    # ── Market Data (REST) ──────────────────────────────────────────

    async def get_ticker(self, symbol: str) -> Ticker:
        data = await self._get(
            "/api/v3/market/ticker/24hr", {"symbol": symbol}
        )
        item = data if isinstance(data, dict) else data[0]
        return Ticker(
            exchange=Exchange.WEEX,
            symbol=symbol,
            bid=Decimal(item["bidPrice"]),
            ask=Decimal(item["askPrice"]),
            last=Decimal(item["lastPrice"]),
            volume_24h=Decimal(item["volume"]),
            timestamp=self._ts(int(item["closeTime"])),
        )

    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        limit = 15 if depth <= 15 else 200
        data = await self._get(
            "/api/v3/market/depth", {"symbol": symbol, "limit": limit}
        )
        return OrderBook(
            exchange=Exchange.WEEX,
            symbol=symbol,
            bids=[(Decimal(p), Decimal(q)) for p, q in data["bids"]],
            asks=[(Decimal(p), Decimal(q)) for p, q in data["asks"]],
            timestamp=datetime.now(tz=timezone.utc),
        )

    async def get_ohlcv(
        self, symbol: str, interval: str = "1m", limit: int = 500
    ) -> list[OHLCV]:
        weex_interval = _INTERVAL_MAP.get(interval, interval)
        data = await self._get(
            "/api/v3/market/klines",
            {"symbol": symbol, "interval": weex_interval, "limit": limit},
        )
        return [
            OHLCV(
                exchange=Exchange.WEEX,
                symbol=symbol,
                interval=interval,
                open=Decimal(str(k[1])),
                high=Decimal(str(k[2])),
                low=Decimal(str(k[3])),
                close=Decimal(str(k[4])),
                volume=Decimal(str(k[5])),
                timestamp=self._ts(k[0]),
            )
            for k in data
        ]

    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]:
        data = await self._get(
            "/api/v3/market/trades", {"symbol": symbol, "limit": limit}
        )
        return [
            Trade(
                exchange=Exchange.WEEX,
                symbol=symbol,
                trade_id=str(t.get("id", t.get("tradeId", ""))),
                side=Side.BUY if t.get("isBuyerMaker", False) else Side.SELL,
                price=Decimal(str(t["price"])),
                quantity=Decimal(str(t["qty"])),
                timestamp=self._ts(t["time"]),
            )
            for t in data
        ]

    # ── Market Data (WebSocket) ─────────────────────────────────────

    async def stream_ticker(self, symbol: str) -> AsyncIterator[Ticker]:
        async for msg in self._ws_subscribe(f"{symbol}@ticker"):
            yield Ticker(
                exchange=Exchange.WEEX,
                symbol=symbol,
                bid=Decimal(msg.get("b", msg.get("bidPrice", "0"))),
                ask=Decimal(msg.get("a", msg.get("askPrice", "0"))),
                last=Decimal(msg.get("c", msg.get("lastPrice", "0"))),
                volume_24h=Decimal(msg.get("v", msg.get("volume", "0"))),
                timestamp=datetime.now(tz=timezone.utc),
            )

    async def stream_orderbook(self, symbol: str) -> AsyncIterator[OrderBook]:
        async for msg in self._ws_subscribe(f"{symbol}@depth20"):
            yield OrderBook(
                exchange=Exchange.WEEX,
                symbol=symbol,
                bids=[(Decimal(p), Decimal(q)) for p, q in msg.get("bids", [])],
                asks=[(Decimal(p), Decimal(q)) for p, q in msg.get("asks", [])],
                timestamp=datetime.now(tz=timezone.utc),
            )

    async def stream_trades(self, symbol: str) -> AsyncIterator[Trade]:
        async for msg in self._ws_subscribe(f"{symbol}@trade"):
            yield Trade(
                exchange=Exchange.WEEX,
                symbol=symbol,
                trade_id=str(msg.get("t", "")),
                side=Side.BUY if msg.get("m", False) else Side.SELL,
                price=Decimal(str(msg["p"])),
                quantity=Decimal(str(msg["q"])),
                timestamp=self._ts(msg.get("T", int(time.time() * 1000))),
            )

    async def stream_ohlcv(
        self, symbol: str, interval: str = "1m"
    ) -> AsyncIterator[OHLCV]:
        weex_interval = _INTERVAL_MAP.get(interval, interval)
        async for msg in self._ws_subscribe(f"{symbol}@kline_{weex_interval}"):
            k = msg.get("k", msg)
            yield OHLCV(
                exchange=Exchange.WEEX,
                symbol=symbol,
                interval=interval,
                open=Decimal(str(k["o"])),
                high=Decimal(str(k["h"])),
                low=Decimal(str(k["l"])),
                close=Decimal(str(k["c"])),
                volume=Decimal(str(k["v"])),
                timestamp=self._ts(k.get("t", int(time.time() * 1000))),
            )

    async def _ws_subscribe(self, stream: str) -> AsyncIterator[dict]:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(_WEEX_WS_PUBLIC) as ws:
                sub_msg = {
                    "method": "SUBSCRIBE",
                    "params": [stream],
                    "id": 1,
                }
                await ws.send_json(sub_msg)
                async for raw in ws:
                    if raw.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(raw.data)
                        if "result" not in data and "id" not in data:
                            yield data
                    elif raw.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.ERROR,
                    ):
                        break

    # ── Trading ─────────────────────────────────────────────────────

    async def place_order(self, request: OrderRequest) -> OrderResponse:
        body: dict = {
            "symbol": request.symbol,
            "side": request.side.value.upper(),
            "type": request.order_type.value.upper()
                if request.order_type in (OrderType.LIMIT, OrderType.MARKET)
                else "LIMIT",
            "quantity": str(request.quantity),
        }
        if request.order_type == OrderType.LIMIT:
            body["timeInForce"] = request.time_in_force.value.upper()
        if request.price is not None:
            body["price"] = str(request.price)
        if request.client_order_id:
            body["newClientOrderId"] = request.client_order_id

        data = await self._post("/api/v3/order", body)
        now = datetime.now(tz=timezone.utc)
        return OrderResponse(
            exchange=Exchange.WEEX,
            order_id=str(data["orderId"]),
            client_order_id=data.get("clientOrderId"),
            symbol=data["symbol"],
            side=request.side,
            order_type=request.order_type,
            status=OrderStatus.OPEN,
            quantity=request.quantity,
            price=request.price,
            created_at=self._ts(data.get("transactTime", int(time.time() * 1000))),
            updated_at=now,
        )

    async def cancel_order(self, order_id: str, symbol: str) -> OrderResponse:
        data = await self._delete(
            "/api/v3/order", {"symbol": symbol, "orderId": order_id}
        )
        now = datetime.now(tz=timezone.utc)
        return OrderResponse(
            exchange=Exchange.WEEX,
            order_id=str(data["orderId"]),
            symbol=symbol,
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.CANCELLED,
            quantity=Decimal("0"),
            created_at=now,
            updated_at=now,
        )

    async def get_order(self, order_id: str, symbol: str) -> OrderResponse:
        data = await self._get(
            "/api/v3/order",
            {"orderId": order_id},
            signed=True,
        )
        return self._parse_order(data)

    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]:
        params: dict = {}
        if symbol:
            params["symbol"] = symbol
        data = await self._get("/api/v3/openOrders", params, signed=True)
        return [self._parse_order(o) for o in data]

    # ── Account ─────────────────────────────────────────────────────

    async def get_balances(self) -> list[Balance]:
        data = await self._get("/api/v3/account/", signed=True)
        return [
            Balance(
                exchange=Exchange.WEEX,
                asset=b["asset"],
                free=Decimal(b["free"]),
                locked=Decimal(b["locked"]),
            )
            for b in data.get("balances", [])
            if Decimal(b["free"]) + Decimal(b["locked"]) > 0
        ]

    async def get_positions(self, symbol: str | None = None) -> list[Position]:
        # WEEX spot does not have positions; contract adapter would override.
        return []

    # ── Lifecycle ───────────────────────────────────────────────────

    async def connect(self) -> None:
        self._session = aiohttp.ClientSession()
        logger.info("WEEX adapter connected")

    async def disconnect(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None
        logger.info("WEEX adapter disconnected")

    # ── private helpers ─────────────────────────────────────────────

    def _parse_order(self, data: dict) -> OrderResponse:
        return OrderResponse(
            exchange=Exchange.WEEX,
            order_id=str(data["orderId"]),
            client_order_id=data.get("clientOrderId"),
            symbol=data["symbol"],
            side=Side.BUY if data["side"] == "BUY" else Side.SELL,
            order_type=OrderType.LIMIT if data["type"] == "LIMIT" else OrderType.MARKET,
            status=_ORDER_STATUS_MAP.get(data.get("status", ""), OrderStatus.PENDING),
            quantity=Decimal(data.get("origQty", "0")),
            filled_quantity=Decimal(data.get("executedQty", "0")),
            price=Decimal(data["price"]) if data.get("price") else None,
            avg_fill_price=(
                Decimal(data["cummulativeQuoteQty"]) / Decimal(data["executedQty"])
                if data.get("cummulativeQuoteQty") and Decimal(data.get("executedQty", "0")) > 0
                else None
            ),
            created_at=self._ts(data.get("time", 0)),
            updated_at=self._ts(data.get("updateTime", data.get("time", 0))),
        )
