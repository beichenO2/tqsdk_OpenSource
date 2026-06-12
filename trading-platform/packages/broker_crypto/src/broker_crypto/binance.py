"""Binance 交易所适配器实现。"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
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
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_BINANCE_REST = "https://api.binance.com"
_BINANCE_REST_TESTNET = "https://testnet.binance.vision"
_BINANCE_WS = "wss://stream.binance.com:9443/ws"
_BINANCE_WS_TESTNET = "wss://testnet.binance.vision/ws"

_ORDER_STATUS_MAP: dict[str, OrderStatus] = {
    "NEW": OrderStatus.OPEN,
    "PARTIALLY_FILLED": OrderStatus.PARTIAL_FILLED,
    "FILLED": OrderStatus.FILLED,
    "CANCELED": OrderStatus.CANCELLED,
    "REJECTED": OrderStatus.REJECTED,
    "EXPIRED": OrderStatus.CANCELLED,
}

_INTERVAL_MAP: dict[str, str] = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "8h": "8h", "12h": "12h",
    "1d": "1d", "3d": "3d", "1w": "1w", "1M": "1M",
}


class BinanceAdapter(ExchangeAdapter):
    """Binance spot & futures REST + WebSocket adapter."""

    def __init__(self, credentials: ExchangeCredentials) -> None:
        super().__init__(credentials)
        self._session: aiohttp.ClientSession | None = None
        base = _BINANCE_REST_TESTNET if credentials.testnet else _BINANCE_REST
        self._base_url = base
        ws_base = _BINANCE_WS_TESTNET if credentials.testnet else _BINANCE_WS
        self._ws_url = ws_base

    @property
    def exchange(self) -> Exchange:
        return Exchange.BINANCE

    # ── helpers ──────────────────────────────────────────────────────

    def _sign(self, params: dict) -> dict:
        params["timestamp"] = int(time.time() * 1000)
        query = urlencode(params)
        sig = hmac.new(
            self._credentials.api_secret.encode(),
            query.encode(),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = sig
        return params

    def _headers(self) -> dict[str, str]:
        return {"X-MBX-APIKEY": self._credentials.api_key}

    async def _get(self, path: str, params: dict | None = None, signed: bool = False) -> dict:
        assert self._session is not None
        params = params or {}
        if signed:
            params = self._sign(params)
        async with self._session.get(
            f"{self._base_url}{path}", params=params, headers=self._headers()
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def _post(self, path: str, params: dict, signed: bool = True) -> dict:
        assert self._session is not None
        if signed:
            params = self._sign(params)
        async with self._session.post(
            f"{self._base_url}{path}", params=params, headers=self._headers()
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def _delete(self, path: str, params: dict, signed: bool = True) -> dict:
        assert self._session is not None
        if signed:
            params = self._sign(params)
        async with self._session.delete(
            f"{self._base_url}{path}", params=params, headers=self._headers()
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    @staticmethod
    def _ts(ms: int) -> datetime:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)

    # ── Market Data (REST) ──────────────────────────────────────────

    async def get_ticker(self, symbol: str) -> Ticker:
        data = await self._get("/api/v3/ticker/bookTicker", {"symbol": symbol})
        price_data = await self._get("/api/v3/ticker/24hr", {"symbol": symbol})
        return Ticker(
            exchange=Exchange.BINANCE,
            symbol=symbol,
            bid=Decimal(data["bidPrice"]),
            ask=Decimal(data["askPrice"]),
            last=Decimal(price_data["lastPrice"]),
            volume_24h=Decimal(price_data["volume"]),
            timestamp=self._ts(int(price_data["closeTime"])),
        )

    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        data = await self._get("/api/v3/depth", {"symbol": symbol, "limit": depth})
        return OrderBook(
            exchange=Exchange.BINANCE,
            symbol=symbol,
            bids=[(Decimal(p), Decimal(q)) for p, q in data["bids"]],
            asks=[(Decimal(p), Decimal(q)) for p, q in data["asks"]],
            timestamp=datetime.now(tz=timezone.utc),
        )

    async def get_ohlcv(
        self, symbol: str, interval: str = "1m", limit: int = 500
    ) -> list[OHLCV]:
        bn_interval = _INTERVAL_MAP.get(interval, interval)
        data = await self._get(
            "/api/v3/klines",
            {"symbol": symbol, "interval": bn_interval, "limit": limit},
        )
        return [
            OHLCV(
                exchange=Exchange.BINANCE,
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
        data = await self._get("/api/v3/trades", {"symbol": symbol, "limit": limit})
        return [
            Trade(
                exchange=Exchange.BINANCE,
                symbol=symbol,
                trade_id=str(t["id"]),
                side=Side.BUY if t["isBuyerMaker"] else Side.SELL,
                price=Decimal(t["price"]),
                quantity=Decimal(t["qty"]),
                timestamp=self._ts(t["time"]),
            )
            for t in data
        ]

    # ── Market Data (WebSocket) ─────────────────────────────────────

    async def stream_ticker(self, symbol: str) -> AsyncIterator[Ticker]:
        stream = f"{symbol.lower()}@bookTicker"
        async for msg in self._ws_listen(stream):
            yield Ticker(
                exchange=Exchange.BINANCE,
                symbol=symbol,
                bid=Decimal(msg["b"]),
                ask=Decimal(msg["a"]),
                last=Decimal(msg.get("b", "0")),
                volume_24h=Decimal("0"),
                timestamp=datetime.now(tz=timezone.utc),
            )

    async def stream_orderbook(self, symbol: str) -> AsyncIterator[OrderBook]:
        stream = f"{symbol.lower()}@depth20@100ms"
        async for msg in self._ws_listen(stream):
            yield OrderBook(
                exchange=Exchange.BINANCE,
                symbol=symbol,
                bids=[(Decimal(p), Decimal(q)) for p, q in msg["bids"]],
                asks=[(Decimal(p), Decimal(q)) for p, q in msg["asks"]],
                timestamp=datetime.now(tz=timezone.utc),
            )

    async def stream_trades(self, symbol: str) -> AsyncIterator[Trade]:
        stream = f"{symbol.lower()}@trade"
        async for msg in self._ws_listen(stream):
            yield Trade(
                exchange=Exchange.BINANCE,
                symbol=symbol,
                trade_id=str(msg["t"]),
                side=Side.BUY if msg["m"] else Side.SELL,
                price=Decimal(msg["p"]),
                quantity=Decimal(msg["q"]),
                timestamp=self._ts(msg["T"]),
            )

    async def stream_ohlcv(
        self, symbol: str, interval: str = "1m"
    ) -> AsyncIterator[OHLCV]:
        bn_interval = _INTERVAL_MAP.get(interval, interval)
        stream = f"{symbol.lower()}@kline_{bn_interval}"
        async for msg in self._ws_listen(stream):
            k = msg["k"]
            yield OHLCV(
                exchange=Exchange.BINANCE,
                symbol=symbol,
                interval=interval,
                open=Decimal(k["o"]),
                high=Decimal(k["h"]),
                low=Decimal(k["l"]),
                close=Decimal(k["c"]),
                volume=Decimal(k["v"]),
                timestamp=self._ts(k["t"]),
            )

    async def _ws_listen(self, stream: str) -> AsyncIterator[dict]:
        url = f"{self._ws_url}/{stream}"
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(url) as ws:
                async for raw in ws:
                    if raw.type == aiohttp.WSMsgType.TEXT:
                        yield json.loads(raw.data)
                    elif raw.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break

    # ── Trading ─────────────────────────────────────────────────────

    async def place_order(self, request: OrderRequest) -> OrderResponse:
        params: dict = {
            "symbol": request.symbol,
            "side": request.side.value.upper(),
            "type": self._map_order_type(request.order_type),
            "quantity": str(request.quantity),
        }
        if request.time_in_force and request.order_type == OrderType.LIMIT:
            params["timeInForce"] = request.time_in_force.value.upper()
        if request.price is not None:
            params["price"] = str(request.price)
        if request.stop_price is not None:
            params["stopPrice"] = str(request.stop_price)
        if request.client_order_id:
            params["newClientOrderId"] = request.client_order_id

        data = await self._post("/api/v3/order", params)
        return self._parse_order(data)

    async def cancel_order(self, order_id: str, symbol: str) -> OrderResponse:
        data = await self._delete(
            "/api/v3/order", {"symbol": symbol, "orderId": order_id}
        )
        return self._parse_order(data)

    async def get_order(self, order_id: str, symbol: str) -> OrderResponse:
        data = await self._get(
            "/api/v3/order", {"symbol": symbol, "orderId": order_id}, signed=True
        )
        return self._parse_order(data)

    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]:
        params = {}
        if symbol:
            params["symbol"] = symbol
        data = await self._get("/api/v3/openOrders", params, signed=True)
        return [self._parse_order(o) for o in data]

    # ── Account ─────────────────────────────────────────────────────

    async def get_balances(self) -> list[Balance]:
        data = await self._get("/api/v3/account", signed=True)
        return [
            Balance(
                exchange=Exchange.BINANCE,
                asset=b["asset"],
                free=Decimal(b["free"]),
                locked=Decimal(b["locked"]),
            )
            for b in data["balances"]
            if Decimal(b["free"]) + Decimal(b["locked"]) > 0
        ]

    async def get_positions(self, symbol: str | None = None) -> list[Position]:
        # Spot does not have positions; return empty for now.
        # Futures adapter will override this.
        return []

    # ── Lifecycle ───────────────────────────────────────────────────

    async def connect(self) -> None:
        self._session = aiohttp.ClientSession()
        logger.info("Binance adapter connected (testnet=%s)", self._credentials.testnet)

    async def disconnect(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None
        logger.info("Binance adapter disconnected")

    # ── private helpers ─────────────────────────────────────────────

    @staticmethod
    def _map_order_type(ot: OrderType) -> str:
        return {
            OrderType.LIMIT: "LIMIT",
            OrderType.MARKET: "MARKET",
            OrderType.STOP_LIMIT: "STOP_LOSS_LIMIT",
            OrderType.STOP_MARKET: "STOP_LOSS",
        }[ot]

    def _parse_order(self, data: dict) -> OrderResponse:
        return OrderResponse(
            exchange=Exchange.BINANCE,
            order_id=str(data["orderId"]),
            client_order_id=data.get("clientOrderId"),
            symbol=data["symbol"],
            side=Side.BUY if data["side"] == "BUY" else Side.SELL,
            order_type=OrderType.LIMIT if data["type"] == "LIMIT" else OrderType.MARKET,
            status=_ORDER_STATUS_MAP.get(data["status"], OrderStatus.PENDING),
            quantity=Decimal(data["origQty"]),
            filled_quantity=Decimal(data.get("executedQty", "0")),
            price=Decimal(data["price"]) if data.get("price") else None,
            avg_fill_price=Decimal(data["avgPrice"]) if data.get("avgPrice") else None,
            created_at=self._ts(data.get("time", data.get("transactTime", 0))),
            updated_at=self._ts(data.get("updateTime", data.get("transactTime", 0))),
        )
