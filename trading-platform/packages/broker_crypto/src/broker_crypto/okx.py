"""OKX 交易所适配器实现（接口桩）。"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import AsyncIterator

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

_OKX_REST = "https://www.okx.com"
_OKX_REST_TESTNET = "https://www.okx.com"  # OKX uses header flag for demo
_OKX_WS = "wss://ws.okx.com:8443/ws/v5/public"


class OKXAdapter(ExchangeAdapter):
    """OKX REST + WebSocket adapter."""

    def __init__(self, credentials: ExchangeCredentials) -> None:
        super().__init__(credentials)
        self._session: aiohttp.ClientSession | None = None

    @property
    def exchange(self) -> Exchange:
        return Exchange.OKX

    # ── Auth helpers ────────────────────────────────────────────────

    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        prehash = f"{timestamp}{method}{path}{body}"
        signature = hmac.new(
            self._credentials.api_secret.encode(),
            prehash.encode(),
            hashlib.sha256,
        ).digest()
        return base64.b64encode(signature).decode()

    def _headers(self, method: str, path: str, body: str = "") -> dict[str, str]:
        ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        sig = self._sign(ts, method, path, body)
        headers = {
            "OK-ACCESS-KEY": self._credentials.api_key,
            "OK-ACCESS-SIGN": sig,
            "OK-ACCESS-TIMESTAMP": ts,
            "OK-ACCESS-PASSPHRASE": self._credentials.passphrase or "",
            "Content-Type": "application/json",
        }
        if self._credentials.testnet:
            headers["x-simulated-trading"] = "1"
        return headers

    async def _get(self, path: str, params: dict | None = None) -> dict:
        assert self._session is not None
        url = f"{_OKX_REST}{path}"
        async with self._session.get(
            url, params=params, headers=self._headers("GET", path)
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def _post(self, path: str, body: dict) -> dict:
        assert self._session is not None
        body_str = json.dumps(body)
        url = f"{_OKX_REST}{path}"
        async with self._session.post(
            url, data=body_str, headers=self._headers("POST", path, body_str)
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    # ── Market Data (REST) ──────────────────────────────────────────

    async def get_ticker(self, symbol: str) -> Ticker:
        data = await self._get("/api/v5/market/ticker", {"instId": symbol})
        t = data["data"][0]
        return Ticker(
            exchange=Exchange.OKX,
            symbol=symbol,
            bid=Decimal(t["bidPx"]),
            ask=Decimal(t["askPx"]),
            last=Decimal(t["last"]),
            volume_24h=Decimal(t["vol24h"]),
            timestamp=datetime.fromtimestamp(int(t["ts"]) / 1000, tz=timezone.utc),
        )

    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        data = await self._get(
            "/api/v5/market/books", {"instId": symbol, "sz": str(depth)}
        )
        book = data["data"][0]
        return OrderBook(
            exchange=Exchange.OKX,
            symbol=symbol,
            bids=[(Decimal(b[0]), Decimal(b[1])) for b in book["bids"]],
            asks=[(Decimal(a[0]), Decimal(a[1])) for a in book["asks"]],
            timestamp=datetime.fromtimestamp(int(book["ts"]) / 1000, tz=timezone.utc),
        )

    async def get_ohlcv(
        self, symbol: str, interval: str = "1m", limit: int = 500
    ) -> list[OHLCV]:
        bar_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1H", "4h": "4H", "1d": "1D"}
        bar = bar_map.get(interval, interval)
        data = await self._get(
            "/api/v5/market/candles",
            {"instId": symbol, "bar": bar, "limit": str(limit)},
        )
        return [
            OHLCV(
                exchange=Exchange.OKX,
                symbol=symbol,
                interval=interval,
                open=Decimal(c[1]),
                high=Decimal(c[2]),
                low=Decimal(c[3]),
                close=Decimal(c[4]),
                volume=Decimal(c[5]),
                timestamp=datetime.fromtimestamp(int(c[0]) / 1000, tz=timezone.utc),
            )
            for c in data["data"]
        ]

    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]:
        data = await self._get(
            "/api/v5/market/trades", {"instId": symbol, "limit": str(limit)}
        )
        return [
            Trade(
                exchange=Exchange.OKX,
                symbol=symbol,
                trade_id=t["tradeId"],
                side=Side.BUY if t["side"] == "buy" else Side.SELL,
                price=Decimal(t["px"]),
                quantity=Decimal(t["sz"]),
                timestamp=datetime.fromtimestamp(int(t["ts"]) / 1000, tz=timezone.utc),
            )
            for t in data["data"]
        ]

    # ── WebSocket streams (stubs) ───────────────────────────────────

    async def stream_ticker(self, symbol: str) -> AsyncIterator[Ticker]:
        async for msg in self._ws_subscribe("tickers", symbol):
            d = msg["data"][0]
            yield Ticker(
                exchange=Exchange.OKX, symbol=symbol,
                bid=Decimal(d["bidPx"]), ask=Decimal(d["askPx"]),
                last=Decimal(d["last"]), volume_24h=Decimal(d.get("vol24h", "0")),
                timestamp=datetime.fromtimestamp(int(d["ts"]) / 1000, tz=timezone.utc),
            )

    async def stream_orderbook(self, symbol: str) -> AsyncIterator[OrderBook]:
        async for msg in self._ws_subscribe("books5", symbol):
            d = msg["data"][0]
            yield OrderBook(
                exchange=Exchange.OKX, symbol=symbol,
                bids=[(Decimal(b[0]), Decimal(b[1])) for b in d["bids"]],
                asks=[(Decimal(a[0]), Decimal(a[1])) for a in d["asks"]],
                timestamp=datetime.fromtimestamp(int(d["ts"]) / 1000, tz=timezone.utc),
            )

    async def stream_trades(self, symbol: str) -> AsyncIterator[Trade]:
        async for msg in self._ws_subscribe("trades", symbol):
            for t in msg["data"]:
                yield Trade(
                    exchange=Exchange.OKX, symbol=symbol,
                    trade_id=t["tradeId"],
                    side=Side.BUY if t["side"] == "buy" else Side.SELL,
                    price=Decimal(t["px"]), quantity=Decimal(t["sz"]),
                    timestamp=datetime.fromtimestamp(int(t["ts"]) / 1000, tz=timezone.utc),
                )

    async def stream_ohlcv(self, symbol: str, interval: str = "1m") -> AsyncIterator[OHLCV]:
        bar_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1H", "4h": "4H", "1d": "1D"}
        bar = bar_map.get(interval, interval)
        async for msg in self._ws_subscribe(f"candle{bar}", symbol):
            for c in msg["data"]:
                yield OHLCV(
                    exchange=Exchange.OKX, symbol=symbol, interval=interval,
                    open=Decimal(c[1]), high=Decimal(c[2]),
                    low=Decimal(c[3]), close=Decimal(c[4]),
                    volume=Decimal(c[5]),
                    timestamp=datetime.fromtimestamp(int(c[0]) / 1000, tz=timezone.utc),
                )

    async def _ws_subscribe(self, channel: str, inst_id: str) -> AsyncIterator[dict]:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(_OKX_WS) as ws:
                sub_msg = {"op": "subscribe", "args": [{"channel": channel, "instId": inst_id}]}
                await ws.send_json(sub_msg)
                async for raw in ws:
                    if raw.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(raw.data)
                        if "data" in data:
                            yield data
                    elif raw.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break

    # ── Trading ─────────────────────────────────────────────────────

    async def place_order(self, request: OrderRequest) -> OrderResponse:
        body = {
            "instId": request.symbol,
            "tdMode": "cash",
            "side": request.side.value,
            "ordType": "limit" if request.price else "market",
            "sz": str(request.quantity),
        }
        if request.price:
            body["px"] = str(request.price)
        if request.client_order_id:
            body["clOrdId"] = request.client_order_id

        result = await self._post("/api/v5/trade/order", body)
        d = result["data"][0]
        now = datetime.now(tz=timezone.utc)
        return OrderResponse(
            exchange=Exchange.OKX,
            order_id=d["ordId"],
            client_order_id=d.get("clOrdId"),
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            status=OrderStatus.OPEN if d["sCode"] == "0" else OrderStatus.REJECTED,
            quantity=request.quantity,
            price=request.price,
            created_at=now,
            updated_at=now,
        )

    async def cancel_order(self, order_id: str, symbol: str) -> OrderResponse:
        await self._post(
            "/api/v5/trade/cancel-order", {"instId": symbol, "ordId": order_id}
        )
        now = datetime.now(tz=timezone.utc)
        return OrderResponse(
            exchange=Exchange.OKX, order_id=order_id, symbol=symbol,
            side=Side.BUY, order_type=OrderType.LIMIT,
            status=OrderStatus.CANCELLED, quantity=Decimal("0"),
            created_at=now, updated_at=now,
        )

    async def get_order(self, order_id: str, symbol: str) -> OrderResponse:
        data = await self._get(
            "/api/v5/trade/order", {"instId": symbol, "ordId": order_id}
        )
        d = data["data"][0]
        return self._parse_okx_order(d)

    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]:
        params = {}
        if symbol:
            params["instId"] = symbol
        data = await self._get("/api/v5/trade/orders-pending", params)
        return [self._parse_okx_order(o) for o in data["data"]]

    # ── Account ─────────────────────────────────────────────────────

    async def get_balances(self) -> list[Balance]:
        data = await self._get("/api/v5/account/balance")
        balances: list[Balance] = []
        for detail in data["data"][0].get("details", []):
            free = Decimal(detail.get("availBal", "0"))
            locked = Decimal(detail.get("frozenBal", "0"))
            if free + locked > 0:
                balances.append(Balance(
                    exchange=Exchange.OKX, asset=detail["ccy"],
                    free=free, locked=locked,
                ))
        return balances

    async def get_positions(self, symbol: str | None = None) -> list[Position]:
        params = {}
        if symbol:
            params["instId"] = symbol
        data = await self._get("/api/v5/account/positions", params)
        return [
            Position(
                exchange=Exchange.OKX,
                symbol=p["instId"],
                side=Side.BUY if Decimal(p.get("pos", "0")) > 0 else Side.SELL,
                quantity=abs(Decimal(p.get("pos", "0"))),
                entry_price=Decimal(p.get("avgPx", "0")),
                unrealized_pnl=Decimal(p.get("upl", "0")),
                leverage=int(p.get("lever", "1")),
                timestamp=datetime.fromtimestamp(int(p["cTime"]) / 1000, tz=timezone.utc),
            )
            for p in data["data"]
            if Decimal(p.get("pos", "0")) != 0
        ]

    # ── Lifecycle ───────────────────────────────────────────────────

    async def connect(self) -> None:
        self._session = aiohttp.ClientSession()
        logger.info("OKX adapter connected (testnet=%s)", self._credentials.testnet)

    async def disconnect(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None
        logger.info("OKX adapter disconnected")

    # ── private helpers ─────────────────────────────────────────────

    def _parse_okx_order(self, d: dict) -> OrderResponse:
        status_map = {
            "live": OrderStatus.OPEN,
            "partially_filled": OrderStatus.PARTIAL_FILLED,
            "filled": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
        }
        return OrderResponse(
            exchange=Exchange.OKX,
            order_id=d["ordId"],
            client_order_id=d.get("clOrdId"),
            symbol=d["instId"],
            side=Side.BUY if d["side"] == "buy" else Side.SELL,
            order_type=OrderType.LIMIT if d["ordType"] == "limit" else OrderType.MARKET,
            status=status_map.get(d.get("state", ""), OrderStatus.PENDING),
            quantity=Decimal(d.get("sz", "0")),
            filled_quantity=Decimal(d.get("accFillSz", "0")),
            price=Decimal(d["px"]) if d.get("px") else None,
            avg_fill_price=Decimal(d["avgPx"]) if d.get("avgPx") else None,
            created_at=datetime.fromtimestamp(int(d["cTime"]) / 1000, tz=timezone.utc),
            updated_at=datetime.fromtimestamp(int(d["uTime"]) / 1000, tz=timezone.utc),
        )
