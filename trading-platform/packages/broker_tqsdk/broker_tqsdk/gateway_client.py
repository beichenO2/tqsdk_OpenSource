"""HTTP client for TqSdk Gateway — trading-platform never holds TqSdk passwords."""

from __future__ import annotations

import logging
import os
from decimal import Decimal
from typing import Any, Optional

import httpx

from core.enums.direction import Direction
from core.enums.market import Exchange
from core.models.position import Position

logger = logging.getLogger(__name__)

DEFAULT_GATEWAY_URL = os.getenv("TQSDK_GATEWAY_URL", "http://127.0.0.1:12891")


class TqGatewayBrokerClient:
    """Broker client that delegates all TqSdk I/O to the gateway service."""

    def __init__(self, base_url: str | None = None, timeout: float = 30.0) -> None:
        self._base_url = (base_url or DEFAULT_GATEWAY_URL).rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=timeout)
        self._connected = False

    @property
    def tqsdk_api(self) -> None:
        """Raw TqApi is intentionally unavailable outside the gateway."""
        return None

    async def connect(self) -> None:
        resp = await self._client.get("/health")
        resp.raise_for_status()
        data = resp.json()
        if not data.get("connected"):
            raise ConnectionError(
                f"TqSdk gateway at {self._base_url} is up but TqSdk session is not connected"
            )
        self._connected = True
        logger.info(
            "Connected to TqSdk gateway %s (mode=%s)",
            self._base_url,
            data.get("account_mode"),
        )

    async def disconnect(self) -> None:
        await self._client.aclose()
        self._connected = False

    async def __aenter__(self) -> "TqGatewayBrokerClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.disconnect()

    async def place_order(
        self,
        symbol: str,
        direction: Direction,
        offset: Any,
        price: Decimal,
        volume: int,
    ) -> str:
        tq_direction = "BUY" if direction == Direction.LONG else "SELL"
        resp = await self._client.post(
            "/api/v1/orders",
            json={
                "symbol": symbol,
                "direction": tq_direction,
                "offset": offset.value if hasattr(offset, "value") else str(offset),
                "price": float(price),
                "volume": volume,
            },
        )
        resp.raise_for_status()
        return resp.json()["order_id"]

    async def cancel_order(self, order_id: str) -> bool:
        resp = await self._client.delete(f"/api/v1/orders/{order_id}")
        resp.raise_for_status()
        return bool(resp.json().get("cancelled"))

    async def get_positions(self) -> list[Position]:
        resp = await self._client.get("/api/v1/positions")
        resp.raise_for_status()
        items = resp.json().get("items", [])
        result: list[Position] = []
        for row in items:
            exchange_str = row["symbol"].split(".")[0] if "." in row["symbol"] else "UNKNOWN"
            try:
                exchange = Exchange(exchange_str)
            except ValueError:
                exchange = Exchange.SHFE
            result.append(
                Position(
                    symbol=row["symbol"],
                    exchange=exchange,
                    direction=Direction.LONG if row["direction"] == "LONG" else Direction.SHORT,
                    volume=int(row["volume"]),
                    available=int(row.get("available", row["volume"])),
                    float_pnl=Decimal(str(row.get("float_pnl", 0))),
                )
            )
        return result

    async def get_account_info(self) -> dict[str, Any]:
        resp = await self._client.get("/api/v1/account")
        resp.raise_for_status()
        return resp.json()
