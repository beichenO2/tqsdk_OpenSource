"""Market data via TqSdk Gateway HTTP API."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from decimal import Decimal
from typing import Any, AsyncIterator

import httpx

from core.models.bar import Bar
from core.models.tick import Tick

logger = logging.getLogger(__name__)

DEFAULT_GATEWAY_URL = os.getenv("TQSDK_GATEWAY_URL", "http://127.0.0.1:12890")


class TqGatewayMarketAdapter:
    """Fetch quotes and klines from TqSdk Gateway — no local TqApi."""

    def __init__(self, base_url: str | None = None, timeout: float = 30.0) -> None:
        self._base_url = (base_url or DEFAULT_GATEWAY_URL).rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=timeout)

    def set_api(self, _api: Any = None) -> None:
        """Compatibility no-op — gateway adapter does not use raw TqApi."""

    async def aclose(self) -> None:
        await self._client.aclose()

    async def get_quote(self, symbol: str) -> Tick | None:
        try:
            resp = await self._client.get(f"/api/v1/market/quote/{symbol}")
            resp.raise_for_status()
            q = resp.json()
        except Exception:
            logger.exception("gateway get_quote failed for %s", symbol)
            return None
        raw_dt = q.get("datetime")
        if raw_dt:
            tick_dt = datetime.fromtimestamp(int(raw_dt) / 1e9)
        else:
            tick_dt = datetime.now()
        return Tick(
            symbol=symbol,
            datetime=tick_dt,
            last_price=Decimal(str(q["last_price"])),
            highest=Decimal(str(q["highest"])),
            lowest=Decimal(str(q["lowest"])),
            volume=int(q["volume"]),
            amount=Decimal(str(q["amount"])),
            open_interest=int(q.get("open_interest") or 0) or None,
            bid_price1=Decimal(str(q["bid_price1"])) if q.get("bid_price1") is not None else None,
            bid_volume1=int(q["bid_volume1"]) if q.get("bid_volume1") is not None else None,
            ask_price1=Decimal(str(q["ask_price1"])) if q.get("ask_price1") is not None else None,
            ask_volume1=int(q["ask_volume1"]) if q.get("ask_volume1") is not None else None,
        )

    async def get_klines(
        self, symbol: str, duration_seconds: int = 60, data_length: int = 200
    ) -> list[Bar]:
        try:
            resp = await self._client.get(
                f"/api/v1/market/klines/{symbol}",
                params={"duration": duration_seconds, "length": data_length},
            )
            resp.raise_for_status()
            rows = resp.json().get("items", [])
        except Exception:
            logger.exception("gateway get_klines failed for %s", symbol)
            return []
        bars: list[Bar] = []
        for row in rows:
            # Right after gateway (re)start the newest kline rows may carry
            # null/NaN OHLC before TqSdk fills them in — skip instead of 500.
            ohlc = [row.get(k) for k in ("open", "high", "low", "close")]
            if row.get("datetime") is None or any(
                v is None or v != v for v in ohlc
            ):
                continue
            bars.append(
                Bar(
                    symbol=symbol,
                    datetime=datetime.fromtimestamp(row["datetime"] / 1e9),
                    open=Decimal(str(row["open"])),
                    high=Decimal(str(row["high"])),
                    low=Decimal(str(row["low"])),
                    close=Decimal(str(row["close"])),
                    volume=int(row["volume"]),
                    open_interest=int(row["open_interest"]) if row.get("open_interest") else None,
                    duration_seconds=duration_seconds,
                )
            )
        return bars

    async def list_instruments(
        self, exchange_id: str | None = None, ins_class: str = "FUTURE"
    ) -> list[dict[str, str]]:
        try:
            params: dict[str, str] = {"ins_class": ins_class}
            if exchange_id:
                params["exchange_id"] = exchange_id
            resp = await self._client.get("/api/v1/market/instruments", params=params)
            resp.raise_for_status()
            return resp.json().get("items", [])
        except Exception:
            logger.exception("gateway list_instruments failed")
            return []

    async def subscribe_ticks(self, symbol: str) -> AsyncIterator[Tick]:
        if False:  # pragma: no cover — async generator stub
            yield Tick(symbol=symbol, datetime=datetime.utcnow(), last_price=Decimal(0))
        raise NotImplementedError("Use TqGatewayLiveFeed for streaming via gateway polling")
