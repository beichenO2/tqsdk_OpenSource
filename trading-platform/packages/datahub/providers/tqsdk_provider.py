"""TqSdk 数据提供者 - 对接天勤量化的行情数据"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import AsyncIterator, Optional

from datahub.models import OHLCV, MarketSnapshot, TickData, TimeFrame

from .base import DataProvider

logger = logging.getLogger(__name__)

TIMEFRAME_MAP: dict[TimeFrame, int] = {
    TimeFrame.M1: 60,
    TimeFrame.M5: 300,
    TimeFrame.M15: 900,
    TimeFrame.M30: 1800,
    TimeFrame.H1: 3600,
    TimeFrame.H4: 14400,
    TimeFrame.D1: 86400,
}


class TqSdkProvider(DataProvider):
    """通过 TqSdk 获取国内期货行情

    需要在 worker 进程中运行 TqApi 事件循环，
    本 provider 通过队列与 TqApi 通信。
    """

    def __init__(
        self,
        auth_account: Optional[str] = None,
        auth_password: Optional[str] = None,
    ):
        self._auth_account = auth_account
        self._auth_password = auth_password
        self._api = None
        self._connected = False

    @property
    def name(self) -> str:
        return "tqsdk"

    @property
    def supported_exchanges(self) -> list[str]:
        return ["SHFE", "DCE", "CZCE", "INE", "CFFEX", "GFEX"]

    async def connect(self) -> None:
        try:
            from tqsdk import TqApi, TqAuth

            auth = None
            if self._auth_account and self._auth_password:
                auth = TqAuth(self._auth_account, self._auth_password)

            self._api = TqApi(auth=auth)
            self._connected = True
            logger.info("TqSdk provider connected")
        except ImportError:
            raise RuntimeError(
                "tqsdk is not installed. Install it with: pip install tqsdk"
            )

    async def disconnect(self) -> None:
        if self._api is not None:
            self._api.close()
            self._api = None
            self._connected = False
            logger.info("TqSdk provider disconnected")

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: datetime,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> list[OHLCV]:
        self._ensure_connected()
        duration_seconds = TIMEFRAME_MAP.get(timeframe)
        if duration_seconds is None:
            raise ValueError(f"TqSdk does not support timeframe {timeframe}")

        data_length = limit or 8964
        klines = await asyncio.to_thread(
            self._api.get_kline_serial, symbol, duration_seconds, data_length
        )

        result: list[OHLCV] = []
        for _, row in klines.iterrows():
            ts = datetime.fromtimestamp(row["datetime"] / 1e9)
            if ts < start:
                continue
            if end and ts > end:
                break
            bar = OHLCV(
                symbol=symbol,
                exchange=self._parse_exchange(symbol),
                timeframe=timeframe,
                timestamp=ts,
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                open_interest=row.get("close_oi"),
            )
            result.append(bar)
        return result

    async def subscribe_ticks(
        self,
        symbols: list[str],
    ) -> AsyncIterator[TickData]:
        self._ensure_connected()
        quotes = {s: await asyncio.to_thread(self._api.get_quote, s) for s in symbols}

        while self._connected:
            await asyncio.to_thread(self._api.wait_update)
            for sym, quote in quotes.items():
                if self._api.is_changing(quote):
                    yield TickData(
                        symbol=sym,
                        exchange=self._parse_exchange(sym),
                        timestamp=datetime.fromtimestamp(
                            quote.datetime / 1e9
                        ),
                        last_price=quote.last_price,
                        volume=quote.volume,
                        bid_price_1=quote.bid_price1,
                        bid_volume_1=quote.bid_volume1,
                        ask_price_1=quote.ask_price1,
                        ask_volume_1=quote.ask_volume1,
                        open_interest=quote.open_interest,
                    )

    async def get_snapshot(self, symbol: str) -> MarketSnapshot:
        self._ensure_connected()
        quote = await asyncio.to_thread(self._api.get_quote, symbol)
        await asyncio.to_thread(self._api.wait_update)

        return MarketSnapshot(
            symbol=symbol,
            exchange=self._parse_exchange(symbol),
            timestamp=datetime.fromtimestamp(quote.datetime / 1e9),
            last_price=quote.last_price,
            open=quote.open,
            high=quote.highest,
            low=quote.lowest,
            pre_close=quote.pre_close,
            volume=quote.volume,
            turnover=quote.amount,
            open_interest=quote.open_interest,
            upper_limit=quote.upper_limit,
            lower_limit=quote.lower_limit,
        )

    async def get_available_symbols(self, exchange: str) -> list[str]:
        self._ensure_connected()
        ls = await asyncio.to_thread(self._api.query_quotes, ins_class="FUTURE", exchange_id=exchange)
        return list(ls)

    def _ensure_connected(self) -> None:
        if not self._connected or self._api is None:
            raise RuntimeError("TqSdk provider is not connected")

    @staticmethod
    def _parse_exchange(symbol: str) -> str:
        parts = symbol.split(".")
        return parts[0] if len(parts) > 1 else "UNKNOWN"
