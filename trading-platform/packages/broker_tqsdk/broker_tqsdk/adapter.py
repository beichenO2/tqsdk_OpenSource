"""TqSdk 行情适配器 — 将 TqSdk 行情转换为统一的 Bar/Tick 模型."""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, AsyncIterator

from core.models.bar import Bar
from core.models.tick import Tick

logger = logging.getLogger(__name__)


class TqMarketAdapter:
    """行情适配层：从 TqSdk 获取 tick / kline 并转为核心模型."""

    def __init__(self, api: Any = None) -> None:
        self._api = api

    def set_api(self, api: Any) -> None:
        self._api = api

    async def get_quote(self, symbol: str) -> Tick | None:
        """获取最新行情快照."""
        if self._api is None:
            return None
        quote = self._api.get_quote(symbol)
        return Tick(
            symbol=symbol,
            datetime=datetime.fromtimestamp(quote.datetime / 1e9),
            last_price=Decimal(str(quote.last_price)),
            highest=Decimal(str(quote.highest)),
            lowest=Decimal(str(quote.lowest)),
            volume=int(quote.volume),
            amount=Decimal(str(quote.amount)),
            open_interest=int(quote.open_interest) if quote.open_interest else None,
            bid_price1=Decimal(str(quote.bid_price1)) if quote.bid_price1 else None,
            bid_volume1=int(quote.bid_volume1) if quote.bid_volume1 else None,
            ask_price1=Decimal(str(quote.ask_price1)) if quote.ask_price1 else None,
            ask_volume1=int(quote.ask_volume1) if quote.ask_volume1 else None,
        )

    async def get_klines(
        self, symbol: str, duration_seconds: int = 60, data_length: int = 200
    ) -> list[Bar]:
        """获取 K 线数据."""
        if self._api is None:
            return []
        klines = self._api.get_kline_serial(symbol, duration_seconds, data_length)
        bars: list[Bar] = []
        for _, row in klines.iterrows():
            bars.append(Bar(
                symbol=symbol,
                datetime=datetime.fromtimestamp(row["datetime"] / 1e9),
                open=Decimal(str(row["open"])),
                high=Decimal(str(row["high"])),
                low=Decimal(str(row["low"])),
                close=Decimal(str(row["close"])),
                volume=int(row["volume"]),
                open_interest=int(row["close_oi"]) if "close_oi" in row else None,
                duration_seconds=duration_seconds,
            ))
        return bars

    async def list_instruments(
        self, exchange_id: str | None = None, ins_class: str = "FUTURE"
    ) -> list[dict[str, str]]:
        """列出合约代码（TqSdk query_quotes）."""
        if self._api is None:
            return []
        kwargs: dict[str, Any] = {"ins_class": ins_class}
        if exchange_id:
            kwargs["exchange_id"] = exchange_id
        try:
            symbols = self._api.query_quotes(**kwargs)
        except Exception:
            logger.exception("query_quotes failed")
            return []
        return [{"symbol": s} for s in symbols]

    async def subscribe_ticks(self, symbol: str) -> AsyncIterator[Tick]:
        """订阅逐笔行情（异步迭代器）."""
        if self._api is None:
            return
        ticks = self._api.get_tick_serial(symbol)
        last_idx = len(ticks) - 1
        while True:
            if len(ticks) > last_idx + 1:
                for i in range(last_idx + 1, len(ticks)):
                    row = ticks.iloc[i]
                    yield Tick(
                        symbol=symbol,
                        datetime=datetime.fromtimestamp(row["datetime"] / 1e9),
                        last_price=Decimal(str(row["last_price"])),
                        highest=Decimal(str(row["highest"])),
                        lowest=Decimal(str(row["lowest"])),
                        volume=int(row["volume"]),
                        amount=Decimal(str(row.get("amount", 0))),
                    )
                last_idx = len(ticks) - 1
            self._api.wait_update()
