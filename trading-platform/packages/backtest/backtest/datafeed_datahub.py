"""DataHub 适配的 DataFeed - 从分层存储引擎读取历史K线数据。

对接 Ch28 的 datahub.StorageEngine，将 Parquet/DuckDB 中的 OHLCV
数据转换为回测引擎需要的 Bar 序列。
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from .datafeed import DataFeed
from .events import Event, EventBus, EventType
from .models import Bar

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class DataHubFeed(DataFeed):
    """从 DataHub StorageEngine 读取数据的 DataFeed 实现。

    Usage::

        from datahub.storage import StorageEngine, StorageLayer

        storage = StorageEngine(base_dir="./data")
        feed = DataHubFeed(event_bus, storage, layer=StorageLayer.GOLD)
    """

    def __init__(
        self,
        event_bus: EventBus,
        storage_engine: object,
        layer: object | None = None,
    ) -> None:
        super().__init__(event_bus)
        self._storage = storage_engine
        self._layer = layer
        self._bars: list[Bar] = []
        self._index = 0

    def load(self, symbols: list[str], start: datetime, end: datetime) -> None:
        self._bars = []
        for symbol in symbols:
            try:
                df = self._storage.read_ohlcv(  # type: ignore[attr-defined]
                    symbol=symbol,
                    timeframe=self._resolve_timeframe(),
                    layer=self._layer,
                    start=start,
                    end=end,
                )
                self._bars.extend(self._df_to_bars(df, symbol))
            except FileNotFoundError:
                logger.warning("No data found for %s, skipping", symbol)

        self._bars.sort(key=lambda b: b.dt)
        self._index = 0
        logger.info("DataHubFeed loaded %d bars for %s", len(self._bars), symbols)

    def _resolve_timeframe(self) -> object:
        """延迟导入 TimeFrame 避免硬依赖。"""
        try:
            from datahub.models import TimeFrame
            return TimeFrame.M1
        except ImportError:
            return "1m"

    @staticmethod
    def _df_to_bars(df: pd.DataFrame, fallback_symbol: str) -> list[Bar]:
        bars: list[Bar] = []
        for _, row in df.iterrows():
            bars.append(
                Bar(
                    symbol=row.get("symbol", fallback_symbol),
                    dt=row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"],
                    open=Decimal(str(row["open"])),
                    high=Decimal(str(row["high"])),
                    low=Decimal(str(row["low"])),
                    close=Decimal(str(row["close"])),
                    volume=int(row["volume"]),
                    open_interest=int(row.get("open_interest", 0) or 0),
                )
            )
        return bars

    def __iter__(self) -> Iterator[Bar]:
        self._index = 0
        return self

    def __next__(self) -> Bar:
        if self._index >= len(self._bars):
            raise StopIteration
        bar = self._bars[self._index]
        self._index += 1
        self._event_bus.publish(Event(type=EventType.BAR, data=bar, dt=bar.dt, source="datahub_feed"))
        return bar

    def __len__(self) -> int:
        return len(self._bars)
