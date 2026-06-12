"""数据馈送 - 将历史数据按时间顺序喂给回测引擎。"""

from __future__ import annotations

import abc
import logging
from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
from typing import Any

from .events import Event, EventBus, EventType
from .models import Bar, Tick

logger = logging.getLogger(__name__)


class DataFeed(abc.ABC):
    """数据馈送抽象基类。"""

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus

    @abc.abstractmethod
    def load(self, symbols: list[str], start: datetime, end: datetime) -> None:
        """加载指定时间范围的数据。"""

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Bar | Tick]:
        """按时间顺序迭代所有数据。"""

    @abc.abstractmethod
    def __len__(self) -> int:
        """数据总条数。"""


class BarDataFeed(DataFeed):
    """K线数据馈送 - 支持从内存列表加载。"""

    def __init__(self, event_bus: EventBus) -> None:
        super().__init__(event_bus)
        self._bars: list[Bar] = []
        self._index = 0

    def load(self, symbols: list[str], start: datetime, end: datetime) -> None:
        """筛选并排序已添加的Bar数据。"""
        self._bars = sorted(
            [b for b in self._bars if start <= b.dt <= end and b.symbol in symbols],
            key=lambda b: b.dt,
        )
        self._index = 0
        logger.info("Loaded %d bars for %s", len(self._bars), symbols)

    def add_bars(self, bars: list[Bar]) -> None:
        """添加K线数据。"""
        self._bars.extend(bars)

    @classmethod
    def from_dicts(cls, event_bus: EventBus, records: list[dict[str, Any]]) -> BarDataFeed:
        """从字典列表构造。"""
        feed = cls(event_bus)
        bars = [
            Bar(
                symbol=r["symbol"],
                dt=r["dt"] if isinstance(r["dt"], datetime) else datetime.fromisoformat(r["dt"]),
                open=Decimal(str(r["open"])),
                high=Decimal(str(r["high"])),
                low=Decimal(str(r["low"])),
                close=Decimal(str(r["close"])),
                volume=int(r["volume"]),
                open_interest=int(r.get("open_interest", 0)),
            )
            for r in records
        ]
        feed.add_bars(bars)
        return feed

    def __iter__(self) -> Iterator[Bar]:
        self._index = 0
        return self

    def __next__(self) -> Bar:
        if self._index >= len(self._bars):
            raise StopIteration
        bar = self._bars[self._index]
        self._index += 1
        self._event_bus.publish(Event(type=EventType.BAR, data=bar, dt=bar.dt, source="datafeed"))
        return bar

    def __len__(self) -> int:
        return len(self._bars)
