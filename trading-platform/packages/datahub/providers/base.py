"""数据提供者抽象基类"""

from __future__ import annotations

import abc
from datetime import datetime
from typing import AsyncIterator, Optional

from datahub.models import OHLCV, MarketSnapshot, TickData, TimeFrame


class DataProvider(abc.ABC):
    """数据源适配器基类

    所有具体数据源（TqSdk、交易所 API、CSV 文件等）都实现此接口。
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """数据源名称标识"""

    @property
    @abc.abstractmethod
    def supported_exchanges(self) -> list[str]:
        """支持的交易所列表"""

    @abc.abstractmethod
    async def connect(self) -> None:
        """建立连接"""

    @abc.abstractmethod
    async def disconnect(self) -> None:
        """断开连接"""

    @abc.abstractmethod
    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: datetime,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> list[OHLCV]:
        """获取历史K线数据"""

    @abc.abstractmethod
    async def subscribe_ticks(
        self,
        symbols: list[str],
    ) -> AsyncIterator[TickData]:
        """订阅实时逐笔行情"""

    @abc.abstractmethod
    async def get_snapshot(self, symbol: str) -> MarketSnapshot:
        """获取最新市场快照"""

    async def get_available_symbols(self, exchange: str) -> list[str]:
        """获取可用合约/交易对列表（可选实现）"""
        raise NotImplementedError(
            f"{self.name} does not support listing symbols"
        )

    async def __aenter__(self) -> DataProvider:
        await self.connect()
        return self

    async def __aexit__(self, *args) -> None:
        await self.disconnect()
