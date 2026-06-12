"""CSV 文件数据提供者 - 从本地CSV文件加载历史数据"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Optional

import pandas as pd

from datahub.models import OHLCV, MarketSnapshot, TickData, TimeFrame

from .base import DataProvider

logger = logging.getLogger(__name__)


class CsvProvider(DataProvider):
    """从 CSV 文件加载历史行情数据

    支持标准格式：datetime, open, high, low, close, volume [, turnover, open_interest]
    """

    def __init__(self, data_dir: str | Path):
        self._data_dir = Path(data_dir)
        self._connected = False

    @property
    def name(self) -> str:
        return "csv"

    @property
    def supported_exchanges(self) -> list[str]:
        return []

    async def connect(self) -> None:
        if not self._data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self._data_dir}")
        self._connected = True
        logger.info("CSV provider connected, data_dir=%s", self._data_dir)

    async def disconnect(self) -> None:
        self._connected = False

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: datetime,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> list[OHLCV]:
        file_path = self._resolve_file(symbol, timeframe)
        df = pd.read_csv(file_path, parse_dates=["datetime"])
        df = df.sort_values("datetime")

        mask = df["datetime"] >= pd.Timestamp(start)
        if end:
            mask &= df["datetime"] <= pd.Timestamp(end)
        df = df[mask]

        if limit:
            df = df.tail(limit)

        results: list[OHLCV] = []
        for _, row in df.iterrows():
            results.append(
                OHLCV(
                    symbol=symbol,
                    exchange="CSV",
                    timeframe=timeframe,
                    timestamp=row["datetime"].to_pydatetime(),
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"],
                    turnover=row.get("turnover"),
                    open_interest=row.get("open_interest"),
                )
            )
        return results

    async def subscribe_ticks(self, symbols: list[str]) -> AsyncIterator[TickData]:
        raise NotImplementedError("CSV provider does not support live tick subscription")
        yield  # type: ignore[misc]

    async def get_snapshot(self, symbol: str) -> MarketSnapshot:
        raise NotImplementedError("CSV provider does not support live snapshots")

    def _resolve_file(self, symbol: str, timeframe: TimeFrame) -> Path:
        candidates = [
            self._data_dir / f"{symbol}_{timeframe.value}.csv",
            self._data_dir / symbol / f"{timeframe.value}.csv",
            self._data_dir / f"{symbol}.csv",
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(
            f"No CSV file found for {symbol}/{timeframe.value} in {self._data_dir}"
        )
