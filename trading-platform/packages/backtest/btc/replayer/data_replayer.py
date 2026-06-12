"""Historical data replayer for BTC backtesting.

Supports loading from Parquet files, CSV, or in-memory lists.
Yields bars in chronological order with optional downsampling.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from ..models.types import OHLCV

logger = logging.getLogger(__name__)


class DataReplayer:
    """Iterate over historical BTC OHLCV data for backtesting."""

    def __init__(
        self,
        symbol: str,
        interval: str = "1m",
    ) -> None:
        self._symbol = symbol
        self._interval = interval
        self._bars: list[OHLCV] = []
        self._loaded = False

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def bar_count(self) -> int:
        return len(self._bars)

    def load_from_list(self, bars: list[OHLCV]) -> None:
        """Load bars from an in-memory list."""
        self._bars = sorted(bars, key=lambda b: b.timestamp)
        self._loaded = True
        logger.info("Loaded %d bars for %s from memory", len(self._bars), self._symbol)

    def load_from_csv(self, path: str | Path) -> None:
        """Load OHLCV data from a CSV file.

        Expected columns: timestamp, open, high, low, close, volume
        """
        import csv

        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        bars: list[OHLCV] = []
        with open(file_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                bars.append(
                    OHLCV(
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                        open=Decimal(row["open"]),
                        high=Decimal(row["high"]),
                        low=Decimal(row["low"]),
                        close=Decimal(row["close"]),
                        volume=Decimal(row.get("volume", "0")),
                        turnover=Decimal(row.get("turnover", "0")),
                    )
                )
        self._bars = sorted(bars, key=lambda b: b.timestamp)
        self._loaded = True
        logger.info("Loaded %d bars for %s from CSV: %s", len(self._bars), self._symbol, file_path)

    def load_from_parquet(self, path: str | Path) -> None:
        """Load OHLCV data from a Parquet file.

        Requires pyarrow or fastparquet.
        """
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("pyarrow is required for Parquet support: pip install pyarrow")

        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {file_path}")

        table = pq.read_table(file_path)
        df = table.to_pandas()

        bars: list[OHLCV] = []
        for _, row in df.iterrows():
            bars.append(
                OHLCV(
                    timestamp=row["timestamp"],
                    open=Decimal(str(row["open"])),
                    high=Decimal(str(row["high"])),
                    low=Decimal(str(row["low"])),
                    close=Decimal(str(row["close"])),
                    volume=Decimal(str(row.get("volume", 0))),
                    turnover=Decimal(str(row.get("turnover", 0))),
                )
            )
        self._bars = sorted(bars, key=lambda b: b.timestamp)
        self._loaded = True
        logger.info("Loaded %d bars for %s from Parquet: %s", len(self._bars), self._symbol, file_path)

    def replay(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> Iterator[OHLCV]:
        """Yield bars in chronological order within the given time range."""
        if not self._loaded:
            raise RuntimeError("No data loaded. Call load_from_* first.")

        for bar in self._bars:
            if start and bar.timestamp < start:
                continue
            if end and bar.timestamp > end:
                break
            yield bar

    async def areplay(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> AsyncIterator[OHLCV]:
        """Async version of replay for compatibility with async engines."""
        for bar in self.replay(start, end):
            yield bar

    def slice(self, start: datetime, end: datetime) -> list[OHLCV]:
        """Return a list of bars within the time range."""
        return list(self.replay(start, end))

    def resample(self, target_interval: str) -> list[OHLCV]:
        """Downsample bars to a coarser interval.

        Currently supports: 1m -> 5m, 15m, 1h, 4h, 1d
        """
        multiplier = _interval_to_minutes(target_interval) // _interval_to_minutes(self._interval)
        if multiplier <= 1:
            return list(self._bars)

        resampled: list[OHLCV] = []
        for i in range(0, len(self._bars), multiplier):
            chunk = self._bars[i : i + multiplier]
            if not chunk:
                break
            resampled.append(
                OHLCV(
                    timestamp=chunk[0].timestamp,
                    open=chunk[0].open,
                    high=max(b.high for b in chunk),
                    low=min(b.low for b in chunk),
                    close=chunk[-1].close,
                    volume=sum(b.volume for b in chunk),
                    turnover=sum(b.turnover for b in chunk),
                )
            )
        return resampled


def _interval_to_minutes(interval: str) -> int:
    """Convert interval string to minutes."""
    mapping: dict[str, int] = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
    }
    result = mapping.get(interval)
    if result is None:
        raise ValueError(f"Unsupported interval: {interval}. Supported: {list(mapping.keys())}")
    return result
