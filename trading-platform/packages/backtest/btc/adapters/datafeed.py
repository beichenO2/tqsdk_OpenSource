"""BTC DataFeed — bridges Ch29's DataFeed ABC with Ch32's data pipeline.

Supports loading from:
  - Ch32's ParquetStorage (preferred for large datasets)
  - CSV files
  - In-memory bar lists
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from backtest.datafeed import DataFeed
from backtest.events import Event, EventBus, EventType
from backtest.models import Bar

logger = logging.getLogger(__name__)


class BTCDataFeed(DataFeed):
    """K-line data feed for BTC backtesting.

    Wraps Ch32's data storage into Ch29's DataFeed interface so the
    standard BacktestEngine can drive BTC backtests.
    """

    def __init__(self, event_bus: EventBus) -> None:
        super().__init__(event_bus)
        self._bars: list[Bar] = []
        self._index = 0

    def load(self, symbols: list[str], start: datetime, end: datetime) -> None:
        self._bars = sorted(
            [b for b in self._bars if start <= b.dt <= end and b.symbol in symbols],
            key=lambda b: b.dt,
        )
        self._index = 0
        logger.info("BTCDataFeed: loaded %d bars for %s (%s ~ %s)", len(self._bars), symbols, start, end)

    def load_from_parquet(self, storage_dir: str, exchange: str, symbol: str, interval: str = "1m") -> None:
        """Load data from Ch32's ParquetStorage format."""
        try:
            from datahub.btc.storage import ParquetStorage
            storage = ParquetStorage(storage_dir)
            candles = storage.read_ohlcv(exchange=exchange, symbol=symbol, interval=interval)
            self._bars.extend(
                Bar(
                    symbol=symbol,
                    dt=c["timestamp"] if isinstance(c["timestamp"], datetime) else datetime.fromisoformat(str(c["timestamp"])),
                    open=Decimal(str(c["open"])),
                    high=Decimal(str(c["high"])),
                    low=Decimal(str(c["low"])),
                    close=Decimal(str(c["close"])),
                    volume=int(c.get("volume", 0)),
                )
                for c in candles
            )
            logger.info("Loaded %d bars from Parquet: %s/%s", len(candles), exchange, symbol)
        except ImportError:
            logger.warning("datahub.btc.storage not available; use load_from_csv or add_bars instead")
            raise

    def load_from_csv(self, path: str | Path, symbol: str) -> None:
        """Load OHLCV data from a CSV file."""
        import csv
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV not found: {file_path}")

        with open(file_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._bars.append(
                    Bar(
                        symbol=symbol,
                        dt=datetime.fromisoformat(row["timestamp"]),
                        open=Decimal(row["open"]),
                        high=Decimal(row["high"]),
                        low=Decimal(row["low"]),
                        close=Decimal(row["close"]),
                        volume=int(row.get("volume", "0")),
                    )
                )
        logger.info("Loaded %d bars from CSV for %s", len(self._bars), symbol)

    def add_bars(self, bars: list[Bar]) -> None:
        """Add bars directly from memory (for testing or custom data sources)."""
        self._bars.extend(bars)

    def __iter__(self) -> Iterator[Bar]:
        self._index = 0
        return self

    def __next__(self) -> Bar:
        if self._index >= len(self._bars):
            raise StopIteration
        bar = self._bars[self._index]
        self._index += 1
        self._event_bus.publish(Event(type=EventType.BAR, data=bar, dt=bar.dt, source="btc_datafeed"))
        return bar

    def __len__(self) -> int:
        return len(self._bars)
