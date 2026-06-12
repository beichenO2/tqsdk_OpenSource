"""Parquet 分层存储 — 将 BTC 行情数据持久化为 Parquet 文件。

分层策略:
  - raw/     原始采集数据
  - clean/   清洗后的数据
  - feature/ 特征工程结果
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    _HAS_ARROW = True
except ImportError:
    _HAS_ARROW = False


class ParquetStorage:
    """Write & read BTC market data in Parquet columnar format."""

    def __init__(self, base_dir: str | Path) -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    # ── OHLCV ───────────────────────────────────────────────────────

    _OHLCV_SCHEMA = {
        "exchange": str,
        "symbol": str,
        "interval": str,
        "open": float,
        "high": float,
        "low": float,
        "close": float,
        "volume": float,
        "timestamp": str,
    }

    def write_ohlcv(
        self,
        records: list[dict[str, Any]],
        exchange: str,
        symbol: str,
        interval: str,
        layer: str = "raw",
    ) -> Path:
        """Write OHLCV records to a partitioned Parquet file."""
        if not _HAS_ARROW:
            return self._write_json_fallback(records, exchange, symbol, interval, layer)

        dir_path = self._base / layer / "ohlcv" / exchange / symbol / interval
        dir_path.mkdir(parents=True, exist_ok=True)

        table = pa.Table.from_pylist(records)
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        file_path = dir_path / f"{ts}.parquet"
        pq.write_table(table, file_path, compression="snappy")
        logger.info("Wrote %d OHLCV rows → %s", len(records), file_path)
        return file_path

    def read_ohlcv(
        self,
        exchange: str,
        symbol: str,
        interval: str,
        layer: str = "raw",
    ) -> list[dict[str, Any]]:
        """Read all OHLCV data for a given partition."""
        if not _HAS_ARROW:
            return self._read_json_fallback(exchange, symbol, interval, layer)

        dir_path = self._base / layer / "ohlcv" / exchange / symbol / interval
        if not dir_path.exists():
            return []

        tables = []
        for f in sorted(dir_path.glob("*.parquet")):
            tables.append(pq.read_table(f))

        if not tables:
            return []

        merged = pa.concat_tables(tables)
        return merged.to_pylist()

    # ── Trades ──────────────────────────────────────────────────────

    def write_trades(
        self,
        records: list[dict[str, Any]],
        exchange: str,
        symbol: str,
        layer: str = "raw",
    ) -> Path:
        dir_path = self._base / layer / "trades" / exchange / symbol
        dir_path.mkdir(parents=True, exist_ok=True)

        if not _HAS_ARROW:
            import json
            ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            file_path = dir_path / f"{ts}.json"
            file_path.write_text(json.dumps(records, default=str))
            return file_path

        table = pa.Table.from_pylist(records)
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        file_path = dir_path / f"{ts}.parquet"
        pq.write_table(table, file_path, compression="snappy")
        logger.info("Wrote %d trade rows → %s", len(records), file_path)
        return file_path

    # ── Orderbook snapshots ─────────────────────────────────────────

    def write_orderbook_snapshot(
        self,
        snapshot: dict[str, Any],
        exchange: str,
        symbol: str,
        layer: str = "raw",
    ) -> Path:
        import json

        dir_path = self._base / layer / "orderbook" / exchange / symbol
        dir_path.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        file_path = dir_path / f"{ts}.json"
        file_path.write_text(json.dumps(snapshot, default=str))
        return file_path

    # ── Fallbacks (when pyarrow unavailable) ────────────────────────

    def _write_json_fallback(
        self,
        records: list[dict[str, Any]],
        exchange: str,
        symbol: str,
        interval: str,
        layer: str,
    ) -> Path:
        import json

        dir_path = self._base / layer / "ohlcv" / exchange / symbol / interval
        dir_path.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        file_path = dir_path / f"{ts}.json"
        file_path.write_text(json.dumps(records, default=str))
        logger.warning("pyarrow not installed; fell back to JSON: %s", file_path)
        return file_path

    def _read_json_fallback(
        self,
        exchange: str,
        symbol: str,
        interval: str,
        layer: str,
    ) -> list[dict[str, Any]]:
        import json

        dir_path = self._base / layer / "ohlcv" / exchange / symbol / interval
        if not dir_path.exists():
            return []
        records: list[dict[str, Any]] = []
        for f in sorted(dir_path.glob("*.json")):
            records.extend(json.loads(f.read_text()))
        return records
