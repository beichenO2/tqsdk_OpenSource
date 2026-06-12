"""Tick buffer with periodic flush to daily Parquet files.

Storage layout:
  {DATA_ROOT}/tick/{source}/{YYYY-MM-DD}/{symbol}.parquet

Each flush appends to the day's file (read-merge-write since Parquet
doesn't support native append).  Memory buffer is capped per symbol
to bound RAM usage.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, date
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]

DATA_ROOT = Path(os.getenv("DATA_ROOT", os.path.expanduser("~/Polarisor/tqsdk/trading-platform/data")))
TICK_ROOT = DATA_ROOT / "tick"

FLUSH_INTERVAL_SEC = 60
MAX_BUFFER_PER_SYMBOL = 50_000


class TickBuffer:
    """Thread-safe tick buffer that flushes to per-day, per-symbol Parquet."""

    def __init__(self, source: str, flush_interval: float = FLUSH_INTERVAL_SEC) -> None:
        """
        Args:
            source: sub-directory name under tick/ (e.g. "futures" or "crypto").
            flush_interval: seconds between automatic flushes.
        """
        self._source = source
        self._flush_interval = flush_interval
        self._lock = threading.Lock()
        self._buffers: dict[str, list[dict]] = {}  # symbol -> rows
        self._running = False
        self._flush_thread: threading.Thread | None = None
        self._stats = {"total_ticks": 0, "total_flushes": 0, "errors": 0}

    # ── public API ──────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()
        logger.info("tick buffer started: source=%s, flush_interval=%ss", self._source, self._flush_interval)

    def stop(self) -> None:
        self._running = False
        if self._flush_thread:
            self._flush_thread.join(timeout=10)
        self.flush_all()
        logger.info("tick buffer stopped: %s", self._stats)

    def record(self, symbol: str, row: dict) -> None:
        """Append a single tick row for *symbol*."""
        with self._lock:
            buf = self._buffers.setdefault(symbol, [])
            buf.append(row)
            self._stats["total_ticks"] += 1
            if len(buf) >= MAX_BUFFER_PER_SYMBOL:
                self._flush_symbol(symbol)

    def record_batch(self, symbol: str, rows: list[dict]) -> None:
        with self._lock:
            buf = self._buffers.setdefault(symbol, [])
            buf.extend(rows)
            self._stats["total_ticks"] += len(rows)
            if len(buf) >= MAX_BUFFER_PER_SYMBOL:
                self._flush_symbol(symbol)

    def flush_all(self) -> None:
        with self._lock:
            for sym in list(self._buffers.keys()):
                self._flush_symbol(sym)

    def get_stats(self) -> dict:
        with self._lock:
            buffered = sum(len(v) for v in self._buffers.values())
        return {**self._stats, "buffered": buffered, "symbols": len(self._buffers)}

    # ── internal ────────────────────────────────────────────────

    def _flush_loop(self) -> None:
        while self._running:
            time.sleep(self._flush_interval)
            self.flush_all()

    def _flush_symbol(self, symbol: str) -> None:
        """Flush buffer for one symbol. Caller must hold _lock."""
        rows = self._buffers.pop(symbol, [])
        if not rows or pd is None:
            return

        new_df = pd.DataFrame(rows)

        today_str = date.today().isoformat()
        out_dir = TICK_ROOT / self._source / today_str
        out_dir.mkdir(parents=True, exist_ok=True)

        safe_name = symbol.replace("@", "_").replace(".", "_")
        path = out_dir / f"{safe_name}.parquet"

        try:
            if path.exists():
                existing = pd.read_parquet(path)
                merged = pd.concat([existing, new_df], ignore_index=True)
            else:
                merged = new_df

            merged.to_parquet(path, index=False)
            self._stats["total_flushes"] += 1
            logger.debug("flushed %d ticks for %s → %s (total %d)", len(rows), symbol, path.name, len(merged))
        except Exception as e:
            self._stats["errors"] += 1
            logger.error("flush error for %s: %s", symbol, e)
            self._buffers.setdefault(symbol, []).extend(rows)
