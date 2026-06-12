"""BTC 数据清洗工具 — 处理异常值、缺失数据、时间对齐。"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


def remove_outlier_trades(
    trades: list[dict[str, Any]],
    price_std_threshold: float = 3.0,
) -> list[dict[str, Any]]:
    """Remove trades whose price deviates more than N stddevs from the mean."""
    if len(trades) < 10:
        return trades

    prices = [t["price"] for t in trades]
    mean = sum(prices) / len(prices)
    variance = sum((p - mean) ** 2 for p in prices) / len(prices)
    std = variance ** 0.5

    if std == 0:
        return trades

    cleaned = [
        t for t in trades
        if abs(t["price"] - mean) <= price_std_threshold * std
    ]
    removed = len(trades) - len(cleaned)
    if removed > 0:
        logger.info("Removed %d outlier trades (%.2f std threshold)", removed, price_std_threshold)
    return cleaned


def fill_missing_candles(
    candles: list[dict[str, Any]],
    interval_seconds: int = 60,
) -> list[dict[str, Any]]:
    """Forward-fill gaps in OHLCV series where candles are missing."""
    if len(candles) < 2:
        return candles

    candles_sorted = sorted(candles, key=lambda c: c["timestamp"])
    filled: list[dict[str, Any]] = [candles_sorted[0]]

    for i in range(1, len(candles_sorted)):
        prev_ts = _parse_ts(candles_sorted[i - 1]["timestamp"])
        curr_ts = _parse_ts(candles_sorted[i]["timestamp"])
        expected_ts = prev_ts + timedelta(seconds=interval_seconds)

        while expected_ts < curr_ts:
            gap_candle = {
                **candles_sorted[i - 1],
                "timestamp": expected_ts.isoformat(),
                "open": candles_sorted[i - 1]["close"],
                "high": candles_sorted[i - 1]["close"],
                "low": candles_sorted[i - 1]["close"],
                "close": candles_sorted[i - 1]["close"],
                "volume": 0.0,
            }
            filled.append(gap_candle)
            expected_ts += timedelta(seconds=interval_seconds)

        filled.append(candles_sorted[i])

    gaps = len(filled) - len(candles_sorted)
    if gaps > 0:
        logger.info("Forward-filled %d missing candles", gaps)
    return filled


def align_timestamps(
    candles: list[dict[str, Any]],
    interval_seconds: int = 60,
) -> list[dict[str, Any]]:
    """Snap candle timestamps to exact interval boundaries."""
    aligned: list[dict[str, Any]] = []
    for c in candles:
        ts = _parse_ts(c["timestamp"])
        epoch = int(ts.timestamp())
        snapped_epoch = (epoch // interval_seconds) * interval_seconds
        snapped = datetime.fromtimestamp(snapped_epoch, tz=timezone.utc)
        aligned.append({**c, "timestamp": snapped.isoformat()})
    return aligned


def deduplicate(records: list[dict[str, Any]], key: str = "timestamp") -> list[dict[str, Any]]:
    """Remove duplicate records by a key field, keeping the last occurrence."""
    seen: dict[str, dict[str, Any]] = {}
    for r in records:
        seen[r[key]] = r
    return list(seen.values())


def _parse_ts(ts_str: str) -> datetime:
    """Parse ISO timestamp string to datetime."""
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
