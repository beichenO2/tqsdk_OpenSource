"""Download high-resolution (1m + 5m) BTC/ETH data from Binance."""

from __future__ import annotations

import os as _os
import sys as _sys

try:
    from polarisor_port_sdk import submit_task as _sdk_submit, complete_task as _sdk_complete
except ImportError:
    _sdk_submit = _sdk_complete = None


import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://api.binance.com/api/v3"
DATA_DIR = Path(os.path.expanduser("~/Downloads/crypto_data"))

TASKS = [
    ("BTCUSDT", "5m", datetime(2020, 1, 1, tzinfo=timezone.utc)),
    ("BTCUSDT", "1m", datetime(2024, 1, 1, tzinfo=timezone.utc)),
    ("ETHUSDT", "5m", datetime(2020, 1, 1, tzinfo=timezone.utc)),
    ("ETHUSDT", "15m", datetime(2020, 1, 1, tzinfo=timezone.utc)),
    ("ETHUSDT", "1m", datetime(2024, 1, 1, tzinfo=timezone.utc)),
    ("SOLUSDT", "5m", datetime(2020, 9, 1, tzinfo=timezone.utc)),
    ("SOLUSDT", "1h", datetime(2020, 9, 1, tzinfo=timezone.utc)),
    ("SOLUSDT", "4h", datetime(2020, 9, 1, tzinfo=timezone.utc)),
    ("SOLUSDT", "1d", datetime(2020, 9, 1, tzinfo=timezone.utc)),
]

TF_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000,
    "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}


async def download_one(
    session: aiohttp.ClientSession,
    symbol: str, interval: str, start_dt: datetime,
) -> None:
    out_dir = DATA_DIR / symbol.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{interval}.parquet"
    tf_ms = TF_MS[interval]

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    if out_path.exists():
        existing = pd.read_parquet(out_path)
        if len(existing) > 0:
            last_ts = int(existing["open_time"].max().timestamp() * 1000)
            start_ms = last_ts + tf_ms
            logger.info("Resuming %s %s from %d existing rows", symbol, interval, len(existing))
    else:
        existing = None

    if start_ms >= end_ms:
        logger.info("Already up-to-date: %s %s", symbol, interval)
        return

    chunks: list[pd.DataFrame] = []
    cursor = start_ms
    batch = 0

    while cursor < end_ms:
        params = {
            "symbol": symbol, "interval": interval,
            "startTime": cursor, "endTime": end_ms, "limit": 1000,
        }
        async with session.get(f"{BASE_URL}/klines", params=params) as resp:
            if resp.status == 429:
                logger.warning("Rate limited, sleeping 60s")
                await asyncio.sleep(60)
                continue
            resp.raise_for_status()
            raw = await resp.json()

        if not raw:
            break

        cols = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_volume",
            "taker_buy_quote_volume", "ignore",
        ]
        df = pd.DataFrame(raw, columns=cols)
        for c in ["open", "high", "low", "close", "volume", "quote_volume",
                   "taker_buy_volume", "taker_buy_quote_volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        df["trades"] = df["trades"].astype(int)
        df.drop(columns=["ignore"], inplace=True)
        chunks.append(df)

        cursor = int(df["open_time"].iloc[-1].timestamp() * 1000) + tf_ms
        batch += 1

        if batch % 100 == 0:
            rows = sum(len(c) for c in chunks)
            pct = min((cursor - start_ms) / (end_ms - start_ms) * 100, 100)
            logger.info("  %s %s: %d rows (%.0f%%)", symbol, interval, rows, pct)

        await asyncio.sleep(0.12)

    if not chunks:
        logger.info("No new data: %s %s", symbol, interval)
        return

    new_df = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["open_time"], keep="last")
    if existing is not None:
        final = pd.concat([existing, new_df], ignore_index=True).drop_duplicates(subset=["open_time"], keep="last")
    else:
        final = new_df

    final.sort_values("open_time", inplace=True)
    final.reset_index(drop=True, inplace=True)
    final.to_parquet(out_path, index=False)
    logger.info(
        "Saved %s %s: %d rows [%s → %s]",
        symbol, interval, len(final),
        final["open_time"].iloc[0].strftime("%Y-%m-%d"),
        final["open_time"].iloc[-1].strftime("%Y-%m-%d"),
    )


async def main() -> None:

    _task_id = None
    if _sdk_submit:
        try:
            _tr = _sdk_submit(task_type="data-download", command="download_hires.py", requester="download-hires", estimated_duration_sec=600)
            _task_id = _tr.get("task_id")
        except Exception:
            pass
    logger.info("High-resolution data download starting...")
    connector = aiohttp.TCPConnector(limit=3)
    timeout = aiohttp.ClientTimeout(total=180)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for symbol, interval, start_dt in TASKS:
            try:
                await download_one(session, symbol, interval, start_dt)
            except Exception as e:
                logger.error("Failed %s %s: %s", symbol, interval, e)

    logger.info("Download complete!")
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir():
            for f in sorted(d.glob("*.parquet")):
                df = pd.read_parquet(f)
                logger.info("  %s/%s: %d rows", d.name, f.name, len(df))


    if _task_id and _sdk_complete:
        try:
            _sdk_complete(_task_id)
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
