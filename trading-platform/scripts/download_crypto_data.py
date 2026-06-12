"""Download BTC historical OHLCV data from Binance public API.

Saves Parquet files to ~/Downloads/crypto_data/ organized by symbol and timeframe.
No API key needed — uses public endpoints.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

import os as _os
import sys as _sys

try:
    from polarisor_port_sdk import submit_task as _sdk_submit, complete_task as _sdk_complete
except ImportError:
    _sdk_submit = _sdk_complete = None

import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_URL = "https://api.binance.com/api/v3"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "crypto_cache"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]

TIMEFRAMES = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

BINANCE_KLINE_LIMIT = 1000
START_DATE = datetime(2017, 8, 1, tzinfo=timezone.utc)
RATE_LIMIT_PAUSE = 0.15


async def fetch_klines(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> list[list]:
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": BINANCE_KLINE_LIMIT,
    }
    async with session.get(f"{BASE_URL}/klines", params=params) as resp:
        if resp.status == 429:
            logger.warning("Rate limited, sleeping 60s...")
            await asyncio.sleep(60)
            return await fetch_klines(session, symbol, interval, start_ms, end_ms)
        resp.raise_for_status()
        return await resp.json()


def klines_to_df(raw: list[list]) -> pd.DataFrame:
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
    return df


async def download_symbol_tf(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    tf_ms: int,
) -> None:
    out_dir = DATA_DIR / symbol.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{interval}.parquet"

    start_ms = int(START_DATE.timestamp() * 1000)
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    if out_path.exists():
        existing = pd.read_parquet(out_path)
        if len(existing) > 0:
            last_ts = int(existing["open_time"].max().timestamp() * 1000)
            start_ms = last_ts + tf_ms
            logger.info(
                "Resuming %s %s from %s (%d existing rows)",
                symbol, interval,
                datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).isoformat(),
                len(existing),
            )
        else:
            existing = None
    else:
        existing = None

    if start_ms >= end_ms:
        logger.info("✅ %s %s already up to date", symbol, interval)
        return

    all_chunks: list[pd.DataFrame] = []
    cursor = start_ms
    batch_count = 0

    empty_streak = 0
    while cursor < end_ms:
        batch_end = min(cursor + BINANCE_KLINE_LIMIT * tf_ms, end_ms)
        raw = await fetch_klines(session, symbol, interval, cursor, batch_end)
        if not raw:
            empty_streak += 1
            cursor = batch_end
            if empty_streak > 5:
                break
            await asyncio.sleep(RATE_LIMIT_PAUSE)
            continue
        empty_streak = 0

        chunk = klines_to_df(raw)
        all_chunks.append(chunk)
        batch_count += 1

        last = int(chunk["open_time"].iloc[-1].timestamp() * 1000)
        cursor = last + tf_ms

        if batch_count % 50 == 0:
            rows_so_far = sum(len(c) for c in all_chunks)
            pct = min((cursor - start_ms) / (end_ms - start_ms) * 100, 100)
            logger.info(
                "  %s %s: %d rows fetched (%.1f%%)",
                symbol, interval, rows_so_far, pct,
            )

        await asyncio.sleep(RATE_LIMIT_PAUSE)

    if not all_chunks:
        logger.info("No new data for %s %s", symbol, interval)
        return

    new_df = pd.concat(all_chunks, ignore_index=True)
    new_df.drop_duplicates(subset=["open_time"], keep="last", inplace=True)

    if existing is not None:
        final = pd.concat([existing, new_df], ignore_index=True)
        final.drop_duplicates(subset=["open_time"], keep="last", inplace=True)
    else:
        final = new_df

    final.sort_values("open_time", inplace=True)
    final.reset_index(drop=True, inplace=True)
    final.to_parquet(out_path, index=False)

    date_range = (
        f"{final['open_time'].min().strftime('%Y-%m-%d')} → "
        f"{final['open_time'].max().strftime('%Y-%m-%d')}"
    )
    logger.info(
        "✅ %s %s: %d rows saved (%s)",
        symbol, interval, len(final), date_range,
    )


async def main() -> None:

    _task_id = None
    if _sdk_submit:
        try:
            _tr = _sdk_submit(task_type="data-download", command="download_crypto_data.py", requester="download-crypto-data", estimated_duration_sec=600)
            _task_id = _tr.get("task_id")
        except Exception:
            pass
    logger.info("=" * 60)
    logger.info("Crypto Data Downloader")
    logger.info("Output: %s", DATA_DIR)
    logger.info("Symbols: %s", ", ".join(SYMBOLS))
    logger.info("Timeframes: %s", ", ".join(TIMEFRAMES.keys()))
    logger.info("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    connector = aiohttp.TCPConnector(limit=5)
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for symbol in SYMBOLS:
            for interval, tf_ms in TIMEFRAMES.items():
                try:
                    await download_symbol_tf(session, symbol, interval, tf_ms)
                except Exception as e:
                    logger.error("Failed %s %s: %s", symbol, interval, e)
                    continue

    logger.info("=" * 60)
    logger.info("Download complete!")

    total_rows = 0
    for symbol in SYMBOLS:
        sym_dir = DATA_DIR / symbol.lower()
        if sym_dir.exists():
            for f in sym_dir.glob("*.parquet"):
                df = pd.read_parquet(f)
                total_rows += len(df)
                logger.info("  %s: %d rows", f.name, len(df))

    logger.info("Total: %d rows across all files", total_rows)


    if _task_id and _sdk_complete:
        try:
            _sdk_complete(_task_id)
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
