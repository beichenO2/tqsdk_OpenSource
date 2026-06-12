"""Download historical funding rates from Binance Futures public API.

Saves Parquet files alongside OHLCV data.
No API key needed — uses public endpoints.
"""

from __future__ import annotations

import os as _os
import sys as _sys

try:
    from polarisor_port_sdk import submit_task as _sdk_submit, complete_task as _sdk_complete
except ImportError:
    _sdk_submit = _sdk_complete = None


import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FAPI_URL = "https://fapi.binance.com/fapi/v1"
DATA_DIR = Path.home() / "Downloads" / "crypto_data"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
START_DATE = datetime(2019, 9, 1, tzinfo=timezone.utc)
RATE_LIMIT_PAUSE = 0.2


async def fetch_funding(
    session: aiohttp.ClientSession, symbol: str, start_ms: int, end_ms: int,
) -> list[dict]:
    params = {
        "symbol": symbol,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": 1000,
    }
    async with session.get(f"{FAPI_URL}/fundingRate", params=params) as resp:
        if resp.status == 429:
            logger.warning("Rate limited, sleeping 60s...")
            await asyncio.sleep(60)
            return await fetch_funding(session, symbol, start_ms, end_ms)
        resp.raise_for_status()
        return await resp.json()


async def download_symbol_funding(
    session: aiohttp.ClientSession, symbol: str,
) -> None:
    out_dir = DATA_DIR / symbol.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "funding_rate.parquet"

    start_ms = int(START_DATE.timestamp() * 1000)
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    existing = None
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        if len(existing) > 0:
            last_ts = int(existing["funding_time"].max().timestamp() * 1000)
            start_ms = last_ts + 1
            logger.info("Resuming %s funding from %s (%d existing)", symbol,
                        datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).isoformat(),
                        len(existing))

    if start_ms >= end_ms:
        logger.info("%s funding already up to date", symbol)
        return

    all_rows: list[dict] = []
    cursor = start_ms
    while cursor < end_ms:
        raw = await fetch_funding(session, symbol, cursor, end_ms)
        if not raw:
            break
        all_rows.extend(raw)
        last = raw[-1]["fundingTime"]
        cursor = last + 1
        await asyncio.sleep(RATE_LIMIT_PAUSE)

    if not all_rows:
        logger.info("No new funding data for %s", symbol)
        return

    df = pd.DataFrame(all_rows)
    df["funding_time"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df = df[["funding_time", "funding_rate", "symbol"]].copy()
    df.drop_duplicates(subset=["funding_time"], keep="last", inplace=True)

    if existing is not None:
        df = pd.concat([existing, df], ignore_index=True)
        df.drop_duplicates(subset=["funding_time"], keep="last", inplace=True)

    df.sort_values("funding_time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_parquet(out_path, index=False)

    logger.info(
        "%s funding: %d rows [%s -> %s]", symbol, len(df),
        df["funding_time"].iloc[0].strftime("%Y-%m-%d"),
        df["funding_time"].iloc[-1].strftime("%Y-%m-%d"),
    )


async def main() -> None:

    _task_id = None
    if _sdk_submit:
        try:
            _tr = _sdk_submit(task_type="data-download", command="download_funding_rates.py", requester="download-funding-rates", estimated_duration_sec=300)
            _task_id = _tr.get("task_id")
        except Exception:
            pass
    logger.info("Downloading funding rates for: %s", ", ".join(SYMBOLS))
    connector = aiohttp.TCPConnector(limit=3)
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for symbol in SYMBOLS:
            try:
                await download_symbol_funding(session, symbol)
            except Exception as e:
                logger.error("Failed %s: %s", symbol, e)
    logger.info("Done!")


    if _task_id and _sdk_complete:
        try:
            _sdk_complete(_task_id)
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
