"""Download Premium Index Klines from Binance public S3 (no API key needed).

Premium Index ≈ (Mark Price - Index Price) / Index Price.
Funding Rate = average(Premium Index over 8h) + clamp(interest - avg, -0.05%, 0.05%)

This gives us a close proxy for real funding rate data.
"""

from __future__ import annotations

import os as _os
import sys as _sys

try:
    from polarisor_port_sdk import submit_task as _sdk_submit, complete_task as _sdk_complete
except ImportError:
    _sdk_submit = _sdk_complete = None


import asyncio
import io
import logging
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

S3_BASE = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
DATA_DIR = Path.home() / "Downloads" / "crypto_data"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
TIMEFRAME = "4h"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


async def list_files(session: aiohttp.ClientSession, symbol: str) -> list[str]:
    prefix = f"data/futures/um/monthly/premiumIndexKlines/{symbol}/{TIMEFRAME}/"
    url = f"{S3_BASE}?prefix={prefix}&delimiter=/"
    async with session.get(url) as resp:
        text = await resp.text()
        root = ET.fromstring(text)
        keys = [k.text for k in root.findall(".//s3:Key", NS) if k.text and k.text.endswith(".zip")]
        return keys


async def download_and_parse(session: aiohttp.ClientSession, key: str) -> pd.DataFrame:
    url = f"https://data.binance.vision/{key}"
    async with session.get(url) as resp:
        if resp.status != 200:
            logger.warning("Failed to download %s: %d", key, resp.status)
            return pd.DataFrame()
        content = await resp.read()

    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            first_line = f.readline().decode("utf-8", errors="ignore").strip()
            f.seek(0)
            has_header = not first_line[0].isdigit()
            df = pd.read_csv(f, header=0 if has_header else None)

    if not has_header:
        cols = ["open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_volume",
                "taker_buy_quote_volume", "ignore"]
        df.columns = cols[:len(df.columns)]

    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    df.dropna(subset=["open_time"], inplace=True)
    df["open_time"] = pd.to_datetime(df["open_time"].astype(int), unit="ms", utc=True)
    if "close" in df.columns:
        df["close"] = pd.to_numeric(df["close"], errors="coerce")

    return df


async def download_symbol(session: aiohttp.ClientSession, symbol: str) -> None:
    keys = await list_files(session, symbol)
    if not keys:
        logger.warning("No premium index data for %s", symbol)
        return

    logger.info("%s: found %d monthly files", symbol, len(keys))

    out_dir = DATA_DIR / symbol.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "premium_index_4h.parquet"

    existing_months: set[str] = set()
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        if len(existing) > 0:
            existing_months = set(existing["open_time"].dt.strftime("%Y-%m"))
            logger.info("  %d existing rows, %d months", len(existing), len(existing_months))
    else:
        existing = None

    all_dfs: list[pd.DataFrame] = []
    for key in keys:
        month = key.split("-")[-1].replace(".zip", "")
        month_label = "-".join(key.split("-")[-2:]).replace(".zip", "")
        if month_label in existing_months:
            continue

        df = await download_and_parse(session, key)
        if not df.empty:
            all_dfs.append(df)
        await asyncio.sleep(0.1)

    if not all_dfs and existing is not None:
        logger.info("%s premium index already up to date (%d rows)", symbol, len(existing))
        return

    if all_dfs:
        new_df = pd.concat(all_dfs, ignore_index=True)
        if existing is not None:
            new_df = pd.concat([existing, new_df], ignore_index=True)
        new_df.drop_duplicates(subset=["open_time"], keep="last", inplace=True)
        new_df.sort_values("open_time", inplace=True)
        new_df.reset_index(drop=True, inplace=True)
    elif existing is not None:
        new_df = existing
    else:
        logger.info("No data for %s", symbol)
        return

    new_df.to_parquet(out_path, index=False)
    logger.info(
        "%s: saved %d rows [%s → %s]",
        symbol, len(new_df),
        new_df["open_time"].iloc[0].strftime("%Y-%m-%d"),
        new_df["open_time"].iloc[-1].strftime("%Y-%m-%d"),
    )


async def main() -> None:

    _task_id = None
    if _sdk_submit:
        try:
            _tr = _sdk_submit(task_type="data-download", command="download_premium_index.py", requester="download-premium-index", estimated_duration_sec=300)
            _task_id = _tr.get("task_id")
        except Exception:
            pass
    logger.info("Downloading Premium Index Klines (funding rate proxy)")
    connector = aiohttp.TCPConnector(limit=5)
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for symbol in SYMBOLS:
            try:
                await download_symbol(session, symbol)
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
