"""期货 Tick 数据加载器 — 直接从 ZIP 读取 CSV，聚合成 OHLCV K 线。

数据源路径: ~/Downloads/期货数据/  (不复制，原地读取)
ZIP 结构: YYYYMM/YYYYMMDD/contract_YYYYMMDD.csv
CSV 列: TradingDay,InstrumentID,UpdateTime,UpdateMillisec,LastPrice,Volume,
         BidPrice1,BidVolume1,AskPrice1,AskVolume1,AveragePrice,Turnover,
         OpenInterest,UpperLimitPrice,LowerLimitPrice
"""

from __future__ import annotations

import io
import logging
import os
import re
import zipfile
from datetime import timedelta
from pathlib import Path
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

DATA_ROOT = os.environ.get(
    "FUTURES_DATA_ROOT",
    os.path.expanduser("~/Downloads/期货数据"),
)

_INSTRUMENT_RE = re.compile(r"^([A-Za-z]+)\d+$")

TimeFrame = Literal["1m", "5m", "15m", "30m", "1h", "1d"]

_TF_MINUTES: dict[str, int] = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "1d": 1440,
}


def instrument_to_symbol(instrument_id: str) -> str:
    m = _INSTRUMENT_RE.match(instrument_id)
    return m.group(1).upper() if m else instrument_id


def _find_zip_files(
    root: str,
    year_start: int | None = None,
    year_end: int | None = None,
    month_start: int | None = None,
    month_end: int | None = None,
) -> list[str]:
    """Locate monthly zip files covering the requested date range."""
    zips: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.lower().endswith(".zip"):
                continue
            base = fn.replace(".zip", "")
            if not base.isdigit() or len(base) != 6:
                continue
            y, m = int(base[:4]), int(base[4:6])
            if year_start and y < year_start:
                continue
            if year_end and y > year_end:
                continue
            if year_start and y == year_start and month_start and m < month_start:
                continue
            if year_end and y == year_end and month_end and m > month_end:
                continue
            zips.append(os.path.join(dirpath, fn))
    zips.sort()
    return zips


def _read_ticks_from_zip(
    zip_path: str,
    instrument_filter: str | None = None,
) -> pd.DataFrame:
    """Read tick CSVs from a single zip, filtered by instrument prefix."""
    rows: list[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        csvs = [n for n in zf.namelist() if n.endswith(".csv")]
        if instrument_filter:
            pat = instrument_filter.lower()
            csvs = [n for n in csvs if _csv_matches(n, pat)]

        for csv_name in csvs:
            try:
                raw = zf.read(csv_name)
                text = _decode(raw)
                if text is None:
                    continue
                df = pd.read_csv(
                    io.StringIO(text),
                    dtype={
                        "TradingDay": str,
                        "InstrumentID": str,
                        "UpdateTime": str,
                    },
                )
                if df.empty:
                    continue
                rows.append(df)
            except Exception:
                logger.debug("Skip bad CSV: %s in %s", csv_name, zip_path)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _csv_matches(csv_path: str, instrument_lower: str) -> bool:
    base = csv_path.split("/")[-1].split("_")[0].lower()
    sym = re.sub(r"\d+$", "", base)
    return sym == instrument_lower


def _decode(raw: bytes) -> str | None:
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return None


def _ticks_to_ohlcv(
    df: pd.DataFrame,
    timeframe: TimeFrame = "5m",
) -> pd.DataFrame:
    """Aggregate tick snapshots to OHLCV bars.

    Tick volume is cumulative within a day; we diff to get per-tick incremental volume.
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "datetime", "instrument", "open", "high", "low", "close",
            "volume", "turnover", "open_interest",
        ])

    df = df.copy()
    df["datetime"] = pd.to_datetime(
        df["TradingDay"].astype(str) + " " + df["UpdateTime"].astype(str),
        format="%Y%m%d %H:%M:%S",
        errors="coerce",
    )
    df = df.dropna(subset=["datetime", "LastPrice"])
    df = df.sort_values(["InstrumentID", "datetime"])
    df["LastPrice"] = pd.to_numeric(df["LastPrice"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    df["Turnover"] = pd.to_numeric(df["Turnover"], errors="coerce").fillna(0)
    df["OpenInterest"] = pd.to_numeric(df["OpenInterest"], errors="coerce").fillna(0)

    # Compute incremental volume per instrument per day
    df["vol_inc"] = df.groupby(["InstrumentID", "TradingDay"])["Volume"].diff().clip(lower=0).fillna(0)
    df["to_inc"] = df.groupby(["InstrumentID", "TradingDay"])["Turnover"].diff().clip(lower=0).fillna(0)

    minutes = _TF_MINUTES[timeframe]
    if minutes >= 1440:
        df["bar_time"] = df["datetime"].dt.normalize()
    else:
        df["bar_time"] = df["datetime"].dt.floor(f"{minutes}min")

    grouped = df.groupby(["InstrumentID", "bar_time"])
    bars = grouped.agg(
        open=("LastPrice", "first"),
        high=("LastPrice", "max"),
        low=("LastPrice", "min"),
        close=("LastPrice", "last"),
        volume=("vol_inc", "sum"),
        turnover=("to_inc", "sum"),
        open_interest=("OpenInterest", "last"),
    ).reset_index()

    bars = bars.rename(columns={"InstrumentID": "instrument", "bar_time": "datetime"})
    return bars.sort_values(["instrument", "datetime"]).reset_index(drop=True)


class FuturesDataLoader:
    """Load and aggregate Chinese futures tick data from zip archives."""

    def __init__(self, data_root: str | None = None):
        self.data_root = data_root or DATA_ROOT

    def load_bars(
        self,
        instrument: str,
        timeframe: TimeFrame = "5m",
        start_date: str | None = None,
        end_date: str | None = None,
        *,
        cache_dir: str | None = None,
    ) -> pd.DataFrame:
        """Load OHLCV bars for a single instrument (e.g. 'rb', 'IF', 'AP').

        Parameters
        ----------
        instrument
            Symbol prefix, e.g. ``"rb"`` for 螺纹钢, ``"IF"`` for 沪深300股指.
        timeframe
            Bar period: '1m', '5m', '15m', '30m', '1h', '1d'.
        start_date, end_date
            Format ``YYYY-MM-DD`` or ``YYYYMMDD``.
        cache_dir
            If set, cache aggregated parquet to this directory for fast reload.
        """
        cache_path = None
        if cache_dir:
            safe = f"{instrument}_{timeframe}_{start_date or 'all'}_{end_date or 'all'}.parquet"
            cache_path = Path(cache_dir) / safe
            if cache_path.exists():
                logger.info("Loading cached bars: %s", cache_path)
                return pd.read_parquet(cache_path)

        y_start, m_start, y_end, m_end = None, None, None, None
        if start_date:
            sd = start_date.replace("-", "")
            y_start, m_start = int(sd[:4]), int(sd[4:6])
        if end_date:
            ed = end_date.replace("-", "")
            y_end, m_end = int(ed[:4]), int(ed[4:6])

        zips = _find_zip_files(self.data_root, y_start, y_end, m_start, m_end)
        if not zips:
            logger.warning("No zip files found for %s in %s", instrument, self.data_root)
            return pd.DataFrame()

        logger.info("Loading %s from %d zip files (%s→%s)", instrument, len(zips), start_date, end_date)
        all_bars: list[pd.DataFrame] = []
        for zp in zips:
            ticks = _read_ticks_from_zip(zp, instrument.lower())
            if ticks.empty:
                continue
            bars = _ticks_to_ohlcv(ticks, timeframe)
            if not bars.empty:
                all_bars.append(bars)

        if not all_bars:
            return pd.DataFrame()

        result = pd.concat(all_bars, ignore_index=True).sort_values("datetime").reset_index(drop=True)

        if start_date:
            sd_dt = pd.to_datetime(start_date)
            result = result[result["datetime"] >= sd_dt]
        if end_date:
            ed_dt = pd.to_datetime(end_date) + timedelta(days=1)
            result = result[result["datetime"] < ed_dt]

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            result.to_parquet(cache_path, index=False)
            logger.info("Cached %d bars to %s", len(result), cache_path)

        return result.reset_index(drop=True)

    def load_main_contract_bars(
        self,
        symbol: str,
        timeframe: TimeFrame = "5m",
        start_date: str | None = None,
        end_date: str | None = None,
        *,
        cache_dir: str | None = None,
    ) -> pd.DataFrame:
        """Load bars and stitch main contract (by max open interest per day)."""
        bars = self.load_bars(symbol, timeframe, start_date, end_date, cache_dir=cache_dir)
        if bars.empty:
            return bars

        bars["date"] = bars["datetime"].dt.date
        daily_oi = bars.groupby(["instrument", "date"])["open_interest"].mean().reset_index()
        main_idx = daily_oi.loc[daily_oi.groupby("date")["open_interest"].idxmax()]
        main_map = dict(zip(main_idx["date"], main_idx["instrument"]))

        mask = bars.apply(lambda r: main_map.get(r["date"]) == r["instrument"], axis=1)
        result = bars[mask].drop(columns=["date"]).reset_index(drop=True)
        return result

    def list_instruments(self, year: int = 2024) -> list[str]:
        """List available instrument symbols in a given year's data."""
        zips = _find_zip_files(self.data_root, year, year)
        instruments: set[str] = set()
        for zp in zips[:1]:
            with zipfile.ZipFile(zp, "r") as zf:
                for name in zf.namelist():
                    if not name.endswith(".csv"):
                        continue
                    base = name.split("/")[-1].split("_")[0]
                    sym = re.sub(r"\d+$", "", base).upper()
                    if sym:
                        instruments.add(sym)
            break
        return sorted(instruments)

    def to_strategy_format(self, bars: pd.DataFrame) -> list[dict]:
        """Convert bars DataFrame to list of dicts compatible with BaseStrategy.on_bar()."""
        records = []
        for _, row in bars.iterrows():
            records.append({
                "datetime": row["datetime"],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            })
        return records
