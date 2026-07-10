"""Thin adapter: FuturesDataLoader bar DataFrame → BacktestEngine Bar objects."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

import pandas as pd

from .models import Bar


def _coerce_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.to_pydatetime()


def dataframe_to_bars(df: pd.DataFrame, symbol: str | None = None) -> list[Bar]:
    """Convert a FuturesDataLoader OHLCV DataFrame into engine ``Bar`` records.

    Does not mutate *df* or change FuturesDataLoader output semantics.
    """
    if df.empty:
        return []

    bars: list[Bar] = []
    for _, row in df.iterrows():
        bar_symbol = symbol
        if bar_symbol is None and "instrument" in row.index:
            bar_symbol = str(row["instrument"])
        if bar_symbol is None:
            bar_symbol = "unknown"

        dt = _coerce_datetime(row["datetime"])
        volume = int(float(row.get("volume", 0) or 0))
        oi = int(float(row.get("open_interest", 0) or 0))
        turnover = Decimal(str(row.get("turnover", 0) or 0))

        bars.append(
            Bar(
                symbol=bar_symbol,
                dt=dt,
                open=Decimal(str(row.get("open", 0))),
                high=Decimal(str(row.get("high", 0))),
                low=Decimal(str(row.get("low", 0))),
                close=Decimal(str(row.get("close", 0))),
                volume=volume,
                open_interest=oi,
                turnover=turnover,
            )
        )
    return bars
