"""数据清洗 - 去重、补缺、异常值检测、质量报告"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from datahub.models import DataQualityReport, TimeFrame

logger = logging.getLogger(__name__)

TIMEFRAME_SECONDS: dict[TimeFrame, int] = {
    TimeFrame.M1: 60,
    TimeFrame.M5: 300,
    TimeFrame.M15: 900,
    TimeFrame.M30: 1800,
    TimeFrame.H1: 3600,
    TimeFrame.H4: 14400,
    TimeFrame.D1: 86400,
}


def clean_ohlcv(
    df: pd.DataFrame,
    timeframe: TimeFrame,
    fill_method: str = "ffill",
    outlier_std: float = 5.0,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """清洗K线数据

    返回清洗后的 DataFrame 和质量报告。

    Args:
        df: 包含 timestamp, open, high, low, close, volume 列的 DataFrame
        timeframe: K线周期
        fill_method: 缺失值填充方法 ('ffill', 'interpolate', 'drop')
        outlier_std: 异常值检测阈值（标准差倍数）
    """
    symbol = df["symbol"].iloc[0] if "symbol" in df.columns else "UNKNOWN"
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    len(df)
    dup_count = df.duplicated(subset=["timestamp"]).sum()
    df = df.drop_duplicates(subset=["timestamp"], keep="last")

    gap_count = 0
    missing_count = 0
    step = TIMEFRAME_SECONDS.get(timeframe)

    if step and len(df) > 1:
        expected_range = pd.date_range(
            start=df["timestamp"].min(),
            end=df["timestamp"].max(),
            freq=f"{step}s",
        )
        missing_count = len(expected_range) - len(df)
        if missing_count > 0:
            df = df.set_index("timestamp").reindex(expected_range)
            df.index.name = "timestamp"
            gap_count = (df["close"].isna().astype(int).diff().eq(1)).sum()

            if fill_method == "ffill":
                df = df.ffill()
            elif fill_method == "interpolate":
                for col in ["open", "high", "low", "close"]:
                    df[col] = df[col].interpolate(method="linear")
                df["volume"] = df["volume"].fillna(0)
            elif fill_method == "drop":
                df = df.dropna(subset=["close"])

            df = df.reset_index()

    outlier_count = 0
    if len(df) > 30:
        returns = df["close"].pct_change().dropna()
        z_scores = np.abs((returns - returns.mean()) / returns.std())
        outlier_mask = z_scores > outlier_std
        outlier_count = int(outlier_mask.sum())
        if outlier_count > 0:
            logger.warning(
                "Found %d outlier bars for %s/%s", outlier_count, symbol, timeframe.value
            )

    coverage = (len(df) / (len(df) + missing_count)) * 100 if (len(df) + missing_count) > 0 else 100.0

    report = DataQualityReport(
        symbol=symbol,
        timeframe=timeframe,
        total_bars=len(df),
        missing_bars=max(0, missing_count),
        duplicate_bars=int(dup_count),
        outlier_bars=outlier_count,
        gap_count=int(gap_count),
        coverage_pct=round(coverage, 2),
    )

    logger.info(
        "Cleaned %s/%s: %d bars, %d dups removed, %d gaps filled, %.1f%% coverage",
        symbol, timeframe.value, len(df), dup_count, missing_count, coverage,
    )

    return df, report
