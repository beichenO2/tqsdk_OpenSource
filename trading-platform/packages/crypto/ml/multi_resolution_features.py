"""Multi-resolution feature generator for crypto ML.

Computes technical indicators at multiple timeframes (e.g. 1h, 4h, 1d)
and merges them into a single feature matrix. This captures patterns
at different time scales that single-timeframe features miss.

Also adds cross-asset features (ETH/BTC ratio, BTC dominance proxy)
and regime classification as conditioning variables.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RESOLUTIONS = {
    "1h": 1,
    "4h": 4,
    "1d": 24,
}

CORE_INDICATORS = ["rsi", "macd_hist", "atr", "bb_width", "obv_norm"]


def _rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(closes)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss > 1e-10 else 100.0
        rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def _macd_hist(closes: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> np.ndarray:
    n = len(closes)
    result = np.zeros(n)
    if n < slow + signal:
        return result

    def _ema(data, period):
        out = np.zeros_like(data)
        alpha = 2.0 / (period + 1)
        out[0] = data[0]
        for i in range(1, len(data)):
            out[i] = alpha * data[i] + (1 - alpha) * out[i - 1]
        return out

    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    macd_line = ema_fast - ema_slow
    sig_line = _ema(macd_line, signal)
    result = macd_line - sig_line
    return result


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(closes)
    atr = np.zeros(n)
    for i in range(1, n):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        if i < period:
            trs = [max(highs[j] - lows[j], abs(highs[j] - closes[j - 1]), abs(lows[j] - closes[j - 1]))
                   for j in range(1, i + 1)]
            atr[i] = np.mean(trs)
        else:
            atr[i] = (atr[i - 1] * (period - 1) + tr) / period
    return atr


def _bb_width(closes: np.ndarray, period: int = 20) -> np.ndarray:
    n = len(closes)
    width = np.zeros(n)
    for i in range(period - 1, n):
        window = closes[i - period + 1:i + 1]
        mid = np.mean(window)
        std = np.std(window)
        width[i] = (4 * std) / mid if mid > 0 else 0
    return width


def _obv_normalized(closes: np.ndarray, volumes: np.ndarray, norm_period: int = 20) -> np.ndarray:
    n = len(closes)
    obv = np.zeros(n)
    for i in range(1, n):
        if closes[i] > closes[i - 1]:
            obv[i] = obv[i - 1] + volumes[i]
        elif closes[i] < closes[i - 1]:
            obv[i] = obv[i - 1] - volumes[i]
        else:
            obv[i] = obv[i - 1]

    obv_norm = np.zeros(n)
    for i in range(norm_period, n):
        window = obv[i - norm_period:i + 1]
        std = np.std(window)
        if std > 1e-10:
            obv_norm[i] = (obv[i] - np.mean(window)) / std
    return obv_norm


def _resample_ohlcv(df: pd.DataFrame, factor: int) -> pd.DataFrame:
    """Resample OHLCV by grouping every `factor` bars."""
    if factor <= 1:
        return df.copy()

    n = len(df)
    n_groups = n // factor
    if n_groups < 10:
        return df.copy()

    trimmed = df.iloc[-(n_groups * factor):]
    groups = np.arange(len(trimmed)) // factor

    resampled = pd.DataFrame({
        "open": trimmed.groupby(groups)["open"].first().values,
        "high": trimmed.groupby(groups)["high"].max().values,
        "low": trimmed.groupby(groups)["low"].min().values,
        "close": trimmed.groupby(groups)["close"].last().values,
        "volume": trimmed.groupby(groups)["volume"].sum().values,
    })
    return resampled


def compute_indicators(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Compute core indicators for a single timeframe."""
    closes = df["close"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    volumes = df["volume"].values.astype(np.float64)

    return {
        "rsi": _rsi(closes),
        "macd_hist": _macd_hist(closes),
        "atr": _atr(highs, lows, closes),
        "bb_width": _bb_width(closes),
        "obv_norm": _obv_normalized(closes, volumes),
    }


def compute_multi_resolution_features(
    df: pd.DataFrame,
    base_timeframe: str = "1h",
    resolutions: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Compute features at multiple time resolutions and merge.

    For base_timeframe="1h" with resolutions={"1h": 1, "4h": 4, "1d": 24}:
    - 1h indicators computed directly on raw bars
    - 4h indicators computed on 4-bar resampled OHLCV, then repeated to original length
    - 1d indicators computed on 24-bar resampled OHLCV, then repeated to original length

    Returns DataFrame with columns like "rsi_1h", "rsi_4h", "rsi_1d", etc.
    """
    if resolutions is None:
        resolutions = RESOLUTIONS

    result = df[["open", "high", "low", "close", "volume"]].copy()
    n = len(result)

    for res_name, factor in sorted(resolutions.items(), key=lambda x: x[1]):
        resampled = _resample_ohlcv(df, factor)
        indicators = compute_indicators(resampled)

        for ind_name, ind_values in indicators.items():
            col_name = f"{ind_name}_{res_name}"
            if factor <= 1:
                result[col_name] = ind_values[-n:] if len(ind_values) >= n else np.pad(
                    ind_values, (n - len(ind_values), 0), mode="edge"
                )
            else:
                expanded = np.repeat(ind_values, factor)
                if len(expanded) >= n:
                    result[col_name] = expanded[-n:]
                else:
                    result[col_name] = np.pad(expanded, (n - len(expanded), 0), mode="edge")

    closes = df["close"].values.astype(np.float64)
    for p in [5, 10, 20]:
        result[f"returns_{p}"] = df["close"].pct_change(p)

    result["vol_10"] = df["close"].pct_change(1).rolling(10).std()
    result["vol_20"] = df["close"].pct_change(1).rolling(20).std()

    if "volume" in df.columns:
        result["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    if "taker_buy_volume" in df.columns and "volume" in df.columns:
        result["taker_ratio"] = df["taker_buy_volume"] / df["volume"].replace(0, np.nan)
        result["taker_ratio"] = result["taker_ratio"].fillna(0.5)

    result["high_low_range"] = (df["high"] - df["low"]) / df["close"]

    return result


def add_cross_asset_features(
    primary_df: pd.DataFrame,
    secondary_df: pd.DataFrame | None = None,
    secondary_name: str = "eth",
) -> pd.DataFrame:
    """Add cross-asset features (e.g., ETH/BTC ratio)."""
    result = primary_df.copy()

    if secondary_df is not None and len(secondary_df) >= len(primary_df):
        sec_closes = secondary_df["close"].values[-len(primary_df):]
        pri_closes = primary_df["close"].values

        mask = pri_closes > 0
        ratio = np.where(mask, sec_closes / pri_closes, 1.0)
        result[f"{secondary_name}_ratio"] = ratio
        result[f"{secondary_name}_ratio_change"] = pd.Series(ratio).pct_change(5).values

    return result


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add regime classification as one-hot features."""
    result = df.copy()
    closes = df["close"].values.astype(np.float64)
    n = len(closes)

    fast_ma = pd.Series(closes).rolling(20).mean().values
    slow_ma = pd.Series(closes).rolling(50).mean().values
    vol = pd.Series(closes).pct_change(1).rolling(20).std().values

    regime = np.zeros(n, dtype=int)
    for i in range(50, n):
        trend = abs(fast_ma[i] - slow_ma[i]) / slow_ma[i] if slow_ma[i] > 0 else 0
        if trend > 0.03:
            regime[i] = 0  # strong trend
        elif trend > 0.01:
            regime[i] = 1  # weak trend
        else:
            regime[i] = 2  # ranging

    for r_id in range(3):
        result[f"regime_{r_id}"] = (regime == r_id).astype(float)

    return result
