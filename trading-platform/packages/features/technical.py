"""技术指标因子 - 标准技术分析指标实现"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.registry import factor


@factor("ma", category="trend", output_columns=["ma"], period=20)
def ma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """简单移动平均线"""
    df[f"ma_{period}"] = df["close"].rolling(window=period).mean()
    return df


@factor("ema", category="trend", output_columns=["ema"], period=20)
def ema(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """指数移动平均线"""
    df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
    return df


@factor(
    "rsi",
    category="momentum",
    output_columns=["rsi"],
    period=14,
)
def rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """相对强弱指标"""
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


@factor(
    "macd",
    category="momentum",
    output_columns=["macd", "macd_signal", "macd_hist"],
    fast=12,
    slow=26,
    signal=9,
)
def macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD 指标"""
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


@factor(
    "bollinger_bands",
    category="volatility",
    output_columns=["bb_upper", "bb_middle", "bb_lower", "bb_width"],
    period=20,
    std_dev=2.0,
)
def bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.DataFrame:
    """布林带"""
    df["bb_middle"] = df["close"].rolling(window=period).mean()
    rolling_std = df["close"].rolling(window=period).std()
    df["bb_upper"] = df["bb_middle"] + std_dev * rolling_std
    df["bb_lower"] = df["bb_middle"] - std_dev * rolling_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    return df


@factor(
    "atr",
    category="volatility",
    output_columns=["atr"],
    period=14,
)
def atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """平均真实波幅"""
    high_low = df["high"] - df["low"]
    high_prev_close = (df["high"] - df["close"].shift(1)).abs()
    low_prev_close = (df["low"] - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    df["atr"] = true_range.ewm(alpha=1 / period, min_periods=period).mean()
    return df


@factor(
    "obv",
    category="volume",
    output_columns=["obv"],
)
def obv(df: pd.DataFrame) -> pd.DataFrame:
    """能量潮"""
    direction = np.sign(df["close"].diff())
    df["obv"] = (direction * df["volume"]).cumsum()
    return df


@factor(
    "vwap",
    category="volume",
    output_columns=["vwap"],
)
def vwap(df: pd.DataFrame) -> pd.DataFrame:
    """成交量加权平均价"""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (typical_price * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    df["vwap"] = cum_tp_vol / cum_vol.replace(0, np.nan)
    return df


@factor(
    "keltner_channel",
    category="volatility",
    output_columns=["kc_upper", "kc_middle", "kc_lower"],
    ema_period=20,
    atr_period=14,
    multiplier=2.0,
)
def keltner_channel(
    df: pd.DataFrame,
    ema_period: int = 20,
    atr_period: int = 14,
    multiplier: float = 2.0,
) -> pd.DataFrame:
    """肯特纳通道"""
    df["kc_middle"] = df["close"].ewm(span=ema_period, adjust=False).mean()
    high_low = df["high"] - df["low"]
    high_prev_close = (df["high"] - df["close"].shift(1)).abs()
    low_prev_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    atr_val = tr.ewm(alpha=1 / atr_period, min_periods=atr_period).mean()
    df["kc_upper"] = df["kc_middle"] + multiplier * atr_val
    df["kc_lower"] = df["kc_middle"] - multiplier * atr_val
    return df


@factor(
    "stochastic",
    category="momentum",
    output_columns=["stoch_k", "stoch_d"],
    k_period=14,
    d_period=3,
)
def stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    """随机指标 KD"""
    lowest_low = df["low"].rolling(window=k_period).min()
    highest_high = df["high"].rolling(window=k_period).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    df["stoch_k"] = 100 * (df["close"] - lowest_low) / denom
    df["stoch_d"] = df["stoch_k"].rolling(window=d_period).mean()
    return df
