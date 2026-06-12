"""统计因子 - 波动率、分布特征、时间序列统计"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.registry import factor


@factor(
    "realized_volatility",
    category="statistical",
    output_columns=["realized_vol"],
    period=20,
    annualize=True,
)
def realized_volatility(
    df: pd.DataFrame,
    period: int = 20,
    annualize: bool = True,
) -> pd.DataFrame:
    """已实现波动率"""
    returns = np.log(df["close"] / df["close"].shift(1))
    vol = returns.rolling(window=period).std()
    if annualize:
        vol = vol * np.sqrt(252)
    df["realized_vol"] = vol
    return df


@factor(
    "parkinson_volatility",
    category="statistical",
    output_columns=["parkinson_vol"],
    period=20,
)
def parkinson_volatility(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Parkinson 波动率（基于最高最低价）"""
    hl_ratio = np.log(df["high"] / df["low"])
    factor_val = 1 / (4 * np.log(2))
    df["parkinson_vol"] = np.sqrt(
        factor_val * (hl_ratio ** 2).rolling(window=period).mean()
    ) * np.sqrt(252)
    return df


@factor(
    "returns_skewness",
    category="statistical",
    output_columns=["skewness"],
    period=60,
)
def returns_skewness(df: pd.DataFrame, period: int = 60) -> pd.DataFrame:
    """收益率偏度"""
    returns = df["close"].pct_change()
    df["skewness"] = returns.rolling(window=period).skew()
    return df


@factor(
    "returns_kurtosis",
    category="statistical",
    output_columns=["kurtosis"],
    period=60,
)
def returns_kurtosis(df: pd.DataFrame, period: int = 60) -> pd.DataFrame:
    """收益率峰度"""
    returns = df["close"].pct_change()
    df["kurtosis"] = returns.rolling(window=period).kurt()
    return df


@factor(
    "hurst_exponent",
    category="statistical",
    output_columns=["hurst"],
    max_lag=20,
)
def hurst_exponent(df: pd.DataFrame, max_lag: int = 20) -> pd.DataFrame:
    """Hurst 指数 - 衡量趋势/均值回复特性

    H > 0.5: 趋势性
    H = 0.5: 随机游走
    H < 0.5: 均值回复
    """
    closes = df["close"].values
    hursts = []

    for i in range(len(closes)):
        if i < max_lag * 4:
            hursts.append(np.nan)
            continue

        ts = closes[i - max_lag * 4 : i + 1]
        lags = range(2, max_lag + 1)
        tau = []
        for lag in lags:
            diffs = ts[lag:] - ts[:-lag]
            std = np.std(diffs)
            if std > 0:
                tau.append(std)
            else:
                tau.append(np.nan)

        valid = [(l, t) for l, t in zip(lags, tau) if not np.isnan(t)]
        if len(valid) < 3:
            hursts.append(np.nan)
            continue

        log_lags = np.log([v[0] for v in valid])
        log_tau = np.log([v[1] for v in valid])
        poly = np.polyfit(log_lags, log_tau, 1)
        hursts.append(poly[0])

    df["hurst"] = hursts
    return df


@factor(
    "autocorrelation",
    category="statistical",
    output_columns=["autocorr"],
    period=20,
    lag=1,
)
def autocorrelation(
    df: pd.DataFrame,
    period: int = 20,
    lag: int = 1,
) -> pd.DataFrame:
    """收益率自相关系数"""
    returns = df["close"].pct_change()
    df["autocorr"] = returns.rolling(window=period).apply(
        lambda x: x.autocorr(lag=lag), raw=False
    )
    return df


@factor(
    "z_score",
    category="statistical",
    output_columns=["z_score"],
    period=20,
)
def z_score(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """价格 Z-Score（距均值的标准差数）"""
    rolling_mean = df["close"].rolling(window=period).mean()
    rolling_std = df["close"].rolling(window=period).std()
    df["z_score"] = (df["close"] - rolling_mean) / rolling_std.replace(0, np.nan)
    return df
