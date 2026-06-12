"""研究级因子 — 参考量化金融论文的高级特征。

References:
  [1] López de Prado, "Advances in Financial Machine Learning", 2018
      — Triple barrier labeling, volatility estimators
  [2] Parkinson (1980), "The Extreme Value Method for Estimating the Variance of
      the Rate of Return" — Parkinson volatility
  [3] Garman & Klass (1980), "On the Estimation of Security Price Volatilities
      from Historical Data" — GK volatility estimator
  [4] Yang & Zhang (2000), "Drift Independent Volatility Estimation" — YZ volatility
  [5] Cont (2001), "Empirical properties of asset returns" — Return distribution features
  [6] Amihud (2002), "Illiquidity and stock returns" — ILLIQ ratio
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.registry import factor


# ─── Volatility Estimators ───


@factor("parkinson_vol", category="volatility", output_columns=["parkinson_vol"], period=20)
def parkinson_vol(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Parkinson (1980) volatility estimator using high-low range."""
    log_hl = np.log(df["high"] / df["low"].replace(0, np.nan))
    df["parkinson_vol"] = np.sqrt(
        (1 / (4 * np.log(2))) * (log_hl ** 2).rolling(period).mean()
    )
    return df


@factor("garman_klass_vol", category="volatility", output_columns=["gk_vol"], period=20)
def garman_klass_vol(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Garman-Klass (1980) volatility: combines OHLC for better efficiency."""
    log_hl = np.log(df["high"] / df["low"].replace(0, np.nan))
    log_co = np.log(df["close"] / df["open"].replace(0, np.nan))
    gk = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    df["gk_vol"] = np.sqrt(gk.rolling(period).mean().clip(lower=0))
    return df


@factor("yang_zhang_vol", category="volatility", output_columns=["yz_vol"], period=20)
def yang_zhang_vol(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Yang-Zhang (2000) drift-independent volatility estimator."""
    log_oc = np.log(df["open"] / df["close"].shift(1).replace(0, np.nan))
    log_co = np.log(df["close"] / df["open"].replace(0, np.nan))
    log_ho = np.log(df["high"] / df["open"].replace(0, np.nan))
    log_lo = np.log(df["low"] / df["open"].replace(0, np.nan))

    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    overnight_var = log_oc.rolling(period).var()
    close_var = log_co.rolling(period).var()
    rs_var = rs.rolling(period).mean()

    k = 0.34 / (1.34 + (period + 1) / (period - 1))
    yz = overnight_var + k * close_var + (1 - k) * rs_var
    df["yz_vol"] = np.sqrt(yz.clip(lower=0))
    return df


# ─── Return Distribution Features (Cont 2001) ───


@factor("return_skewness", category="statistical", output_columns=["ret_skew"], period=20)
def return_skewness(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Rolling skewness of returns — captures asymmetry in return distribution."""
    rets = df["close"].pct_change()
    df["ret_skew"] = rets.rolling(period).skew()
    return df


@factor("return_kurtosis", category="statistical", output_columns=["ret_kurt"], period=20)
def return_kurtosis(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Rolling excess kurtosis — measures tail heaviness (Cont 2001)."""
    rets = df["close"].pct_change()
    df["ret_kurt"] = rets.rolling(period).kurt()
    return df


@factor("return_autocorr", category="statistical", output_columns=["ret_autocorr"], period=20)
def return_autocorr(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Lag-1 autocorrelation of returns — measures mean-reversion vs momentum."""
    rets = df["close"].pct_change()
    df["ret_autocorr"] = rets.rolling(period).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 1 else 0.0,
        raw=False,
    )
    return df


# ─── Microstructure Features ───


@factor("amihud_illiq", category="microstructure", output_columns=["amihud"], period=20)
def amihud_illiq(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Amihud (2002) illiquidity ratio — |return| / volume."""
    rets = df["close"].pct_change().abs()
    vol = df["volume"].replace(0, np.nan)
    illiq = rets / vol
    df["amihud"] = illiq.rolling(period).mean()
    return df


@factor("volume_imbalance", category="microstructure", output_columns=["vol_imbalance"])
def volume_imbalance(df: pd.DataFrame) -> pd.DataFrame:
    """Signed volume imbalance: positive for up moves, negative for down."""
    direction = np.sign(df["close"].diff())
    cum_up = (direction.clip(lower=0) * df["volume"]).rolling(20).sum()
    cum_down = ((-direction.clip(upper=0)) * df["volume"]).rolling(20).sum()
    total = cum_up + cum_down
    df["vol_imbalance"] = (cum_up - cum_down) / total.replace(0, np.nan)
    return df


@factor("price_acceleration", category="momentum", output_columns=["price_accel"], period=10)
def price_acceleration(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    """Second derivative of price — rate of change of momentum."""
    mom = df["close"].pct_change(period)
    df["price_accel"] = mom.diff()
    return df


# ─── Time Features ───


@factor("bar_of_day", category="temporal", output_columns=["bar_of_day", "session_cos", "session_sin"])
def bar_of_day(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical time-of-day encoding for intraday seasonality."""
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"])
        minutes = dt.dt.hour * 60 + dt.dt.minute
        total_minutes = 24 * 60
        df["bar_of_day"] = minutes / total_minutes
        df["session_cos"] = np.cos(2 * np.pi * minutes / total_minutes)
        df["session_sin"] = np.sin(2 * np.pi * minutes / total_minutes)
    else:
        df["bar_of_day"] = 0.5
        df["session_cos"] = 0.0
        df["session_sin"] = 0.0
    return df


# ─── Triple Barrier Labeling (López de Prado 2018) ───


def triple_barrier_labels(
    close: pd.Series,
    period: int = 10,
    upper_mult: float = 1.5,
    lower_mult: float = 1.5,
    vol_lookback: int = 20,
) -> pd.Series:
    """Triple barrier method for generating labels (AFML Ch. 3).

    - Upper barrier: close + upper_mult * daily_vol → profit-take (+1)
    - Lower barrier: close - lower_mult * daily_vol → stop-loss (-1)
    - Vertical barrier: period bars timeout → sign of return (0 if flat)

    Returns Series of {-1, 0, 1}.
    """
    rets = close.pct_change()
    daily_vol = rets.rolling(vol_lookback).std()

    labels = pd.Series(0, index=close.index, dtype=int)

    for i in range(len(close) - period):
        entry = close.iloc[i]
        vol = daily_vol.iloc[i]
        if np.isnan(vol) or vol <= 0:
            continue

        upper = entry * (1 + upper_mult * vol)
        lower = entry * (1 - lower_mult * vol)

        touched_upper = False
        touched_lower = False

        for j in range(1, period + 1):
            idx = i + j
            if idx >= len(close):
                break
            px = close.iloc[idx]
            if px >= upper:
                touched_upper = True
                break
            if px <= lower:
                touched_lower = True
                break

        if touched_upper:
            labels.iloc[i] = 1
        elif touched_lower:
            labels.iloc[i] = -1
        else:
            final_ret = (close.iloc[min(i + period, len(close) - 1)] - entry) / entry
            labels.iloc[i] = int(np.sign(final_ret))

    return labels
