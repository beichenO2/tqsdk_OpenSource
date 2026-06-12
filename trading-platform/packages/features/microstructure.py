"""微观结构因子 - 订单流、买卖压力、流动性指标"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.registry import factor


@factor(
    "order_flow_imbalance",
    category="microstructure",
    output_columns=["ofi"],
    period=20,
)
def order_flow_imbalance(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """订单流不平衡指标 (OFI)

    衡量买方与卖方的力量对比。
    需要 bid_volume_1 和 ask_volume_1 列。
    """
    if "bid_volume_1" not in df.columns or "ask_volume_1" not in df.columns:
        df["ofi"] = np.nan
        return df

    bid_delta = df["bid_volume_1"].diff()
    ask_delta = df["ask_volume_1"].diff()
    df["ofi"] = (bid_delta - ask_delta).rolling(window=period).sum()
    return df


@factor(
    "buy_sell_pressure",
    category="microstructure",
    output_columns=["buy_pressure", "sell_pressure", "pressure_ratio"],
)
def buy_sell_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """买卖压力指标

    基于价格位置估算买卖力量。
    """
    price_range = df["high"] - df["low"]
    price_range = price_range.replace(0, np.nan)

    df["buy_pressure"] = (df["close"] - df["low"]) / price_range
    df["sell_pressure"] = (df["high"] - df["close"]) / price_range
    df["pressure_ratio"] = df["buy_pressure"] / df["sell_pressure"].replace(0, np.nan)
    return df


@factor(
    "amihud_illiquidity",
    category="microstructure",
    output_columns=["amihud"],
    period=20,
)
def amihud_illiquidity(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Amihud 非流动性指标

    |收益率| / 成交额，衡量价格冲击成本。
    """
    returns = df["close"].pct_change().abs()
    turnover = df.get("turnover", df["volume"] * df["close"])
    turnover = turnover.replace(0, np.nan)
    daily_illiq = returns / turnover
    df["amihud"] = daily_illiq.rolling(window=period).mean()
    return df


@factor(
    "kyle_lambda",
    category="microstructure",
    output_columns=["kyle_lambda"],
    period=20,
)
def kyle_lambda(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Kyle's Lambda - 价格冲击系数

    回归斜率：Δprice ~ λ * signed_volume
    """
    price_change = df["close"].diff()
    signed_vol = np.sign(price_change) * df["volume"]

    lambdas = []
    for i in range(len(df)):
        if i < period:
            lambdas.append(np.nan)
            continue
        window_pc = price_change.iloc[i - period + 1 : i + 1]
        window_sv = signed_vol.iloc[i - period + 1 : i + 1]
        valid = ~(window_pc.isna() | window_sv.isna())
        if valid.sum() < period // 2:
            lambdas.append(np.nan)
            continue
        sv = window_sv[valid]
        pc = window_pc[valid]
        denom = (sv * sv).sum()
        if denom == 0:
            lambdas.append(np.nan)
        else:
            lambdas.append((pc * sv).sum() / denom)

    df["kyle_lambda"] = lambdas
    return df


@factor(
    "realized_spread",
    category="microstructure",
    output_columns=["realized_spread"],
    delay=5,
)
def realized_spread(df: pd.DataFrame, delay: int = 5) -> pd.DataFrame:
    """实现价差

    衡量做市商实际获利的价差成分。
    """
    if "bid_price_1" not in df.columns or "ask_price_1" not in df.columns:
        df["realized_spread"] = np.nan
        return df

    mid = (df["bid_price_1"] + df["ask_price_1"]) / 2
    future_mid = mid.shift(-delay)
    direction = np.sign(df["last_price"] - mid)
    df["realized_spread"] = 2 * direction * (df["last_price"] - future_mid)
    return df
