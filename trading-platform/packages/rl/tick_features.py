"""Tick-level feature extractor for RL environments.

Aggregates raw trade-by-trade data into microstructure features:
- Volume imbalance (buy vs sell pressure)
- Trade intensity (trades per second)
- VWAP deviation
- Price momentum at tick level
- Volume clustering (large trade detection)

Storage layout expected: data/tick/{source}/{YYYY-MM-DD}/{symbol}.parquet
Columns: timestamp, trade_id, price, quantity, is_buyer_maker
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
TICK_ROOT = DATA_ROOT / "tick"


def load_tick_data(
    symbol: str,
    source: str = "crypto",
    date_str: str | None = None,
) -> pd.DataFrame:
    """Load tick data for a symbol, optionally filtered by date."""
    base = TICK_ROOT / source
    if not base.exists():
        return pd.DataFrame()

    frames = []
    dirs = [base / date_str] if date_str else sorted(base.iterdir())

    for d in dirs:
        if not d.is_dir():
            continue
        path = d / f"{symbol.lower()}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def aggregate_tick_features(
    df: pd.DataFrame,
    window_sec: int = 300,
) -> pd.DataFrame:
    """Aggregate raw ticks into fixed-interval microstructure features.

    Parameters
    ----------
    df
        Raw tick dataframe with columns: timestamp, price, quantity, is_buyer_maker
    window_sec
        Aggregation window in seconds (default: 300 = 5 min, matching OHLCV)

    Returns
    -------
    DataFrame with one row per window and the following features:
        vwap, volume_imbalance, trade_intensity, large_trade_ratio,
        price_momentum, spread_proxy, open, high, low, close, volume
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    origin = df["timestamp"].iloc[0]
    df["elapsed"] = (df["timestamp"] - origin).dt.total_seconds()
    df["window"] = (df["elapsed"] // window_sec).astype(int)

    rows = []
    for wid, grp in df.groupby("window"):
        prices = grp["price"].values
        qtys = grp["quantity"].values
        is_buyer_maker = grp["is_buyer_maker"].values

        buy_vol = qtys[~is_buyer_maker].sum()
        sell_vol = qtys[is_buyer_maker].sum()
        total_vol = buy_vol + sell_vol

        volume_imbalance = (buy_vol - sell_vol) / max(total_vol, 1e-12)

        n_trades = len(grp)
        ts_range = grp["elapsed"].max() - grp["elapsed"].min()
        trade_intensity = n_trades / max(ts_range, 1)

        notional = prices * qtys
        vwap = notional.sum() / max(total_vol, 1e-12)

        large_threshold = np.percentile(qtys, 90) if len(qtys) > 10 else qtys.max()
        large_trade_ratio = qtys[qtys >= large_threshold].sum() / max(total_vol, 1e-12)

        price_momentum = (prices[-1] - prices[0]) / max(prices[0], 1e-12)

        sorted_prices = np.sort(np.unique(prices))
        if len(sorted_prices) >= 2:
            spreads = np.diff(sorted_prices)
            spread_proxy = float(np.median(spreads)) / max(vwap, 1e-12)
        else:
            spread_proxy = 0.0

        rows.append({
            "window_start": grp["timestamp"].iloc[0],
            "open": prices[0],
            "high": prices.max(),
            "low": prices.min(),
            "close": prices[-1],
            "volume": total_vol,
            "vwap": vwap,
            "volume_imbalance": volume_imbalance,
            "trade_intensity": trade_intensity,
            "large_trade_ratio": large_trade_ratio,
            "price_momentum": price_momentum,
            "spread_proxy": spread_proxy,
            "n_trades": n_trades,
            "buy_volume": buy_vol,
            "sell_volume": sell_vol,
        })

    return pd.DataFrame(rows)


def make_rl_features(
    symbol: str = "btcusdt",
    source: str = "crypto",
    window_sec: int = 300,
    date_str: str | None = None,
) -> np.ndarray | None:
    """End-to-end: load tick data → aggregate → return feature array for RL.

    Returns numpy array of shape (N, 11) with columns:
        open, high, low, close, volume, vwap, volume_imbalance,
        trade_intensity, large_trade_ratio, price_momentum, spread_proxy
    """
    df = load_tick_data(symbol, source, date_str)
    if df.empty:
        logger.warning("No tick data for %s/%s", source, symbol)
        return None

    features = aggregate_tick_features(df, window_sec)
    if features.empty:
        return None

    cols = [
        "open", "high", "low", "close", "volume",
        "vwap", "volume_imbalance", "trade_intensity",
        "large_trade_ratio", "price_momentum", "spread_proxy",
    ]
    return features[cols].values.astype(np.float64)
