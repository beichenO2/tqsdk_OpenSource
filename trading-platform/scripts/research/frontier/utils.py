"""Shared utilities for frontier research strategies."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

logger = logging.getLogger(__name__)


def load_crypto_bars(symbol: str = "BTCUSDT", timeframe: str = "1h", weeks: int = 80) -> pd.DataFrame:
    from datahub.crypto_loader import CryptoDataLoader
    loader = CryptoDataLoader()
    try:
        bars = loader.load_with_funding(symbol, timeframe)
    except Exception:
        bars = loader.load(symbol, timeframe)
    if weeks:
        hours_per_week = 7 * 24 if timeframe == "1h" else 7 * 24 * 60
        bars = bars.tail(weeks * hours_per_week).reset_index(drop=True)
    logger.info("Loaded %d bars for %s %s", len(bars), symbol, timeframe)
    return bars


def compute_metrics(
    trades: list[dict],
    initial_capital: float,
    final_capital: float,
    equity_curve: list[float],
) -> dict:
    """Compute standard backtest metrics from trade list and equity curve."""
    n_trades = len(trades)
    if n_trades == 0:
        return {
            "total_return_pct": 0.0, "n_trades": 0, "win_rate": 0.0,
            "profit_factor": 0.0, "max_drawdown_pct": 0.0, "sharpe": 0.0,
            "calmar": 0.0, "avg_win_pct": 0.0, "avg_loss_pct": 0.0,
            "final_capital": round(final_capital, 2),
        }

    total_return = (final_capital - initial_capital) / initial_capital * 100
    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    win_rate = len(wins) / n_trades * 100
    avg_win = np.mean([t["pnl_pct"] for t in wins]) * 100 if wins else 0
    avg_loss = np.mean([abs(t["pnl_pct"]) for t in losses]) * 100 if losses else 0

    gross_profit = sum(t["pnl_pct"] for t in wins)
    gross_loss = abs(sum(t["pnl_pct"] for t in losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else 999.0

    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    returns_arr = np.array([t["pnl_pct"] for t in trades])
    sharpe = 0.0
    if len(returns_arr) > 1 and np.std(returns_arr) > 1e-10:
        sharpe = float(np.mean(returns_arr) / np.std(returns_arr) * np.sqrt(252 / max(1, n_trades)))

    calmar = abs(total_return / (max_dd * 100)) if max_dd > 0 else 999.0

    return {
        "total_return_pct": round(total_return, 2),
        "n_trades": n_trades,
        "win_rate": round(win_rate, 1),
        "profit_factor": round(pf, 2),
        "max_drawdown_pct": round(max_dd * 100, 1),
        "sharpe": round(sharpe, 2),
        "calmar": round(calmar, 2),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "final_capital": round(final_capital, 2),
    }


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add common technical indicator columns to OHLCV DataFrame."""
    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    v = df["volume"].values.astype(np.float64)

    df = df.copy()

    df["returns"] = pd.Series(c).pct_change().fillna(0)
    df["log_returns"] = np.log(c / np.roll(c, 1)).clip(-1, 1)
    df.loc[df.index[0], "log_returns"] = 0

    for w in [7, 14, 21, 50]:
        df[f"sma_{w}"] = pd.Series(c).rolling(w).mean().fillna(c[0])
        df[f"ema_{w}"] = pd.Series(c).ewm(span=w).mean().fillna(c[0])

    df["bb_mid"] = pd.Series(c).rolling(20).mean()
    bb_std = pd.Series(c).rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std
    df["bb_pctb"] = ((c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)).clip(-1, 2)

    delta = pd.Series(c).diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss_s = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["rsi"] = (100 - 100 / (1 + gain / (loss_s + 1e-10))).fillna(50) / 100

    ema12 = pd.Series(c).ewm(span=12).mean()
    ema26 = pd.Series(c).ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    df["macd"] = macd.fillna(0)
    df["macd_signal"] = signal.fillna(0)
    df["macd_hist"] = (macd - signal).fillna(0)

    tr = np.maximum(h - low, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(low - np.roll(c, 1))))
    df["atr_14"] = pd.Series(tr).rolling(14).mean().fillna(0)

    df["vol_ratio"] = v / (pd.Series(v).rolling(20).mean().fillna(1) + 1e-10)
    df["high_low_range"] = (h - low) / (c + 1e-10)

    df["volatility_20"] = pd.Series(c).pct_change().rolling(20).std().fillna(0)

    df.fillna(method="bfill", inplace=True)
    df.fillna(0, inplace=True)

    return df
