"""Shared BTC hypothesis factor builders and turnover suppressions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor.alpha_gate import causal_rolling_zscore

# Calendar-aligned bar counts per timeframe (crypto trades 24/7).
TF_CONFIG: dict[str, dict[str, int]] = {
    "15m": {
        "bars_per_year": 4 * 24 * 365,
        "bars_7d": 7 * 24 * 4,
        "bars_180d": 180 * 24 * 4,
        "bars_24h": 24 * 4,
        "bars_30d": 30 * 24 * 4,
    },
    "1h": {
        "bars_per_year": 24 * 365,
        "bars_7d": 7 * 24,
        "bars_180d": 180 * 24,
        "bars_24h": 24,
        "bars_30d": 30 * 24,
    },
    "4h": {
        "bars_per_year": 6 * 365,
        "bars_7d": 7 * 6,
        "bars_180d": 180 * 6,
        "bars_24h": 6,
        "bars_30d": 30 * 6,
    },
    "1d": {
        "bars_per_year": 365,
        "bars_7d": 7,
        "bars_180d": 180,
        "bars_24h": 1,
        "bars_30d": 30,
    },
}


def bars_for_calendar_days(days: int, cfg: dict[str, int]) -> int:
    """Convert calendar days to bars using the 7d reference in *cfg*."""
    return max(int(round(days * cfg["bars_7d"] / 7)), 1)


def factor_h2_vol_adj_momentum(
    ohlcv: pd.DataFrame,
    cfg: dict[str, int],
    momentum_days: int = 7,
    zscore_days: int = 180,
) -> pd.Series:
    """Momentum / realized vol with causal z-score, clipped to [-1, 1]."""
    close = ohlcv["close"].astype(float)
    bars_mom = bars_for_calendar_days(momentum_days, cfg)
    bars_z = bars_for_calendar_days(zscore_days, cfg)
    roc = close.pct_change(bars_mom)
    vol = (
        close.pct_change()
        .rolling(bars_mom, min_periods=max(bars_mom // 2, 2))
        .std()
        .shift(1)
    )
    raw = roc / vol.replace(0.0, np.nan)
    z = causal_rolling_zscore(raw, bars_z)
    return z.clip(-1, 1).fillna(0.0)


def factor_h4_short_high_momentum(ohlcv: pd.DataFrame, cfg: dict[str, int]) -> pd.Series:
    """Negate 24h momentum when |z| is high; z-score window 30d."""
    close = ohlcv["close"].astype(float)
    bars_24h = cfg["bars_24h"]
    bars_30d = cfg["bars_30d"]
    roc = close.pct_change(bars_24h)
    z = causal_rolling_zscore(roc, bars_30d)
    sign = np.sign(roc).replace(0, 0.0)
    position = -sign * np.minimum(z.abs(), 1.0)
    return position.fillna(0.0)


def apply_suppression_none(target: pd.Series, ohlcv: pd.DataFrame) -> pd.Series:
    return target.clip(-1, 1).fillna(0.0)


def apply_suppression_band(
    target: pd.Series, ohlcv: pd.DataFrame, band: float = 0.3
) -> pd.Series:
    """Rebalance only when |target - current| > band."""
    t = target.fillna(0.0).to_numpy(dtype=float)
    n = len(t)
    pos = np.zeros(n, dtype=float)
    pos[0] = np.clip(t[0], -1, 1)
    for i in range(1, n):
        if abs(t[i] - pos[i - 1]) > band:
            pos[i] = np.clip(t[i], -1, 1)
        else:
            pos[i] = pos[i - 1]
    return pd.Series(pos, index=target.index)


def apply_suppression_min_hold(
    target: pd.Series, ohlcv: pd.DataFrame, hold_days: int = 1
) -> pd.Series:
    """Freeze position for *hold_days* calendar days after each rebalance."""
    t = target.fillna(0.0).to_numpy(dtype=float)
    times = pd.to_datetime(ohlcv["open_time"], utc=True)
    n = len(t)
    pos = np.zeros(n, dtype=float)
    pos[0] = np.clip(t[0], -1, 1)
    hold_until: pd.Timestamp | None = None
    for i in range(1, n):
        ts = times.iloc[i]
        if hold_until is not None and ts < hold_until:
            pos[i] = pos[i - 1]
            continue
        if abs(t[i] - pos[i - 1]) > 1e-9:
            pos[i] = np.clip(t[i], -1, 1)
            hold_until = ts + pd.Timedelta(days=hold_days)
        else:
            pos[i] = pos[i - 1]
    return pd.Series(pos, index=target.index)


def apply_suppression(
    target: pd.Series, ohlcv: pd.DataFrame, mode: str
) -> pd.Series:
    if mode == "none":
        return apply_suppression_none(target, ohlcv)
    if mode.startswith("band_"):
        band_val = float(mode.split("_", 1)[1])
        return apply_suppression_band(target, ohlcv, band=band_val)
    if mode == "min_hold_1d":
        return apply_suppression_min_hold(target, ohlcv, hold_days=1)
    raise ValueError(f"Unknown suppression mode: {mode}")


def evaluate_position_gate(
    position: pd.Series,
    ohlcv: pd.DataFrame,
    bars_per_year: int,
    cost_bps: float = 5.0,
) -> dict[str, Any]:
    """Run AlphaGate and return flattened metrics for research scripts."""
    from factor.alpha_gate import AlphaGate

    gate = AlphaGate(one_way_cost_bps=cost_bps, bars_per_year=bars_per_year)
    report = gate.evaluate(position, ohlcv)
    n_pass = sum(g.passed for g in report.gates.values())
    m = report.metrics
    return {
        "net_return_5bp": m.get("net_return_5bp", m.get("total_net_return", float("nan"))),
        "net_return_2bp": m.get("net_return_2bp", float("nan")),
        "net_sharpe": m.get("net_sharpe", float("nan")),
        "annual_turnover": m.get("annual_turnover", float("nan")),
        "trade_expectancy_bp": m.get("trade_expectancy_bp", float("nan")),
        "cost_ratio": m.get("cost_ratio", float("nan")),
        "gates_passed": n_pass,
        "verdict": report.verdict,
        "total_net_return": m.get("total_net_return", float("nan")),
    }
