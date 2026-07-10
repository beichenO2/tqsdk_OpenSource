"""因子合成：等权 / IC 加权 / 简易正交化 / AlphaForge 动态时变权重。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def combine_equal_weight(factor_df: pd.DataFrame) -> pd.Series:
    """等权合成（先 zscore 再均值）。"""
    z = factor_df.apply(lambda s: (s - s.mean()) / (s.std(ddof=0) or 1.0), axis=0)
    return z.mean(axis=1).rename("combined_equal")


def combine_ic_weight(factor_df: pd.DataFrame, ic_means: dict[str, float]) -> pd.Series:
    """按 |IC| 加权合成。"""
    cols = [c for c in factor_df.columns if c in ic_means]
    if not cols:
        return combine_equal_weight(factor_df)
    weights = np.array([abs(float(ic_means[c])) for c in cols], dtype=float)
    if weights.sum() < 1e-12:
        weights = np.ones(len(cols))
    weights = weights / weights.sum()
    z = factor_df[cols].apply(lambda s: (s - s.mean()) / (s.std(ddof=0) or 1.0), axis=0)
    return (z * weights).sum(axis=1).rename("combined_ic")


def orthogonalize(factor_df: pd.DataFrame) -> pd.DataFrame:
    """对因子列做 Gram-Schmidt 风格顺序正交（按列顺序）。"""
    cols = list(factor_df.columns)
    if not cols:
        return factor_df.copy()
    mat = factor_df.dropna().to_numpy(dtype=float)
    if mat.size == 0:
        return pd.DataFrame(columns=cols)
    index = factor_df.dropna().index
    out = np.zeros_like(mat)
    for i in range(mat.shape[1]):
        v = mat[:, i].copy()
        for j in range(i):
            u = out[:, j]
            denom = np.dot(u, u)
            if denom < 1e-12:
                continue
            v = v - (np.dot(u, v) / denom) * u
        out[:, i] = v
    return pd.DataFrame(out, index=index, columns=[f"{c}_orth" for c in cols])


def _rolling_spearman(a: np.ndarray, b: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """Causal rolling Spearman IC; NaN until min_periods observations available."""
    n = len(a)
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        start = max(0, i - window + 1)
        sl_a = a[start : i + 1]
        sl_b = b[start : i + 1]
        mask = np.isfinite(sl_a) & np.isfinite(sl_b)
        if int(mask.sum()) < min_periods:
            continue
        xa, xb = sl_a[mask], sl_b[mask]
        ra = pd.Series(xa).rank().to_numpy(dtype=float)
        rb = pd.Series(xb).rank().to_numpy(dtype=float)
        if float(np.std(ra)) < 1e-12 or float(np.std(rb)) < 1e-12:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.corrcoef(ra, rb)[0, 1]
        if c == c:
            out[i] = float(c)
    return out


def _causal_zscore(factor_df: pd.DataFrame) -> pd.DataFrame:
    """Expanding z-score (no future mean/std)."""
    mu = factor_df.expanding(min_periods=2).mean()
    sig = factor_df.expanding(min_periods=2).std(ddof=0)
    return (factor_df - mu) / sig.replace(0.0, np.nan)


def _normalize_signed_weights(signed: pd.DataFrame, weight_floor: float) -> pd.DataFrame:
    """|w| sum to 1; keep sign; floor tiny |w| then renormalize."""
    mag = signed.abs()
    if weight_floor > 0:
        mag = mag.where(mag >= weight_floor, 0.0)
    row_sum = mag.sum(axis=1)
    mag = mag.div(row_sum.replace(0.0, np.nan), axis=0)
    signs = np.sign(signed.to_numpy(dtype=float))
    signs[signs == 0] = 0.0
    return pd.DataFrame(mag.to_numpy() * signs, index=signed.index, columns=signed.columns)

def rolling_ic_weights(
    factor_df: pd.DataFrame,
    forward_returns: pd.Series,
    *,
    window: int = 120,
    min_periods: int = 60,
    smoothing_halflife: int = 20,
    weight_floor: float = 0.0,
) -> pd.DataFrame:
    """Trailing |IC|-normalized signed weights (AlphaForge-style).

    No look-ahead: weight at t uses rolling IC ending at t-1 only
    (IC window correlates factor with forward returns known by t).
    """
    aligned = factor_df.copy()
    fwd = forward_returns.reindex(aligned.index)
    raw_ic = pd.DataFrame(index=aligned.index, columns=aligned.columns, dtype=float)
    fwd_vals = fwd.to_numpy(dtype=float)
    for col in aligned.columns:
        raw_ic[col] = _rolling_spearman(
            aligned[col].to_numpy(dtype=float),
            fwd_vals,
            window=window,
            min_periods=min_periods,
        )

    # Shift by 1: t uses IC available through t-1
    lagged_ic = raw_ic.shift(1)

    # EMA smooth on lagged IC before converting to weights (reduces jitter)
    if smoothing_halflife and smoothing_halflife > 0:
        smoothed = lagged_ic.ewm(halflife=smoothing_halflife, min_periods=1, adjust=False).mean()
    else:
        smoothed = lagged_ic

    return _normalize_signed_weights(smoothed, weight_floor)


def dynamic_combine(
    factor_df: pd.DataFrame,
    forward_returns: pd.Series,
    *,
    window: int = 120,
    min_periods: int = 60,
    smoothing_halflife: int = 20,
    weight_floor: float = 0.0,
) -> pd.Series:
    """Z-score factors then combine with time-varying IC weights (no look-ahead)."""
    weights = rolling_ic_weights(
        factor_df,
        forward_returns,
        window=window,
        min_periods=min_periods,
        smoothing_halflife=smoothing_halflife,
        weight_floor=weight_floor,
    )
    z = _causal_zscore(factor_df)
    combined = (z * weights).sum(axis=1)
    return combined.rename("combined_dynamic")


def compare_static_vs_dynamic(
    factor_df: pd.DataFrame,
    close: pd.Series,
    *,
    horizon: int = 1,
    window: int = 120,
    min_periods: int = 60,
    smoothing_halflife: int = 20,
    weight_floor: float = 0.0,
) -> dict[str, Any]:
    """Compare static |IC| weighting vs dynamic trailing-IC weighting."""
    from factor.analysis import factor_ic, summarize_ic

    close = close.reindex(factor_df.index)
    fwd = close.shift(-horizon) / close - 1.0

    ic_means: dict[str, float] = {}
    for col in factor_df.columns:
        summary = summarize_ic(factor_ic(factor_df[col], close, horizon=horizon))
        if summary["ic_mean"] is not None:
            ic_means[col] = float(summary["ic_mean"])

    static = combine_ic_weight(factor_df, ic_means)
    dynamic = dynamic_combine(
        factor_df,
        fwd,
        window=window,
        min_periods=min_periods,
        smoothing_halflife=smoothing_halflife,
        weight_floor=weight_floor,
    )

    def _ic_ir(series: pd.Series) -> dict[str, float | None]:
        ic = factor_ic(series, close, horizon=horizon)
        s = summarize_ic(ic)
        return {"ic": s["ic_mean"], "ir": s["ir"]}

    weights = rolling_ic_weights(
        factor_df,
        fwd,
        window=window,
        min_periods=min_periods,
        smoothing_halflife=smoothing_halflife,
        weight_floor=weight_floor,
    )
    last_row = weights.dropna(how="all").iloc[-1] if len(weights.dropna(how="all")) else pd.Series(dtype=float)
    weights_last = {str(k): (None if v != v else float(v)) for k, v in last_row.items()}

    return {
        "static": _ic_ir(static),
        "dynamic": _ic_ir(dynamic),
        "weights_last": weights_last,
    }
