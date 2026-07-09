"""Alphalens 风格截面 IC 与分位收益。

数据约定：
- factor_panel: DataFrame index=datetime, columns=asset, values=factor
- close_panel:  同形状收盘价面板
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor.analysis import summarize_ic


def forward_returns_panel(close: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """多品种远期收益：close.shift(-h)/close - 1。"""
    return close.shift(-horizon) / close - 1.0


def cross_sectional_ic(
    factor_panel: pd.DataFrame,
    close_panel: pd.DataFrame,
    *,
    horizon: int = 1,
    method: str = "spearman",
    min_assets: int = 3,
) -> pd.Series:
    """逐期截面 IC（Alphalens factor_information_coefficient 简化版）。

    每个时间戳上，对当时有值的资产做 factor vs forward_return 的秩相关。
    """
    fwd = forward_returns_panel(close_panel, horizon)
    # align columns
    cols = sorted(set(factor_panel.columns) & set(fwd.columns))
    if len(cols) < min_assets:
        return pd.Series(dtype=float, name="cs_ic")

    f = factor_panel[cols]
    r = fwd[cols]
    common_idx = f.index.intersection(r.index)
    f = f.loc[common_idx]
    r = r.loc[common_idx]

    ics: list[float] = []
    idx: list[Any] = []
    for ts in common_idx:
        row_f = f.loc[ts]
        row_r = r.loc[ts]
        mask = row_f.notna() & row_r.notna()
        if int(mask.sum()) < min_assets:
            continue
        a = row_f[mask]
        b = row_r[mask]
        if method == "pearson":
            c = float(a.corr(b, method="pearson"))
        else:
            c = float(a.corr(b, method="spearman"))
        if c == c:  # not NaN
            ics.append(c)
            idx.append(ts)
    return pd.Series(ics, index=idx, name="cs_ic")


def quantile_returns(
    factor_panel: pd.DataFrame,
    close_panel: pd.DataFrame,
    *,
    horizon: int = 1,
    quantiles: int = 5,
    min_assets: int = 5,
) -> dict[str, Any]:
    """分位多空收益（Alphalens 风格简化）。

    每期按因子值分 quantiles 组，计算组内等权远期收益均值；
    返回各分位平均收益 + long-short (Q_high - Q_low)。
    """
    fwd = forward_returns_panel(close_panel, horizon)
    cols = sorted(set(factor_panel.columns) & set(fwd.columns))
    if len(cols) < min_assets:
        return {"quantiles": quantiles, "mean_returns": {}, "long_short": None, "n_periods": 0}

    f = factor_panel[cols]
    r = fwd[cols]
    common_idx = f.index.intersection(r.index)

    bucket_rets: dict[int, list[float]] = {q: [] for q in range(1, quantiles + 1)}
    ls: list[float] = []

    for ts in common_idx:
        row_f = f.loc[ts]
        row_r = r.loc[ts]
        mask = row_f.notna() & row_r.notna()
        if int(mask.sum()) < min_assets:
            continue
        a = row_f[mask]
        b = row_r[mask]
        try:
            # rank into quantiles 1..Q
            ranks = a.rank(method="first")
            q_labels = pd.qcut(ranks, quantiles, labels=False, duplicates="drop") + 1
        except ValueError:
            continue
        for q in range(1, quantiles + 1):
            sel = q_labels == q
            if not sel.any():
                continue
            bucket_rets[q].append(float(b[sel].mean()))
        q_hi = b[q_labels == quantiles]
        q_lo = b[q_labels == 1]
        if len(q_hi) and len(q_lo):
            ls.append(float(q_hi.mean() - q_lo.mean()))

    mean_returns = {
        f"Q{q}": (float(np.mean(v)) if v else None) for q, v in bucket_rets.items()
    }
    return {
        "quantiles": quantiles,
        "horizon": horizon,
        "mean_returns": mean_returns,
        "long_short": float(np.mean(ls)) if ls else None,
        "long_short_std": float(np.std(ls, ddof=1)) if len(ls) > 1 else None,
        "n_periods": len(ls),
    }


def analyze_cross_section(
    factor_panel: pd.DataFrame,
    close_panel: pd.DataFrame,
    *,
    horizon: int = 1,
    quantiles: int = 5,
    method: str = "spearman",
) -> dict[str, Any]:
    """截面 IC 摘要 + 分位收益一站式。"""
    ic = cross_sectional_ic(
        factor_panel, close_panel, horizon=horizon, method=method
    )
    summary = summarize_ic(ic)
    qret = quantile_returns(
        factor_panel, close_panel, horizon=horizon, quantiles=quantiles
    )
    ic_tail = ic.dropna().tail(80)
    return {
        "mode": "cross_sectional",
        "horizon": horizon,
        "summary": summary,
        "quantile_returns": qret,
        "ic_series": [
            {
                "t": (idx.isoformat() if hasattr(idx, "isoformat") else str(idx)),
                "v": float(v),
            }
            for idx, v in ic_tail.items()
        ],
        "n_assets": int(len(set(factor_panel.columns) & set(close_panel.columns))),
    }
