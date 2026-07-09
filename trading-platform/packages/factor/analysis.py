"""因子 IC / RankIC / IR / 衰减 / 相关性 / 去重。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _forward_returns(close: pd.Series, horizon: int = 1) -> pd.Series:
    return close.shift(-horizon) / close - 1.0


def factor_ic(
    factor: pd.Series,
    close: pd.Series,
    *,
    horizon: int = 1,
    method: str = "spearman",
) -> pd.Series:
    """逐期截面不可用时，用滚动窗口近似 IC（单品种时间序列）。

    对单品种 bars：在 rolling window 内计算 factor 与 forward return 的相关。
    """
    fwd = _forward_returns(close, horizon)
    aligned = pd.concat([factor.rename("f"), fwd.rename("r")], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)

    window = max(20, min(60, len(aligned) // 5 or 20))

    def _corr(x: pd.DataFrame) -> float:
        if len(x) < 5:
            return float("nan")
        a, b = x["f"], x["r"]
        if method == "pearson":
            return float(a.corr(b, method="pearson"))
        return float(a.corr(b, method="spearman"))

    # rolling apply on index positions
    ics: list[float] = []
    idx: list[Any] = []
    vals_f = aligned["f"].to_numpy()
    vals_r = aligned["r"].to_numpy()
    index = aligned.index
    for i in range(window - 1, len(aligned)):
        sl_f = vals_f[i - window + 1 : i + 1]
        sl_r = vals_r[i - window + 1 : i + 1]
        if method == "pearson":
            c = np.corrcoef(sl_f, sl_r)[0, 1]
        else:
            # rank spearman
            rf = pd.Series(sl_f).rank().to_numpy()
            rr = pd.Series(sl_r).rank().to_numpy()
            c = np.corrcoef(rf, rr)[0, 1]
        ics.append(float(c) if c == c else float("nan"))
        idx.append(index[i])
    return pd.Series(ics, index=idx, name="ic")


def summarize_ic(ic: pd.Series) -> dict[str, float | int | None]:
    s = ic.dropna()
    if s.empty:
        return {
            "ic_mean": None,
            "ic_std": None,
            "ir": None,
            "ic_positive_ratio": None,
            "n": 0,
        }
    mean = float(s.mean())
    std = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    ir = mean / std if std > 1e-12 else None
    return {
        "ic_mean": mean,
        "ic_std": std,
        "ir": ir,
        "ic_positive_ratio": float((s > 0).mean()),
        "n": int(len(s)),
    }


def ic_decay(
    factor: pd.Series,
    close: pd.Series,
    horizons: list[int] | None = None,
    method: str = "spearman",
) -> list[dict[str, Any]]:
    horizons = horizons or [1, 2, 3, 5, 10]
    out: list[dict[str, Any]] = []
    for h in horizons:
        ic = factor_ic(factor, close, horizon=h, method=method)
        summary = summarize_ic(ic)
        out.append({"horizon": h, **summary})
    return out


def correlation_matrix(factor_df: pd.DataFrame, method: str = "spearman") -> dict[str, Any]:
    """因子值矩阵相关性。返回 labels + matrix。"""
    clean = factor_df.dropna(how="all")
    if clean.empty or clean.shape[1] == 0:
        return {"labels": [], "matrix": []}
    corr = clean.corr(method=method)
    labels = list(corr.columns)
    matrix = [[None if (v != v) else float(v) for v in row] for row in corr.to_numpy()]
    return {"labels": labels, "matrix": matrix}


def deduplicate_factors(
    factor_df: pd.DataFrame,
    *,
    threshold: float = 0.99,
    method: str = "spearman",
) -> dict[str, Any]:
    """相关性 |ρ| ≥ threshold 时保留先出现的列，剔除后者。"""
    labels = list(factor_df.columns)
    if len(labels) <= 1:
        return {"kept": labels, "dropped": [], "pairs": []}

    corr = factor_df.corr(method=method).abs()
    kept: list[str] = []
    dropped: list[str] = []
    pairs: list[dict[str, Any]] = []

    for col in labels:
        if col in dropped:
            continue
        redundant = False
        for k in kept:
            rho = float(corr.loc[col, k]) if col in corr.index and k in corr.columns else 0.0
            if rho >= threshold:
                redundant = True
                dropped.append(col)
                pairs.append({"kept": k, "dropped": col, "abs_corr": rho})
                break
        if not redundant:
            kept.append(col)

    return {"kept": kept, "dropped": dropped, "pairs": pairs, "threshold": threshold}
