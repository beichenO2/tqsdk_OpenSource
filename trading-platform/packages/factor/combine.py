"""因子合成：等权 / IC 加权 / 简易正交化。"""

from __future__ import annotations

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
