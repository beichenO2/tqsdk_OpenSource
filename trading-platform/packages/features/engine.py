"""特征引擎 - 批量计算因子、标准化、特征选择"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from features.registry import FactorRegistry

logger = logging.getLogger(__name__)


class FeatureEngine:
    """特征工程引擎

    提供批量因子计算、标准化、特征矩阵生成功能。
    与 DataHub 的 Gold 层对接。
    """

    def __init__(self, registry: Optional[FactorRegistry] = None):
        self._registry = registry or FactorRegistry()

    def compute_factors(
        self,
        df: pd.DataFrame,
        factor_names: list[str],
        params: Optional[dict[str, dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        """批量计算多个因子

        Args:
            df: 原始 OHLCV 数据（需包含 open, high, low, close, volume 列）
            factor_names: 要计算的因子名称列表
            params: 因子参数覆盖，如 {"rsi": {"period": 21}}
        """
        params = params or {}
        result = df.copy()

        for name in factor_names:
            try:
                factor_params = params.get(name, {})
                result = self._registry.compute(name, result, **factor_params)
                logger.debug("Computed factor: %s", name)
            except Exception as e:
                logger.error("Failed to compute factor %s: %s", name, e)
                raise

        return result

    def compute_all(
        self,
        df: pd.DataFrame,
        categories: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """计算指定类别（或全部）的所有已注册因子"""
        factors = self._registry.list_factors()
        if categories:
            factors = [f for f in factors if f.category in categories]

        names = [f.name for f in factors]
        return self.compute_factors(df, names)

    @staticmethod
    def normalize(
        df: pd.DataFrame,
        columns: list[str],
        method: str = "zscore",
        window: Optional[int] = None,
    ) -> pd.DataFrame:
        """特征标准化

        Args:
            method: 'zscore', 'minmax', 'rank'
            window: 滚动窗口大小（None 表示全局标准化）
        """
        result = df.copy()

        for col in columns:
            if col not in result.columns:
                continue

            if method == "zscore":
                if window:
                    roll_mean = result[col].rolling(window).mean()
                    roll_std = result[col].rolling(window).std()
                    result[f"{col}_norm"] = (result[col] - roll_mean) / roll_std.replace(0, np.nan)
                else:
                    mean = result[col].mean()
                    std = result[col].std()
                    result[f"{col}_norm"] = (result[col] - mean) / (std if std != 0 else np.nan)

            elif method == "minmax":
                if window:
                    roll_min = result[col].rolling(window).min()
                    roll_max = result[col].rolling(window).max()
                    denom = (roll_max - roll_min).replace(0, np.nan)
                    result[f"{col}_norm"] = (result[col] - roll_min) / denom
                else:
                    col_min = result[col].min()
                    col_max = result[col].max()
                    denom = col_max - col_min
                    result[f"{col}_norm"] = (
                        (result[col] - col_min) / denom if denom != 0 else 0.5
                    )

            elif method == "rank":
                if window:
                    result[f"{col}_norm"] = result[col].rolling(window).apply(
                        lambda x: pd.Series(x).rank(pct=True).iloc[-1],
                        raw=False,
                    )
                else:
                    result[f"{col}_norm"] = result[col].rank(pct=True)

        return result

    @staticmethod
    def build_feature_matrix(
        df: pd.DataFrame,
        feature_columns: list[str],
        target_column: Optional[str] = None,
        lookahead: int = 1,
        dropna: bool = True,
    ) -> tuple[pd.DataFrame, Optional[pd.Series]]:
        """构建用于模型训练的特征矩阵

        Args:
            df: 包含特征列的 DataFrame
            feature_columns: 特征列名
            target_column: 目标变量列名（若指定，会自动创建前瞻标签）
            lookahead: 前瞻期数
            dropna: 是否丢弃含 NaN 的行
        """
        X = df[feature_columns].copy()
        y = None

        if target_column and target_column in df.columns:
            y = df[target_column].shift(-lookahead)

        if dropna:
            if y is not None:
                valid = X.notna().all(axis=1) & y.notna()
                X = X[valid]
                y = y[valid]
            else:
                X = X.dropna()

        return X, y

    def get_feature_importance(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
    ) -> pd.Series:
        """简单相关性排序的特征重要度"""
        correlations = {}
        for col in feature_columns:
            if col in df.columns and target_column in df.columns:
                corr = df[col].corr(df[target_column])
                correlations[col] = abs(corr)

        return pd.Series(correlations).sort_values(ascending=False)
