"""因子注册中心 - 可扩展的因子注册与发现机制"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FactorMeta:
    """因子元信息"""
    name: str
    category: str
    description: str
    params: dict[str, Any]
    output_columns: list[str]
    compute_fn: Callable[..., pd.DataFrame]


class FactorRegistry:
    """全局因子注册中心

    通过装饰器或手动注册方式添加因子，支持按类别检索。
    """

    _instance: Optional[FactorRegistry] = None
    _factors: dict[str, FactorMeta] = {}

    def __new__(cls) -> FactorRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._factors = {}
        return cls._instance

    def register(
        self,
        name: str,
        category: str,
        description: str,
        output_columns: list[str],
        compute_fn: Callable[..., pd.DataFrame],
        **default_params: Any,
    ) -> None:
        """注册一个因子"""
        if name in self._factors:
            logger.warning("Factor '%s' already registered, will be overwritten", name)
        self._factors[name] = FactorMeta(
            name=name,
            category=category,
            description=description,
            params=default_params,
            output_columns=output_columns,
            compute_fn=compute_fn,
        )
        logger.debug("Registered factor: %s [%s]", name, category)

    def get(self, name: str) -> FactorMeta:
        if name not in self._factors:
            raise KeyError(f"Factor '{name}' not registered")
        return self._factors[name]

    def list_factors(self, category: Optional[str] = None) -> list[FactorMeta]:
        factors = list(self._factors.values())
        if category:
            factors = [f for f in factors if f.category == category]
        return factors

    def compute(self, name: str, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """计算指定因子"""
        meta = self.get(name)
        params = {**meta.params, **kwargs}
        return meta.compute_fn(df, **params)

    @property
    def categories(self) -> list[str]:
        return sorted(set(f.category for f in self._factors.values()))


def factor(
    name: str,
    category: str = "custom",
    description: str = "",
    output_columns: Optional[list[str]] = None,
    **default_params: Any,
) -> Callable:
    """因子注册装饰器

    Example:
        @factor("my_factor", category="technical", output_columns=["my_factor"])
        def compute_my_factor(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
            df["my_factor"] = df["close"].rolling(period).mean()
            return df
    """
    def decorator(fn: Callable) -> Callable:
        cols = output_columns or [name]
        registry = FactorRegistry()
        registry.register(
            name=name,
            category=category,
            description=description or fn.__doc__ or "",
            output_columns=cols,
            compute_fn=fn,
            **default_params,
        )

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> pd.DataFrame:
            return fn(*args, **kwargs)

        return wrapper

    return decorator
