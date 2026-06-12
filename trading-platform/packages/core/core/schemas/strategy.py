"""策略配置与元信息 schema."""

from __future__ import annotations

from pydantic import BaseModel, Field


class StrategyConfig(BaseModel):
    """策略运行时配置 (API schema — 与 strategy.base.StrategyConfig 字段对齐)."""

    strategy_id: str
    name: str
    version: str = "1.0.0"
    symbols: list[str] = Field(default_factory=list)
    params: dict[str, float | int | str | bool] = Field(default_factory=dict)
    risk_limits: dict[str, float] = Field(default_factory=dict)
    enabled: bool = True
    max_position: int = 10
    capital: float = 1_000_000.0


class StrategyMeta(BaseModel):
    """策略元信息，用于注册中心."""

    name: str
    version: str
    author: str = ""
    description: str = ""
    asset_class: str = "FUTURES"
    tags: list[str] = Field(default_factory=list)
