"""委托单类型枚举 — 与平台统一 OrderRequest / Broker 层对齐."""

from enum import StrEnum


class OrderType(StrEnum):
    LIMIT = "limit"
    MARKET = "market"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
