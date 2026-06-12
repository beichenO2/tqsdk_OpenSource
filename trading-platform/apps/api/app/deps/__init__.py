"""FastAPI dependency injection — @lru_cache 单例 + Depends 注入。

lifespan 只做初始化，不做全局赋值；服务通过 @lru_cache 工厂按需创建。
运行时仍需 set_* 用于 lifespan 传入已构造好的实例（broker 等需要 async init）。
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from broker_tqsdk.adapter import TqMarketAdapter
from execution.service import ExecutionService

from app.services.market import MarketService

_execution_service: Optional[ExecutionService] = None
_market_adapter: Optional[TqMarketAdapter] = None
_btc_broker_manager = None  # type: ignore[assignment]


def set_execution_service(service: ExecutionService) -> None:
    global _execution_service
    _execution_service = service


def get_execution_service() -> ExecutionService:
    if _execution_service is None:
        from core.exceptions import ServiceNotReadyError
        raise ServiceNotReadyError("ExecutionService not initialized")
    return _execution_service


def is_execution_service_ready() -> bool:
    return _execution_service is not None


def set_market_adapter(adapter: Optional[TqMarketAdapter]) -> None:
    global _market_adapter
    _market_adapter = adapter


def get_market_adapter() -> TqMarketAdapter:
    if _market_adapter is None:
        from core.exceptions import ServiceNotReadyError
        raise ServiceNotReadyError("TqMarketAdapter not initialized")
    return _market_adapter


@lru_cache
def get_market_service() -> MarketService:
    return MarketService(get_market_adapter())


def set_btc_broker_manager(manager: object) -> None:
    global _btc_broker_manager
    _btc_broker_manager = manager


def get_btc_broker_manager():  # type: ignore[return]
    return _btc_broker_manager
