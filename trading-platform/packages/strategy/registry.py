"""策略注册表 - 管理策略类型注册与实例化。

与 Ch39（策略注册中心）集成后，此模块将作为本地注册的适配层。
"""

from __future__ import annotations

import logging

from .base import BaseStrategy, StrategyConfig

logger = logging.getLogger(__name__)

_STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {}
_STRATEGY_INSTANCES: dict[str, StrategyConfig] = {}


class StrategyRegistry:
    """策略类型注册和查找。"""

    @staticmethod
    def register(name: str, strategy_cls: type[BaseStrategy]) -> None:
        if name in _STRATEGY_REGISTRY:
            logger.warning("策略 '%s' 已注册，将被覆盖", name)
        _STRATEGY_REGISTRY[name] = strategy_cls
        logger.info("注册策略: %s -> %s", name, strategy_cls.__name__)

    @staticmethod
    def get(name: str) -> type[BaseStrategy] | None:
        return _STRATEGY_REGISTRY.get(name)

    @staticmethod
    def create(name: str, config: StrategyConfig) -> BaseStrategy:
        cls = _STRATEGY_REGISTRY.get(name)
        if cls is None:
            raise KeyError(f"未注册的策略类型: {name}")
        return cls(config)

    @staticmethod
    def list_registered() -> list[str]:
        return list(_STRATEGY_REGISTRY.keys())

    @staticmethod
    def unregister(name: str) -> bool:
        return _STRATEGY_REGISTRY.pop(name, None) is not None

    # --- 策略实例（API CRUD）---

    @staticmethod
    def add_instance(config: StrategyConfig) -> StrategyConfig:
        _STRATEGY_INSTANCES[config.strategy_id] = config
        logger.info("保存策略实例: %s (%s)", config.strategy_id, config.name)
        return config

    @staticmethod
    def list_instances() -> list[StrategyConfig]:
        return list(_STRATEGY_INSTANCES.values())

    @staticmethod
    def get_instance(strategy_id: str) -> StrategyConfig | None:
        return _STRATEGY_INSTANCES.get(strategy_id)

    @staticmethod
    def delete_instance(strategy_id: str) -> bool:
        return _STRATEGY_INSTANCES.pop(strategy_id, None) is not None

    @staticmethod
    def set_instance_enabled(strategy_id: str, enabled: bool) -> StrategyConfig | None:
        cfg = _STRATEGY_INSTANCES.get(strategy_id)
        if cfg is None:
            return None
        updated = cfg.model_copy(update={"enabled": enabled})
        _STRATEGY_INSTANCES[strategy_id] = updated
        return updated


def auto_register(name: str):
    """装饰器：自动将策略类注册到全局注册表。"""
    def decorator(cls: type[BaseStrategy]) -> type[BaseStrategy]:
        StrategyRegistry.register(name, cls)
        return cls
    return decorator
