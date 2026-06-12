"""Crypto strategy registry — market-isolated wrapper over the shared registry.

Provides crypto-specific registration and lookup that won't interfere with
futures strategies. Uses the shared StrategyRegistry under the hood but
prefixes crypto strategies with 'crypto:' namespace internally.
"""

from __future__ import annotations

import logging

from strategy.base import BaseStrategy, StrategyConfig
from strategy.registry import StrategyRegistry, auto_register as _shared_auto_register

logger = logging.getLogger(__name__)

_CRYPTO_PREFIX = ""

CRYPTO_STRATEGIES: set[str] = set()


class CryptoStrategyRegistry:
    """Market-isolated registry for crypto strategies."""

    @staticmethod
    def register(name: str, strategy_cls: type[BaseStrategy]) -> None:
        key = f"{_CRYPTO_PREFIX}{name}"
        StrategyRegistry.register(key, strategy_cls)
        CRYPTO_STRATEGIES.add(key)

    @staticmethod
    def get(name: str) -> type[BaseStrategy] | None:
        return StrategyRegistry.get(f"{_CRYPTO_PREFIX}{name}")

    @staticmethod
    def create(name: str, config: StrategyConfig) -> BaseStrategy:
        return StrategyRegistry.create(f"{_CRYPTO_PREFIX}{name}", config)

    @staticmethod
    def list_registered() -> list[str]:
        """List only crypto strategies (not futures)."""
        return [
            name.removeprefix(_CRYPTO_PREFIX)
            for name in StrategyRegistry.list_registered()
            if name in CRYPTO_STRATEGIES
        ]


def crypto_auto_register(name: str):
    """Decorator: register a crypto strategy and track it in CRYPTO_STRATEGIES."""
    def decorator(cls: type[BaseStrategy]) -> type[BaseStrategy]:
        key = f"{_CRYPTO_PREFIX}{name}"
        StrategyRegistry.register(key, cls)
        CRYPTO_STRATEGIES.add(key)
        return cls
    return decorator
