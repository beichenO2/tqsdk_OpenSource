"""Strategy abstraction layer — manual, semi-auto, and fully-automatic modes."""

from .base import BaseStrategy, OrderSide, Signal, SignalType, StrategyConfig, StrategyState
from .registry import StrategyRegistry

__all__ = [
    "BaseStrategy",
    "OrderSide",
    "Signal",
    "SignalType",
    "StrategyConfig",
    "StrategyState",
    "StrategyRegistry",
]


def load_all_strategies() -> None:
    """Import all strategy subpackages so their @auto_register decorators run."""
    import importlib
    import logging
    _logger = logging.getLogger(__name__)
    for mod in ("strategy.futures", "strategy.btc", "strategy.templates", "crypto.strategies"):
        try:
            importlib.import_module(mod)
        except ImportError as exc:
            _logger.warning("Failed to load strategy subpackage %s: %s", mod, exc)
