"""Local live-simulation matching engine for paper trading.

Heavy modules (engine, order_book, shadow) require core.* which needs
special PYTHONPATH setup. Light modules (account_manager, paper_scheduler,
reporter, strategy_factory) are always importable.
"""

from .account_manager import AccountManager, SimAccount
from .models import ExecutionQuality, Fill, SimConfig

__all__ = [
    "AccountManager",
    "ExecutionQuality",
    "Fill",
    "SimAccount",
    "SimConfig",
]


def __getattr__(name: str):
    """Lazy-load modules that depend on core.* (OrderBook, engine, shadow)."""
    _lazy = {
        "SimMatchingEngine": (".engine", "SimMatchingEngine"),
        "SubmitOrderResult": (".engine", "SubmitOrderResult"),
        "OrderBook": (".order_book", "OrderBook"),
        "aggressor_is_buy": (".order_book", "aggressor_is_buy"),
        "ShadowTrader": (".shadow", "ShadowTrader"),
        "PaperScheduler": (".paper_scheduler", "PaperScheduler"),
        "PaperReporter": (".reporter", "PaperReporter"),
        "create_all_strategies": (".strategy_factory", "create_all_strategies"),
        "create_strategy": (".strategy_factory", "create_strategy"),
        "LiveScheduler": (".live_scheduler", "LiveScheduler"),
        "TradingMode": (".live_scheduler", "TradingMode"),
        "TqSdkLiveFeed": (".live_feed", "TqSdkLiveFeed"),
        "UnifiedLiveFeed": (".live_feed", "UnifiedLiveFeed"),
    }
    if name in _lazy:
        import importlib
        mod_path, attr = _lazy[name]
        mod = importlib.import_module(mod_path, __package__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
