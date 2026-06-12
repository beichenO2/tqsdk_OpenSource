"""BTC Backtesting & Simulation package.

Public API — Two usage modes:

1. Standalone mode (btc-native engine):
    BTCBacktestEngine, SimulatedExchange, DataReplayer, BacktestAnalyzer

2. Integrated mode (Ch29 engine + Ch32 data + Ch33 strategies):
    BTCBacktestRunner, BTCDataFeed, BTCStrategyAdapter, SimulatedExchangeAdapter
"""

from .engine import BTCBacktestEngine
from .exchange import SimulatedExchange
from .replayer import DataReplayer
from .report import BacktestAnalyzer
from .models import (
    OHLCV,
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    Fill,
    Position,
    BacktestConfig,
    BacktestResult,
    PerformanceMetrics,
)

__all__ = [
    # Standalone mode
    "BTCBacktestEngine",
    "SimulatedExchange",
    "DataReplayer",
    "BacktestAnalyzer",
    # Models
    "OHLCV",
    "Order",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "Fill",
    "Position",
    "BacktestConfig",
    "BacktestResult",
    "PerformanceMetrics",
]
