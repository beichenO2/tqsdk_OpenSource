"""回测引擎包 - 提供事件驱动的回测框架。"""

from .models import (
    Bar,
    Tick,
    Order,
    Trade,
    Position,
    BacktestConfig,
    BacktestResult,
    OrderSide,
    OrderType,
    OrderStatus,
)
from .engine import BacktestEngine
from .events import EventBus, Event, EventType
from .matching import MatchingEngine
from .datafeed import DataFeed, BarDataFeed
from .report import ReportGenerator
from .strategy import Strategy
from .datafeed_datahub import DataHubFeed
from .sim_broker import SimBrokerAdapter
from .persistence import BacktestPersistence
from .strategy_adapter import StrategyAdapter
try:
    from .optimizer import GridOptimizer, OptimizationResult
except ModuleNotFoundError:  # pragma: no cover - optional strategy package not on sys.path
    GridOptimizer = None  # type: ignore[assignment]
    OptimizationResult = None  # type: ignore[assignment]

__all__ = [
    "Bar",
    "Tick",
    "Order",
    "Trade",
    "Position",
    "BacktestConfig",
    "BacktestResult",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "BacktestEngine",
    "EventBus",
    "Event",
    "EventType",
    "MatchingEngine",
    "DataFeed",
    "BarDataFeed",
    "ReportGenerator",
    "Strategy",
    "DataHubFeed",
    "SimBrokerAdapter",
    "BacktestPersistence",
    "StrategyAdapter",
    "GridOptimizer",
    "OptimizationResult",
]
