"""Integration adapters connecting BTC backtest with upstream components.

- BTCDataFeed: Ch29 DataFeed ← Ch32 data pipeline
- BTCStrategyAdapter: Ch29 Strategy ← Ch33 async signal-based strategies
- SimulatedExchangeAdapter: Ch31 ExchangeAdapter for paper trading
- BTCBacktestRunner: high-level orchestration
"""

from .datafeed import BTCDataFeed
from .strategy_adapter import BTCStrategyAdapter
from .exchange_adapter import SimulatedExchangeAdapter
from .runner import BTCBacktestRunner

__all__ = [
    "BTCDataFeed",
    "BTCStrategyAdapter",
    "SimulatedExchangeAdapter",
    "BTCBacktestRunner",
]
