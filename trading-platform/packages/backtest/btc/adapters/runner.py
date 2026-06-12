"""High-level BTC backtest runner — one-call orchestration.

Usage:
    from btc.adapters import BTCBacktestRunner

    result = BTCBacktestRunner.run_momentum(
        data_dir="./data/btc",
        symbol="BTCUSDT",
        start="2025-01-01",
        end="2025-12-31",
    )
    print(result.sharpe_ratio, result.max_drawdown_pct)
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from backtest.engine import BacktestEngine
from backtest.models import BacktestConfig, BacktestResult

from .datafeed import BTCDataFeed
from .strategy_adapter import BTCStrategyAdapter

logger = logging.getLogger(__name__)

BTC_DEFAULTS = {
    "initial_capital": Decimal("100000"),
    "commission_rate": Decimal("0.001"),
    "slippage_ticks": 1,
    "tick_size": Decimal("0.01"),
    "contract_multiplier": 1,
}


class BTCBacktestRunner:
    """Convenience class for running BTC backtests with minimal boilerplate."""

    @staticmethod
    def run(
        strategy_name: str,
        symbol: str = "BTCUSDT",
        start: str | datetime = "2025-01-01",
        end: str | datetime = "2025-12-31",
        data_source: str | Path | None = None,
        data_format: str = "csv",
        initial_capital: Decimal = BTC_DEFAULTS["initial_capital"],
        commission_rate: Decimal = BTC_DEFAULTS["commission_rate"],
        strategy_params: dict | None = None,
        qty_pct: float = 0.02,
    ) -> BacktestResult:
        """Run a full BTC backtest using Ch29 engine + Ch33 strategy.

        Args:
            strategy_name: registered strategy name (e.g. "btc_momentum")
            symbol: trading pair
            start/end: backtest period
            data_source: path to CSV/Parquet data, or Parquet storage dir
            data_format: "csv", "parquet"
            initial_capital: starting equity
            commission_rate: per-trade commission
            strategy_params: override strategy parameters
            qty_pct: fraction of equity per signal
        """
        start_dt = _parse_dt(start)
        end_dt = _parse_dt(end)

        config = BacktestConfig(
            strategy_id=strategy_name,
            symbols=[symbol],
            start_date=start_dt,
            end_date=end_dt,
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            slippage_ticks=BTC_DEFAULTS["slippage_ticks"],
            tick_size=BTC_DEFAULTS["tick_size"],
            contract_multiplier=BTC_DEFAULTS["contract_multiplier"],
        )

        engine = BacktestEngine(config)

        datafeed = BTCDataFeed(engine.event_bus)
        if data_source:
            if data_format == "csv":
                datafeed.load_from_csv(data_source, symbol)
            elif data_format == "parquet":
                datafeed.load_from_parquet(str(data_source), "binance", symbol)
            else:
                raise ValueError(f"Unsupported data_format: {data_format}")
        engine.set_datafeed(datafeed)

        from strategy.base import StrategyConfig
        from strategy.registry import StrategyRegistry

        strat_config = StrategyConfig(
            name=strategy_name,
            symbols=[symbol],
            params=strategy_params or {},
        )
        inner_strategy = StrategyRegistry.create(strategy_name, strat_config)
        adapter = BTCStrategyAdapter(inner_strategy, qty_pct=qty_pct)
        engine.set_strategy(adapter)

        logger.info(
            "Running BTC backtest: strategy=%s symbol=%s period=%s~%s capital=%s",
            strategy_name, symbol, start_dt, end_dt, initial_capital,
        )

        result = engine.run()

        logger.info(
            "Backtest complete: return=%.2f%% sharpe=%.3f maxdd=%.2f%% trades=%d",
            float(result.total_return * 100),
            float(result.sharpe_ratio),
            float(result.max_drawdown_pct * 100),
            result.total_trades,
        )
        return result

    @staticmethod
    def run_momentum(
        data_source: str | Path,
        symbol: str = "BTCUSDT",
        start: str = "2025-01-01",
        end: str = "2025-12-31",
        **kwargs,
    ) -> BacktestResult:
        """Shortcut to run a BTC momentum backtest."""
        return BTCBacktestRunner.run(
            strategy_name="btc_momentum",
            symbol=symbol,
            start=start,
            end=end,
            data_source=data_source,
            **kwargs,
        )

    @staticmethod
    def run_mean_reversion(
        data_source: str | Path,
        symbol: str = "BTCUSDT",
        start: str = "2025-01-01",
        end: str = "2025-12-31",
        **kwargs,
    ) -> BacktestResult:
        """Shortcut to run a BTC mean reversion backtest."""
        return BTCBacktestRunner.run(
            strategy_name="btc_mean_reversion",
            symbol=symbol,
            start=start,
            end=end,
            data_source=data_source,
            **kwargs,
        )

    @staticmethod
    def run_grid(
        data_source: str | Path,
        symbol: str = "BTCUSDT",
        start: str = "2025-01-01",
        end: str = "2025-12-31",
        **kwargs,
    ) -> BacktestResult:
        """Shortcut to run a BTC grid trading backtest."""
        return BTCBacktestRunner.run(
            strategy_name="btc_grid",
            symbol=symbol,
            start=start,
            end=end,
            data_source=data_source,
            **kwargs,
        )


def _parse_dt(v: str | datetime) -> datetime:
    if isinstance(v, datetime):
        return v
    return datetime.fromisoformat(v)
