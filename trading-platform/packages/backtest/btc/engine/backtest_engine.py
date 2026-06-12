"""BTC backtesting engine — orchestrates strategy, exchange, and data replay.

Usage:
    engine = BTCBacktestEngine(config)
    engine.load_data(replayer)
    engine.set_strategy(my_strategy)
    result = engine.run()
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from decimal import Decimal

from strategy.base import BaseStrategy

from ..exchange.simulated import SimulatedExchange
from ..models.types import BacktestConfig, BacktestResult, OHLCV
from ..replayer.data_replayer import DataReplayer
from ..report.analyzer import BacktestAnalyzer
from ..strategy_bridge import (
    ohlcv_to_bar_dict,
    run_async,
    signals_to_btc_orders,
    sync_bt_positions_to_strategy,
)

logger = logging.getLogger(__name__)


class BTCBacktestEngine:
    """Run a BTC strategy against historical data through a simulated exchange."""

    def __init__(self, config: BacktestConfig) -> None:
        self._config = config
        self._exchange = SimulatedExchange(
            initial_capital=config.initial_capital,
            commission_rate=config.commission_rate,
            slippage_bps=config.slippage_bps,
            leverage_limit=config.leverage_limit,
            max_position_pct=config.max_position_pct,
        )
        self._replayer: DataReplayer | None = None
        self._strategy: BaseStrategy | None = None
        self._funding_rates: dict[datetime, Decimal] = {}
        self._bar_count = 0
        self._fill_count = 0

    @property
    def config(self) -> BacktestConfig:
        return self._config

    def load_data(self, replayer: DataReplayer) -> None:
        """Attach a data replayer as the bar source."""
        self._replayer = replayer
        logger.info(
            "Data loaded: %d bars for %s",
            replayer.bar_count,
            replayer.symbol,
        )

    def set_strategy(self, strategy: BaseStrategy) -> None:
        """Attach a strategy instance."""
        self._strategy = strategy

    def set_funding_rates(self, rates: dict[datetime, Decimal]) -> None:
        """Pre-load historical funding rates keyed by timestamp."""
        self._funding_rates = rates

    def run(self) -> BacktestResult:
        """Execute the backtest synchronously. Returns full result with metrics."""
        if self._replayer is None:
            raise RuntimeError("No data replayer attached. Call load_data() first.")
        if self._strategy is None:
            raise RuntimeError("No strategy attached. Call set_strategy() first.")

        started_at = datetime.now(UTC)
        wall_start = time.monotonic()

        run_async(self._strategy.on_start())

        logger.info(
            "Backtest starting: strategy=%s, symbols=%s, period=%s to %s",
            self._config.strategy_id,
            self._config.symbols,
            self._config.start_date,
            self._config.end_date,
        )

        for bar in self._replayer.replay(
            start=self._config.start_date,
            end=self._config.end_date,
        ):
            self._process_bar(bar)

        finished_at = datetime.now(UTC)
        wall_elapsed = time.monotonic() - wall_start

        logger.info(
            "Backtest completed: %d bars, %d fills, %.2fs wall time",
            self._bar_count,
            self._fill_count,
            wall_elapsed,
        )

        analyzer = BacktestAnalyzer(
            config=self._config,
            equity_curve=self._exchange._equity_history,
            fills=self._exchange.fills,
            started_at=started_at,
            finished_at=finished_at,
        )

        result = analyzer.compute()
        self._strategy.on_backtest_complete(result)
        run_async(self._strategy.on_stop())
        return result

    def _process_bar(self, bar: OHLCV) -> None:
        """Process a single bar through the simulation pipeline."""
        self._bar_count += 1

        fills = self._exchange.on_bar(bar)
        self._fill_count += len(fills)

        for fill in fills:
            self._strategy.on_fill(fill)

        if self._config.funding_rate_enabled and bar.timestamp in self._funding_rates:
            for symbol in self._config.symbols:
                rate = self._funding_rates[bar.timestamp]
                self._exchange.apply_funding_rate(symbol, rate)

        assert self._strategy is not None
        assert self._replayer is not None
        sync_bt_positions_to_strategy(self._strategy, self._exchange.positions)
        symbol = self._replayer.symbol
        bar_dict = ohlcv_to_bar_dict(bar)
        signals = run_async(self._strategy.on_bar(symbol, bar_dict))
        max_orders = int(self._config.metadata.get("max_open_orders", 5))
        orders = signals_to_btc_orders(
            signals,
            self._exchange.positions,
            float(self._exchange.equity),
            float(bar.close),
            position_size_pct=float(self._config.max_position_pct),
            max_open_orders=max_orders,
        )

        for order in orders:
            self._exchange.submit_order(order)

        if self._bar_count % 10000 == 0:
            logger.debug(
                "Progress: %d bars processed, equity=%.2f",
                self._bar_count,
                self._exchange.equity,
            )
