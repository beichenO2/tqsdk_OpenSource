"""Futures strategy×symbol matrix backtest — unified BacktestEngine path."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pandas as pd

from .datafeed import BarDataFeed
from .engine import BacktestEngine
from .futures_datafeed import dataframe_to_bars
from .models import BacktestConfig, BacktestResult
from .strategy_adapter import StrategyAdapter

if TYPE_CHECKING:
    from strategy.base import BaseStrategy
    from strategy.registry import StrategyRegistry

logger = logging.getLogger(__name__)

DEFAULT_INITIAL_CAPITAL = Decimal("100000")
DEFAULT_COMMISSION_RATE = Decimal("0.00005")
DEFAULT_SLIPPAGE_TICKS = 1
DEFAULT_TICK_SIZE = Decimal("1")
DEFAULT_CONTRACT_MULTIPLIER = 10
DEFAULT_VOLUME = 1


def _infer_symbol(bars: pd.DataFrame, symbol: str | None) -> str:
    if symbol:
        return symbol
    if "instrument" in bars.columns and len(bars) > 0:
        return str(bars.iloc[0]["instrument"])
    return "unknown"


def _date_bounds(bars: pd.DataFrame) -> tuple[datetime, datetime]:
    series = pd.to_datetime(bars["datetime"])
    start = series.iloc[0].to_pydatetime()
    end = series.iloc[-1].to_pydatetime()
    if hasattr(start, "tzinfo") and start.tzinfo is not None:
        start = start.replace(tzinfo=None)
    if hasattr(end, "tzinfo") and end.tzinfo is not None:
        end = end.replace(tzinfo=None)
    return start, end


def result_to_report_dict(
    result: BacktestResult,
    *,
    strategy: str = "",
    symbol: str = "",
    bars: int = 0,
    duration_s: float = 0.0,
) -> dict[str, Any]:
    """Map ``BacktestResult`` to JSON report fields (legacy + formal metrics)."""
    total_return = float(result.total_return)
    sharpe = float(result.sharpe_ratio)
    max_dd_pct = float(result.max_drawdown_pct)
    return {
        "strategy": strategy,
        "symbol": symbol,
        "bars": bars,
        "duration_s": round(duration_s, 1),
        "total_return": round(total_return, 6),
        "sharpe": round(sharpe, 4),
        "max_dd": round(max_dd_pct, 6),
        "trades": result.total_trades,
        "win_rate": round(float(result.win_rate), 4),
        "profit_factor": round(float(result.profit_factor), 4),
        "final_capital": round(float(result.final_equity), 2),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(float(result.max_drawdown), 6),
        "max_drawdown_pct": round(max_dd_pct, 6),
        "total_trades": result.total_trades,
        "sortino_ratio": round(float(result.sortino_ratio), 4),
        "annual_return": round(float(result.annual_return), 6),
        "calmar_ratio": round(float(result.calmar_ratio), 4),
        "avg_trade_pnl": round(float(result.avg_trade_pnl), 2),
        "avg_holding_period": round(float(result.avg_holding_period), 2),
        "final_equity": round(float(result.final_equity), 2),
    }


def backtest_strategy_on_bars(
    strategy: BaseStrategy,
    bars: pd.DataFrame,
    *,
    symbol: str | None = None,
    initial_capital: Decimal = DEFAULT_INITIAL_CAPITAL,
    commission_rate: Decimal = DEFAULT_COMMISSION_RATE,
    slippage_ticks: int = DEFAULT_SLIPPAGE_TICKS,
    tick_size: Decimal = DEFAULT_TICK_SIZE,
    contract_multiplier: int = DEFAULT_CONTRACT_MULTIPLIER,
    default_volume: int = DEFAULT_VOLUME,
) -> dict[str, Any]:
    """Run one BaseStrategy on a bar DataFrame via BacktestEngine."""
    if bars.empty:
        raise ValueError("bars DataFrame is empty")

    bar_symbol = _infer_symbol(bars, symbol)
    start_date, end_date = _date_bounds(bars)
    engine_bars = dataframe_to_bars(bars, symbol=bar_symbol)

    config = BacktestConfig(
        strategy_id=strategy.config.strategy_id,
        symbols=[bar_symbol],
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        slippage_ticks=slippage_ticks,
        tick_size=tick_size,
        contract_multiplier=contract_multiplier,
    )

    engine = BacktestEngine(config)
    feed = BarDataFeed(engine.event_bus)
    feed.add_bars(engine_bars)
    engine.set_datafeed(feed)
    engine.set_strategy(StrategyAdapter(strategy, default_volume=default_volume))

    t0 = time.monotonic()
    result = engine.run()
    duration_s = time.monotonic() - t0

    return result_to_report_dict(
        result,
        strategy=strategy.config.strategy_id,
        symbol=bar_symbol if symbol is None else symbol,
        bars=len(bars),
        duration_s=duration_s,
    )


def run_futures_matrix(
    symbol_bars: dict[str, pd.DataFrame],
    strategy_names: list[str],
    registry: StrategyRegistry,
    *,
    initial_capital: Decimal = DEFAULT_INITIAL_CAPITAL,
    commission_rate: Decimal = DEFAULT_COMMISSION_RATE,
    slippage_ticks: int = DEFAULT_SLIPPAGE_TICKS,
    tick_size: Decimal = DEFAULT_TICK_SIZE,
    contract_multiplier: int = DEFAULT_CONTRACT_MULTIPLIER,
    default_volume: int = DEFAULT_VOLUME,
) -> list[dict[str, Any]]:
    """Run strategy×symbol matrix; each cell uses BacktestEngine."""
    from strategy.base import StrategyConfig

    results: list[dict[str, Any]] = []

    for sym, bars in symbol_bars.items():
        if bars.empty:
            logger.warning("No data for %s, skipping", sym)
            continue

        for strat_name in strategy_names:
            t0 = time.monotonic()
            try:
                strategy_cls = registry.get(strat_name)
                if strategy_cls is None:
                    logger.warning("Strategy %s not found in registry", strat_name)
                    continue

                config = StrategyConfig(name=strat_name, strategy_id=strat_name)
                strategy_instance = strategy_cls(config)
                row = backtest_strategy_on_bars(
                    strategy_instance,
                    bars,
                    symbol=sym,
                    initial_capital=initial_capital,
                    commission_rate=commission_rate,
                    slippage_ticks=slippage_ticks,
                    tick_size=tick_size,
                    contract_multiplier=contract_multiplier,
                    default_volume=default_volume,
                )
                row["strategy"] = strat_name
                row["symbol"] = sym
                row["duration_s"] = round(time.monotonic() - t0, 1)
                results.append(row)
            except Exception as exc:
                logger.error("%s/%s FAILED: %s", strat_name, sym, exc)
                results.append(
                    {
                        "strategy": strat_name,
                        "symbol": sym,
                        "error": str(exc),
                        "duration_s": round(time.monotonic() - t0, 1),
                    }
                )

    return results
