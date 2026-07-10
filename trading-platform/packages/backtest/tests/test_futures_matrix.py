"""Futures strategy×symbol matrix must route through BacktestEngine."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from backtest.futures_matrix import backtest_strategy_on_bars, run_futures_matrix
from strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig
from strategy.registry import StrategyRegistry


class _AlternatingStrategy(BaseStrategy):
    """Every 4 bars open long, exit 2 bars later — forces trades on any path."""

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self._tick = 0
        self._in_pos = False

    async def on_bar(self, symbol, bar):
        self._tick += 1
        out: list[Signal] = []
        if not self._in_pos and self._tick % 4 == 0:
            out.append(
                Signal(
                    symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strategy_id=self.config.strategy_id,
                    strength=1.0,
                )
            )
            self._in_pos = True
        elif self._in_pos and self._tick % 4 == 2:
            out.append(
                Signal(
                    symbol=symbol,
                    signal_type=SignalType.LONG_EXIT,
                    strategy_id=self.config.strategy_id,
                    strength=1.0,
                )
            )
            self._in_pos = False
        return out

    async def generate_signals(self, market_data):
        return []


def _make_uptrend_bars(n: int = 120, start: float = 3000.0, step: float = 5.0) -> pd.DataFrame:
    prices = start + step * np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-02 09:00", periods=n, freq="5min"),
            "open": prices,
            "high": prices + 2.0,
            "low": prices - 1.0,
            "close": prices + 1.0,
            "volume": np.full(n, 5000.0),
            "open_interest": np.full(n, 100_000.0),
            "instrument": ["rb2505"] * n,
        }
    )


def _register_alternating(registry: StrategyRegistry) -> None:
    registry.register("matrix_alt", _AlternatingStrategy)


def test_futures_matrix_uses_backtest_engine() -> None:
    bars = _make_uptrend_bars(n=80)
    registry = StrategyRegistry()
    _register_alternating(registry)

    results = run_futures_matrix(
        {"rb": bars},
        ["matrix_alt"],
        registry,
        initial_capital=Decimal("100000"),
    )

    assert len(results) == 1
    row = results[0]
    assert row["strategy"] == "matrix_alt"
    assert row["symbol"] == "rb"
    assert "error" not in row

    for key in (
        "sharpe_ratio",
        "max_drawdown",
        "max_drawdown_pct",
        "total_trades",
        "total_return",
        "win_rate",
        "profit_factor",
        "final_equity",
    ):
        assert key in row, f"missing BacktestResult field: {key}"

    assert row["total_trades"] > 0
    assert row["sharpe"] == pytest.approx(row["sharpe_ratio"], rel=1e-9)
    assert row["trades"] == row["total_trades"]
    assert row["max_dd"] == pytest.approx(row["max_drawdown_pct"], rel=1e-9)


def test_synthetic_trend_strategy_profits() -> None:
    import strategy.futures  # noqa: F401 — trigger @auto_register

    bars = _make_uptrend_bars(n=200, step=8.0)
    registry = StrategyRegistry()

    config = StrategyConfig(
        name="kalman_trend",
        strategy_id="kalman_trend",
    )
    strategy = registry.create("kalman_trend", config)
    assert strategy is not None

    result = backtest_strategy_on_bars(
        strategy,
        bars,
        symbol="rb2505",
        initial_capital=Decimal("1000000"),
        contract_multiplier=10,
        default_volume=1,
    )

    assert result["total_trades"] > 0, "matching engine should have filled orders"
    assert float(result["total_return"]) > 0, "uptrend + trend strategy should be profitable"
