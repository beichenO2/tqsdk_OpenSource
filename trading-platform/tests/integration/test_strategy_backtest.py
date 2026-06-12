"""策略 → 回测引擎端到端集成测试。"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from backtest import BacktestConfig, BacktestEngine, Bar, BarDataFeed, StrategyAdapter
from strategy.base import StrategyConfig
from strategy.futures.dual_ma import FuturesDualMAStrategy
from strategy.futures.cta_trend import CTATrendStrategy
from strategy.futures.rbreaker import RBreakerStrategy
from strategy.registry import StrategyRegistry


def _make_trending_bars(
    symbol: str,
    start: datetime,
    count: int = 200,
    base_price: float = 3500.0,
) -> list[Bar]:
    """生成有明确趋势的模拟 K 线。"""
    bars: list[Bar] = []
    price = base_price
    dt = start
    for i in range(count):
        if i < count // 3:
            price += 5
        elif i < 2 * count // 3:
            price -= 8
        else:
            price += 6
        bars.append(Bar(
            symbol=symbol,
            dt=dt,
            open=Decimal(str(round(price - 2, 1))),
            high=Decimal(str(round(price + 6, 1))),
            low=Decimal(str(round(price - 6, 1))),
            close=Decimal(str(round(price, 1))),
            volume=2000 + i * 10,
        ))
        dt += timedelta(minutes=1)
    return bars


def _make_multi_day_bars(
    symbol: str,
    start: datetime,
    days: int = 5,
    bars_per_day: int = 120,
    base_price: float = 3500.0,
) -> list[Bar]:
    """生成跨多个交易日的 K 线，日间有明显趋势切换。

    Day 1: 震荡（建立前日 OHLC 基准）
    Day 2: 强势上涨（突破上轨）
    Day 3: 高位反转下跌（突破下轨）
    Day 4+: 继续波动
    """
    bars: list[Bar] = []
    price = base_price
    dt = start

    for day in range(days):
        day_start = start + timedelta(days=day, hours=9)
        dt = day_start

        for i in range(bars_per_day):
            if day == 0:
                price += 0.5 * (1 if i % 7 < 4 else -1)
            elif day == 1:
                price += 3.0
            elif day == 2:
                price -= 5.0
            elif day == 3:
                price += 4.0
            else:
                price -= 2.5

            spread = max(abs(price) * 0.003, 3.0)
            bars.append(Bar(
                symbol=symbol,
                dt=dt,
                open=Decimal(str(round(price - spread * 0.3, 2))),
                high=Decimal(str(round(price + spread, 2))),
                low=Decimal(str(round(price - spread, 2))),
                close=Decimal(str(round(price, 2))),
                volume=1500 + day * 200 + i * 5,
            ))
            dt += timedelta(minutes=1)

    return bars


@pytest.fixture
def backtest_config() -> BacktestConfig:
    return BacktestConfig(
        strategy_id="test",
        symbols=["SHFE.rb2501"],
        start_date=datetime(2026, 1, 1),
        end_date=datetime(2026, 12, 31),
        initial_capital=Decimal("1000000"),
        commission_rate=Decimal("0.0001"),
        slippage_ticks=1,
        tick_size=Decimal("1"),
        contract_multiplier=10,
    )


def test_dual_ma_backtest_runs_and_produces_trades(backtest_config: BacktestConfig) -> None:
    engine = BacktestEngine(backtest_config)
    feed = BarDataFeed(engine.event_bus)
    feed.add_bars(_make_trending_bars("SHFE.rb2501", datetime(2026, 3, 1), count=300))
    engine.set_datafeed(feed)

    config = StrategyConfig(
        name="dual_ma_test",
        symbols=["SHFE.rb2501"],
        params={"fast_period": 3, "slow_period": 8, "volume_ma_period": 8, "volume_surge_ratio": 0.5},
    )
    inner = FuturesDualMAStrategy(config)
    adapter = StrategyAdapter(inner, default_volume=1)
    engine.set_strategy(adapter)

    result = engine.run()

    assert result.total_trades > 0, "应产生至少 1 笔交易"
    assert len(result.equity_curve) > 0, "应有权益曲线"
    assert result.final_equity != backtest_config.initial_capital, "权益应发生变化"


def test_cta_trend_backtest_produces_signals(backtest_config: BacktestConfig) -> None:
    engine = BacktestEngine(backtest_config)
    feed = BarDataFeed(engine.event_bus)
    feed.add_bars(_make_trending_bars("SHFE.rb2501", datetime(2026, 3, 1), count=200))
    engine.set_datafeed(feed)

    config = StrategyConfig(
        name="cta_test",
        symbols=["SHFE.rb2501"],
        params={"entry_period": 8, "exit_period": 4, "atr_period": 6},
    )
    inner = CTATrendStrategy(config)
    adapter = StrategyAdapter(inner, default_volume=1)
    engine.set_strategy(adapter)

    result = engine.run()

    assert result.total_trades > 0, "CTA 趋势策略应产生交易"


def test_strategy_registry_contains_futures_strategies() -> None:
    registered = StrategyRegistry.list_registered()
    assert "futures_dual_ma" in registered
    assert "cta_trend" in registered
    assert "rbreaker" in registered


def test_strategy_registry_btc_after_import() -> None:
    import strategy.btc  # noqa: F401 — triggers auto_register
    registered = StrategyRegistry.list_registered()
    assert "btc_momentum" in registered
    assert "btc_grid" in registered


def test_strategy_registry_create_and_instantiate() -> None:
    config = StrategyConfig(name="test_create", symbols=["X"])
    strat = StrategyRegistry.create("futures_dual_ma", config)
    assert strat.name == "test_create"
    assert hasattr(strat, "on_bar")


def test_backtest_equity_curve_monotonic_at_start(backtest_config: BacktestConfig) -> None:
    """初始阶段（策略未触发前）权益应保持不变。"""
    engine = BacktestEngine(backtest_config)
    feed = BarDataFeed(engine.event_bus)
    feed.add_bars(_make_trending_bars("SHFE.rb2501", datetime(2026, 3, 1), count=50))
    engine.set_datafeed(feed)

    config = StrategyConfig(
        name="slow_strat",
        symbols=["SHFE.rb2501"],
        params={"fast_period": 20, "slow_period": 40, "volume_ma_period": 40},
    )
    inner = FuturesDualMAStrategy(config)
    adapter = StrategyAdapter(inner, default_volume=1)
    engine.set_strategy(adapter)

    result = engine.run()

    if result.total_trades == 0:
        assert result.final_equity == backtest_config.initial_capital


# ---------- T-06: CTA 趋势策略增强测试 ----------


def test_cta_trend_backtest_full_metrics(backtest_config: BacktestConfig) -> None:
    """CTA 趋势策略端到端：验证交易数、权益曲线、回测报告字段完整。"""
    engine = BacktestEngine(backtest_config)
    feed = BarDataFeed(engine.event_bus)
    feed.add_bars(_make_trending_bars("SHFE.rb2501", datetime(2026, 3, 1), count=400))
    engine.set_datafeed(feed)

    config = StrategyConfig(
        name="cta_full_test",
        symbols=["SHFE.rb2501"],
        params={"entry_period": 10, "exit_period": 5, "atr_period": 8},
    )
    inner = CTATrendStrategy(config)
    adapter = StrategyAdapter(inner, default_volume=1)
    engine.set_strategy(adapter)

    result = engine.run()

    assert result.total_trades >= 2, f"400 根趋势 K 线应至少 2 笔交易，实际 {result.total_trades}"
    assert len(result.equity_curve) > 0, "应有权益曲线"
    assert result.final_equity != backtest_config.initial_capital, "权益应有变化"
    assert result.max_drawdown_pct >= Decimal(0), "最大回撤应 >= 0"


# ---------- T-06: R-Breaker 策略回测测试 ----------


def test_rbreaker_backtest_produces_trades(backtest_config: BacktestConfig) -> None:
    """R-Breaker 策略端到端回测：跨多日数据应触发趋势/反转信号并产生交易。"""
    engine = BacktestEngine(backtest_config)
    feed = BarDataFeed(engine.event_bus)
    feed.add_bars(_make_multi_day_bars(
        "SHFE.rb2501",
        datetime(2026, 3, 1),
        days=5,
        bars_per_day=120,
        base_price=3500.0,
    ))
    engine.set_datafeed(feed)

    config = StrategyConfig(
        name="rbreaker_test",
        symbols=["SHFE.rb2501"],
        params={
            "f1": 0.35,
            "f2": 0.07,
            "f3": 0.25,
            "position_size": 1.0,
            "atr_period": 8,
            "trailing_stop_atr_mult": 2.0,
        },
    )
    inner = RBreakerStrategy(config)
    adapter = StrategyAdapter(inner, default_volume=1)
    engine.set_strategy(adapter)

    result = engine.run()

    assert result.total_trades > 0, "R-Breaker 策略在多日趋势数据下应产生交易"
    assert len(result.equity_curve) > 0, "应有权益曲线"


def test_rbreaker_backtest_equity_changes(backtest_config: BacktestConfig) -> None:
    """R-Breaker 在高波动数据下权益应发生变化。"""
    engine = BacktestEngine(backtest_config)
    feed = BarDataFeed(engine.event_bus)
    feed.add_bars(_make_multi_day_bars(
        "SHFE.rb2501",
        datetime(2026, 3, 1),
        days=6,
        bars_per_day=150,
        base_price=3500.0,
    ))
    engine.set_datafeed(feed)

    config = StrategyConfig(
        name="rbreaker_equity_test",
        symbols=["SHFE.rb2501"],
        params={
            "f1": 0.35,
            "f2": 0.07,
            "f3": 0.25,
            "position_size": 2.0,
            "atr_period": 6,
            "trailing_stop_atr_mult": 1.5,
        },
    )
    inner = RBreakerStrategy(config)
    adapter = StrategyAdapter(inner, default_volume=2)
    engine.set_strategy(adapter)

    result = engine.run()

    assert result.total_trades > 0, "R-Breaker 应有交易"
    assert result.final_equity != backtest_config.initial_capital, "权益应有变化"


# ---------- GridOptimizer + 双均线参数网格 ----------


def test_optimizer_finds_best_params(backtest_config: BacktestConfig) -> None:
    """GridOptimizer 在合成 K 线上遍历参数网格，应得到 best_params 与全部组合结果。"""
    try:
        from src.optimizer import GridOptimizer
    except ImportError:
        pytest.skip("src.optimizer.GridOptimizer 尚未提供")

    symbol = "SHFE.rb2501"
    bars = _make_trending_bars(symbol, datetime(2026, 3, 1), count=200)
    param_grid = {"fast_period": [3, 5], "slow_period": [10, 15]}

    def evaluate(params: dict) -> float:
        engine = BacktestEngine(backtest_config)
        feed = BarDataFeed(engine.event_bus)
        feed.add_bars(bars)
        engine.set_datafeed(feed)
        config = StrategyConfig(
            name="grid_dual_ma",
            symbols=[symbol],
            params={
                "fast_period": params["fast_period"],
                "slow_period": params["slow_period"],
                "volume_ma_period": 8,
                "volume_surge_ratio": 0.5,
            },
        )
        inner = FuturesDualMAStrategy(config)
        adapter = StrategyAdapter(inner, default_volume=1)
        engine.set_strategy(adapter)
        result = engine.run()
        return float(result.final_equity)

    optimizer = GridOptimizer(param_grid)
    out = optimizer.run(evaluate, maximize=True)

    assert out.best_params, "应返回最优参数字典"
    assert set(out.best_params.keys()) == {"fast_period", "slow_period"}
    assert len(out.results) == 4, "2×2 网格应评估 4 组参数"
    assert {tuple(r["params"].items()) for r in out.results} == {
        tuple(sorted({"fast_period": fp, "slow_period": sp}.items()))
        for fp in param_grid["fast_period"]
        for sp in param_grid["slow_period"]
    }
