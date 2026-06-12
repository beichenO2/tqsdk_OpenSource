"""BTC 策略 → BTC 回测引擎端到端集成测试。

验证 strategy/btc/ 下的策略可通过 backtest/btc/engine/ 完整运行。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from btc import (
    BTCBacktestEngine,
    DataReplayer,
    OHLCV,
    BacktestConfig as BTCBacktestConfig,
)
from strategy.base import StrategyConfig
from strategy.btc.momentum import BTCMomentumStrategy
from strategy.btc.trend_following import BTCTrendFollowingStrategy
from strategy.btc.backtest_adapter import BacktestStrategyAdapter


def _make_btc_bars(
    start: datetime,
    count: int = 500,
    base_price: float = 65000.0,
) -> list[OHLCV]:
    """生成有趋势切换的 BTC 模拟 K 线。"""
    bars: list[OHLCV] = []
    price = base_price
    dt = start

    for i in range(count):
        phase = i / count
        if phase < 0.3:
            price += 80
        elif phase < 0.55:
            price -= 130
        elif phase < 0.8:
            price += 100
        else:
            price -= 60

        spread = max(price * 0.002, 20.0)
        bars.append(OHLCV(
            timestamp=dt,
            open=Decimal(str(round(price - spread * 0.3, 2))),
            high=Decimal(str(round(price + spread, 2))),
            low=Decimal(str(round(price - spread, 2))),
            close=Decimal(str(round(price, 2))),
            volume=Decimal(str(round(50 + i * 0.5, 2))),
        ))
        dt += timedelta(minutes=1)

    return bars


@pytest.fixture
def btc_backtest_config() -> BTCBacktestConfig:
    return BTCBacktestConfig(
        strategy_id="btc_test",
        symbols=["BTCUSDT"],
        start_date=datetime(2026, 1, 1),
        end_date=datetime(2026, 12, 31),
        initial_capital=Decimal("100000"),
        commission_rate=Decimal("0.001"),
        slippage_bps=Decimal("5"),
        leverage_limit=Decimal("1"),
        max_position_pct=Decimal("0.1"),
        funding_rate_enabled=False,
    )


def _build_engine_with_data(
    config: BTCBacktestConfig,
    bar_count: int = 500,
) -> BTCBacktestEngine:
    engine = BTCBacktestEngine(config)
    replayer = DataReplayer(symbol="BTCUSDT")
    replayer.load_from_list(_make_btc_bars(datetime(2026, 3, 1), count=bar_count))
    engine.load_data(replayer)
    return engine


def test_btc_momentum_backtest_runs(btc_backtest_config: BTCBacktestConfig) -> None:
    """BTCMomentumStrategy 通过 BTCBacktestEngine 端到端运行。"""
    engine = _build_engine_with_data(btc_backtest_config, bar_count=500)

    config = StrategyConfig(
        name="momentum_bt",
        symbols=["BTCUSDT"],
        params={
            "fast_period": 5,
            "slow_period": 15,
            "volume_ma_period": 10,
            "momentum_threshold": 0.01,
            "volume_surge_ratio": 0.8,
            "atr_period": 8,
            "trailing_stop_atr_mult": 2.0,
        },
    )
    strategy = BTCMomentumStrategy(config)
    engine.set_strategy(strategy)

    result = engine.run()

    assert result.metrics.total_trades > 0, "BTC 动量策略应产生交易"
    assert len(result.equity_curve) > 0, "应有权益曲线"
    assert result.metrics.total_return != Decimal(0), "收益率不应为零"


def test_btc_trend_following_backtest_runs(btc_backtest_config: BTCBacktestConfig) -> None:
    """BTCTrendFollowingStrategy 通过 BTCBacktestEngine 端到端运行。"""
    engine = _build_engine_with_data(btc_backtest_config, bar_count=1200)

    config = StrategyConfig(
        name="trend_bt",
        symbols=["BTCUSDT"],
        params={
            "ema_fast": 5,
            "ema_slow": 12,
            "ema_trend": 20,
            "adx_period": 8,
            "adx_threshold": 10.0,
            "atr_period": 6,
            "risk_per_trade_pct": 0.02,
            "trailing_stop_atr_mult": 1.5,
            "partial_take_profit_atr_mult": 2.5,
            "partial_close_pct": 0.5,
        },
    )
    strategy = BTCTrendFollowingStrategy(config)
    engine.set_strategy(strategy)

    result = engine.run()

    assert result.metrics.total_trades > 0, "BTC 趋势跟随策略应产生交易"
    assert len(result.fills) > 0, "应有成交记录"


def test_btc_backtest_adapter_with_registry(btc_backtest_config: BTCBacktestConfig) -> None:
    """BacktestStrategyAdapter 通过注册表名称加载策略并运行回测。"""
    import strategy.btc  # noqa: F401 — trigger auto_register

    engine = _build_engine_with_data(btc_backtest_config, bar_count=800)

    adapter = BacktestStrategyAdapter(
        "btc_momentum",
        StrategyConfig(
            name="adapter_bt",
            symbols=["BTCUSDT"],
            params={
                "fast_period": 3,
                "slow_period": 10,
                "volume_ma_period": 8,
                "momentum_threshold": 0.005,
                "volume_surge_ratio": 0.5,
                "atr_period": 6,
                "trailing_stop_atr_mult": 1.5,
            },
        ),
    )
    engine.set_strategy(adapter)

    result = engine.run()

    assert result.metrics.total_trades > 0, "适配器应正确桥接策略并产生交易"


def test_btc_backtest_result_has_performance_metrics(
    btc_backtest_config: BTCBacktestConfig,
) -> None:
    """验证回测结果包含完整的绩效指标。"""
    engine = _build_engine_with_data(btc_backtest_config, bar_count=500)

    config = StrategyConfig(
        name="metrics_bt",
        symbols=["BTCUSDT"],
        params={
            "fast_period": 5,
            "slow_period": 15,
            "volume_ma_period": 10,
            "momentum_threshold": 0.01,
            "volume_surge_ratio": 0.8,
            "atr_period": 8,
        },
    )
    strategy = BTCMomentumStrategy(config)
    engine.set_strategy(strategy)

    result = engine.run()

    assert result.metrics is not None
    assert isinstance(result.metrics.sharpe_ratio, float)
    assert isinstance(result.metrics.max_drawdown, Decimal)
    assert result.started_at is not None
    assert result.finished_at is not None
