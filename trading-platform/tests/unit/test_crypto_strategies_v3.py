"""Unit tests for portfolio / ensemble / funding-rate crypto strategies (v3)."""

from __future__ import annotations

import pytest
from strategy.base import OrderSide, Position, SignalType, StrategyConfig
from strategy.btc.ensemble_strategy import EnsembleStrategy
from strategy.btc.funding_rate_arb import FundingRateArbitrage
from strategy.btc.portfolio_strategy import PortfolioStrategy
from strategy.registry import StrategyRegistry


def _bar(close: float, funding: float | None = None) -> dict:
    b = {
        "open": close * 0.999,
        "high": close * 1.002,
        "low": close * 0.997,
        "close": close,
        "volume": 1_000.0,
        "taker_buy_volume": 520.0,
    }
    if funding is not None:
        b["funding_rate"] = funding
    return b


@pytest.mark.asyncio
async def test_portfolio_strategy_registry_and_metadata() -> None:
    cfg = StrategyConfig(
        name="PortTest",
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        params={
            "allocation_weights": {"BTCUSDT": 0.5, "ETHUSDT": 0.3, "SOLUSDT": 0.2},
            "sub_strategy_by_symbol": {
                "BTCUSDT": "scalp_momentum",
                "ETHUSDT": "scalp_momentum",
                "SOLUSDT": "scalp_momentum",
            },
            "rebalance_interval_bars": 10_000,
        },
    )
    cls = StrategyRegistry.get("crypto_portfolio")
    assert cls is not None
    strat = PortfolioStrategy(cfg)
    md = {"BTCUSDT": _bar(50_000), "ETHUSDT": _bar(3000), "SOLUSDT": _bar(100)}
    sigs = await strat.generate_signals(md)
    for s in sigs:
        assert s.strategy_id == strat.strategy_id
        if s.signal_type != SignalType.HOLD:
            assert "allocation_weight" in s.metadata


@pytest.mark.asyncio
async def test_portfolio_rebalance_emits_trim_for_overweight_long() -> None:
    cfg = StrategyConfig(
        name="PortRebal",
        symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
        params={
            "allocation_weights": {"BTCUSDT": 0.34, "ETHUSDT": 0.33, "BNBUSDT": 0.33},
            "rebalance_interval_bars": 1,
            "rebalance_drift_threshold": 0.01,
            "sub_strategy_by_symbol": {
                "BTCUSDT": "scalp_momentum",
                "ETHUSDT": "scalp_momentum",
                "BNBUSDT": "scalp_momentum",
            },
        },
    )
    strat = PortfolioStrategy(cfg)
    strat.update_position(
        Position(symbol="BTCUSDT", side=OrderSide.BUY, qty=2.0, avg_price=50_000.0)
    )
    strat.update_position(
        Position(symbol="ETHUSDT", side=OrderSide.BUY, qty=5.0, avg_price=3000.0)
    )
    strat._last_closes["BTCUSDT"] = 50_000.0
    strat._last_closes["ETHUSDT"] = 3000.0
    strat._last_closes["BNBUSDT"] = 400.0
    strat._generate_cycles = 1
    sigs = await strat._maybe_rebalance()
    assert any(s.signal_type == SignalType.LONG_EXIT and s.metadata.get("rebalance") for s in sigs)


@pytest.mark.asyncio
async def test_ensemble_vote_produces_signal_when_children_align() -> None:
    cfg = StrategyConfig(
        name="EnsTest",
        symbols=["BTCUSDT"],
        params={
            "ensemble_mode": "vote",
            "action_threshold": 0.34,
            "sub_strategy_weights": [0.34, 0.33, 0.33],
            # btc_trend_following/btc_multifactor/btc_momentum archived in 260505
            "sub_strategy_types": (
                "time_series_momentum",
                "scalp_momentum",
                "vol_breakout_scalp",
            ),
        },
    )
    strat = EnsembleStrategy(cfg)
    close = 42_000.0
    last: list = []
    for i in range(200):
        c = close + i * 80
        last = await strat.on_bar("BTCUSDT", _bar(c))
    assert isinstance(last, list)
    assert all(s.strategy_id == strat.strategy_id for s in last)


@pytest.mark.asyncio
async def test_funding_rate_short_on_high_funding() -> None:
    cfg = StrategyConfig(
        name="FundArb",
        symbols=["BTCUSDT"],
        params={"rolling_window": 24, "z_score_mult": 0.5, "static_threshold": 0.0001},
    )
    strat = FundingRateArbitrage(cfg)
    base = 0.00005
    for i in range(30):
        await strat.on_bar("BTCUSDT", _bar(40_000.0 + i, funding=base))
    sigs = await strat.on_bar("BTCUSDT", _bar(40_100.0, funding=0.01))
    assert any(s.signal_type == SignalType.SHORT_ENTRY for s in sigs)


@pytest.mark.asyncio
async def test_funding_rate_long_on_deeply_negative_funding() -> None:
    cfg = StrategyConfig(
        name="FundArb2",
        symbols=["BTCUSDT"],
        params={"rolling_window": 20, "z_score_mult": 0.3, "static_threshold": 0.00005},
    )
    strat = FundingRateArbitrage(cfg)
    base = -0.00002
    for _ in range(25):
        await strat.on_bar("BTCUSDT", _bar(40_000.0, funding=base))
    sigs = await strat.on_bar("BTCUSDT", _bar(40_000.0, funding=-0.005))
    assert any(s.signal_type == SignalType.LONG_ENTRY for s in sigs)


def test_auto_registered_names() -> None:
    assert StrategyRegistry.get("crypto_portfolio") is PortfolioStrategy
    assert StrategyRegistry.get("crypto_ensemble") is EnsembleStrategy
    assert StrategyRegistry.get("funding_rate_arb") is FundingRateArbitrage
