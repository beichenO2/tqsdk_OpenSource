"""Smoke tests for futures strategies v3.

Covers registry-backed names: spread_arb, vol_breakout, volume_price, pairs_trading, bollinger_mr.
"""

from __future__ import annotations

import pytest

from strategy.base import StrategyConfig
from strategy.futures import (
    BollingerMRStrategy,
    PairsTradingStrategy,
    SpreadArbitrage,
    VolBreakoutStrategy,
    VolumePriceStrategy,
)


@pytest.fixture
def single_symbol_cfg() -> StrategyConfig:
    return StrategyConfig(name="t", symbols=["SHFE.rb2501"], params={})


@pytest.fixture
def pair_cfg() -> StrategyConfig:
    return StrategyConfig(
        name="pair",
        symbols=["NEAR.X", "FAR.Y"],
        params={},
    )


def test_spread_arbitrage_instantiated(pair_cfg: StrategyConfig) -> None:
    s = SpreadArbitrage(pair_cfg)
    assert s.name == "pair"
    assert s.strategy_id


def test_vol_breakout_instantiated(single_symbol_cfg: StrategyConfig) -> None:
    s = VolBreakoutStrategy(single_symbol_cfg)
    assert s.name == "t"


def test_volume_price_instantiated(single_symbol_cfg: StrategyConfig) -> None:
    s = VolumePriceStrategy(single_symbol_cfg)
    assert s.name == "t"


def test_pairs_trading_instantiated(pair_cfg: StrategyConfig) -> None:
    s = PairsTradingStrategy(pair_cfg)
    assert s.name == "pair"


def test_bollinger_mr_instantiated(single_symbol_cfg: StrategyConfig) -> None:
    s = BollingerMRStrategy(single_symbol_cfg)
    assert s.name == "t"


def test_generate_signal_returns_signal(single_symbol_cfg: StrategyConfig) -> None:
    s = BollingerMRStrategy(single_symbol_cfg)
    sig = s.generate_signal(
        "SHFE.rb2501",
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000.0},
    )
    assert sig.symbol == "SHFE.rb2501"
    assert sig.signal_type.value in ("hold", "long_entry", "short_entry", "long_exit", "short_exit")
