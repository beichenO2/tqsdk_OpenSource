"""Tests for AdaptiveTrendStrategy — verifying DEMA/TEMA/KAMA fixes."""

import asyncio

from strategy.base import StrategyConfig
from strategy.templates.adaptive_trend import AdaptiveTrendStrategy


def _make_strategy(ma_type: str = "kama", fast: int = 5, slow: int = 10, er: int = 5) -> AdaptiveTrendStrategy:
    cfg = StrategyConfig(
        name=f"test_{ma_type}",
        symbols=["TEST"],
        params={
            "ma_type": ma_type,
            "fast_period": fast,
            "slow_period": slow,
            "er_period": er,
            "atr_period": 5,
            "trailing_stop_atr_mult": 2.0,
            "trend_strength_min": 0.01,
            "max_hold_bars": 200,
        },
    )
    return AdaptiveTrendStrategy(cfg)


def _feed_bars(strategy: AdaptiveTrendStrategy, prices: list[float]) -> list:
    all_signals = []
    for p in prices:
        bar = {"open": p, "high": p + 1, "low": p - 1, "close": p}
        sigs = asyncio.run(strategy.on_bar("TEST", bar))
        all_signals.extend(sigs)
    return all_signals


def test_dema_differs_from_ema():
    """DEMA should not equal EMA — the old bug made them identical."""
    data = list(range(1, 31))
    data_float = [float(x) for x in data]
    strategy = _make_strategy("dema")
    ema_val = strategy._ema(data_float, 5)
    dema_val = strategy._dema(data_float, 5)
    assert ema_val is not None
    assert dema_val is not None
    assert dema_val != ema_val, "DEMA should differ from EMA"


def test_tema_differs_from_dema():
    """TEMA should not equal DEMA."""
    data = [float(x) for x in range(1, 51)]
    strategy = _make_strategy("tema")
    dema_val = strategy._dema(data, 5)
    tema_val = strategy._tema(data, 5)
    assert dema_val is not None
    assert tema_val is not None
    assert tema_val != dema_val, "TEMA should differ from DEMA"


def test_kama_fast_slow_differ():
    """KAMA with different er_period values should produce different results."""
    strategy = _make_strategy("kama", fast=5, slow=15)
    prices = [100 + i * 0.5 for i in range(50)]
    _feed_bars(strategy, prices)

    fast_val = strategy._kama_value("TEST", er_period=5)
    slow_val = strategy._kama_value("TEST", er_period=15)
    assert fast_val is not None
    assert slow_val is not None
    assert fast_val != slow_val, "KAMA with different er_period must produce different values"


def test_dema_trend_produces_signals():
    """In a strong trend, DEMA mode should eventually produce entry signals."""
    strategy = _make_strategy("dema", fast=3, slow=8)
    flat = [100.0] * 30
    trending = [100.0 + i * 3.0 for i in range(40)]
    prices = flat + trending
    signals = _feed_bars(strategy, prices)
    assert len(signals) > 0, "Strong trend after flat should trigger DEMA signals"


def test_ema_series_length():
    """_ema_series should produce a series shorter than input by (period - 1)."""
    strategy = _make_strategy("dema")
    data = [float(x) for x in range(20)]
    series = strategy._ema_series(data, 5)
    assert series is not None
    assert len(series) == len(data) - 5 + 1
