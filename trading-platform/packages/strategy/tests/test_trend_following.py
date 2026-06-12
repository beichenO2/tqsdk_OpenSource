"""Tests for BTC trend_following strategy signal logic."""

import asyncio

from strategy.base import StrategyConfig
from strategy.btc.trend_following import BTCTrendFollowingStrategy


def _run(coro):
    return asyncio.run(coro)


def _make_strategy() -> BTCTrendFollowingStrategy:
    cfg = StrategyConfig(
        name="test_trend",
        symbols=["BTCUSDT"],
        params={"adx_threshold": 20.0},
    )
    return BTCTrendFollowingStrategy(cfg)


def _bar(close: float, high: float | None = None, low: float | None = None):
    h = high if high is not None else close + 50
    l = low if low is not None else close - 50
    return {"open": close, "high": h, "low": l, "close": close, "volume": 1000}


def test_warmup_no_signals():
    """During warmup period, no signals should be produced."""
    strat = _make_strategy()
    for price in range(30000, 30010):
        signals = _run(strat.on_bar("BTCUSDT", _bar(float(price))))
        assert signals == []


def test_trend_signal_after_warmup():
    """After enough bars, the strategy should produce signals in trending markets."""
    strat = _make_strategy()
    signals_collected = []
    for i in range(200):
        price = 30000 + i * 10
        sigs = _run(strat.on_bar("BTCUSDT", _bar(float(price))))
        signals_collected.extend(sigs)
    # In a strong uptrend, we expect at least one LONG_ENTRY
    long_entries = [s for s in signals_collected if s.signal_type.value == "long_entry"]
    assert len(long_entries) >= 1, "Expected at least one long entry signal in uptrend"


def test_no_duplicate_entry_with_position():
    """Should not generate entry when already holding a position."""
    strat = _make_strategy()
    from strategy.base import OrderSide, Position
    strat.update_position(Position(
        symbol="BTCUSDT", side=OrderSide.BUY, qty=1.0, avg_price=30000.0
    ))
    for i in range(100):
        price = 30000 + i * 10
        sigs = _run(strat.on_bar("BTCUSDT", _bar(float(price))))
        entries = [s for s in sigs if "entry" in s.signal_type.value]
        assert entries == [], f"Got entry signal while holding position at bar {i}"
