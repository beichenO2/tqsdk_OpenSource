"""Unit tests for RiskEngine and RiskLimit implementations."""

from __future__ import annotations

from decimal import Decimal

import pytest

from core.enums.direction import Direction, Offset
from core.enums.market import Exchange
from core.models.position import Position
from execution.order_manager import OrderRequest
from risk.engine import RiskEngine
from risk.limits import (
    DailyLossLimit,
    MarginUtilizationLimit,
    MaxOrderSizeLimit,
    MaxPositionLimit,
    OrderFrequencyLimit,
    PriceBandLimit,
    RiskContext,
)


def _open_request(
    symbol: str = "rb2505",
    volume: int = 1,
    price: Decimal = Decimal("3500"),
) -> OrderRequest:
    return OrderRequest(
        symbol=symbol,
        exchange="SHFE",
        direction=Direction.LONG,
        offset=Offset.OPEN,
        price=price,
        volume=volume,
    )


def test_risk_engine_passes_when_no_limits_configured() -> None:
    engine = RiskEngine()
    ok, reason = engine.pre_trade_check(_open_request())
    assert ok is True
    assert reason == ""


def test_max_order_size_limit_rejects_oversized_order() -> None:
    engine = RiskEngine()
    engine.add_limit(MaxOrderSizeLimit(max_volume=10))
    req = _open_request(volume=50)
    ok, reason = engine.pre_trade_check(req)
    assert ok is False
    assert "[MaxOrderSize]" in reason
    assert "50" in reason
    assert "exceeds max 10" in reason


def test_max_order_size_limit_accepts_order_within_limit() -> None:
    engine = RiskEngine()
    engine.add_limit(MaxOrderSizeLimit(max_volume=100))
    ok, reason = engine.pre_trade_check(_open_request(volume=10))
    assert ok is True
    assert reason == ""


def test_max_position_limit_rejects_open_that_exceeds_cap() -> None:
    engine = RiskEngine()
    engine.add_limit(MaxPositionLimit(max_position=100))
    pos = Position(
        symbol="rb2505",
        exchange=Exchange.SHFE,
        direction=Direction.LONG,
        volume=90,
    )
    engine.update_positions([pos])
    ok, reason = engine.pre_trade_check(_open_request(volume=20))
    assert ok is False
    assert "[MaxPosition]" in reason
    assert "110" in reason
    assert "limit 100" in reason


def test_max_position_limit_allows_close_even_if_large() -> None:
    engine = RiskEngine()
    engine.add_limit(MaxPositionLimit(max_position=10))
    req = OrderRequest(
        symbol="rb2505",
        exchange="SHFE",
        direction=Direction.LONG,
        offset=Offset.CLOSE,
        price=Decimal("3500"),
        volume=500,
    )
    ok, reason = engine.pre_trade_check(req)
    assert ok is True


def test_price_band_limit_rejects_far_from_market() -> None:
    engine = RiskEngine()
    engine.add_limit(PriceBandLimit(max_deviation_pct=Decimal("0.05")))
    engine.update_prices({"rb2505": Decimal("100")})
    ok, reason = engine.pre_trade_check(_open_request(price=Decimal("200")))
    assert ok is False
    assert "[PriceBand]" in reason
    assert "deviation" in reason.lower()


def test_price_band_limit_accepts_near_market() -> None:
    engine = RiskEngine()
    engine.add_limit(PriceBandLimit(max_deviation_pct=Decimal("0.05")))
    engine.update_prices({"rb2505": Decimal("100")})
    ok, reason = engine.pre_trade_check(_open_request(price=Decimal("101")))
    assert ok is True


def test_margin_utilization_limit_blocks_open_when_ratio_high() -> None:
    engine = RiskEngine()
    engine.add_limit(MarginUtilizationLimit(max_ratio=Decimal("0.5")))
    engine.update_account(
        balance=Decimal("100000"),
        available=Decimal("10000"),
        margin_ratio=Decimal("0.9"),
    )
    ok, reason = engine.pre_trade_check(_open_request())
    assert ok is False
    assert "[MarginUtilization]" in reason


def test_daily_loss_limit_tripped_rejects_all_orders() -> None:
    engine = RiskEngine()
    limit = DailyLossLimit(max_loss_pct=Decimal("0.05"))
    engine.add_limit(limit)
    limit.trip()
    ok, reason = engine.pre_trade_check(_open_request())
    assert ok is False
    assert "[DailyLoss]" in reason
    assert "circuit-breaker" in reason


def test_check_daily_loss_trips_when_loss_exceeds_threshold() -> None:
    engine = RiskEngine()
    engine.add_limit(DailyLossLimit(max_loss_pct=Decimal("0.05")))
    engine.update_account(
        balance=Decimal("10000"),
        available=Decimal("5000"),
        margin_ratio=Decimal("0.1"),
    )
    # loss_pct = -daily_pnl / balance >= max_loss_pct  => daily_pnl <= -balance * max_loss_pct
    assert engine.check_daily_loss(Decimal("-600")) is False
    assert engine.get_status()["daily_loss_tripped"] is True


def test_order_frequency_limit_blocks_after_burst() -> None:
    limit = OrderFrequencyLimit(max_orders=3, window_seconds=3600.0)
    ctx = RiskContext(positions=[])
    req = _open_request(symbol="ag2506")
    for _ in range(3):
        ok, _ = limit.check(req, ctx)
        assert ok is True
    ok, reason = limit.check(req, ctx)
    assert ok is False
    assert "Frequency" in reason
