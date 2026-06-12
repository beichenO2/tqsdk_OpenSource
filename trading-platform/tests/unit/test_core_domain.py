"""Unit tests for trading-core domain models and enums (packages/core)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from core.enums.direction import Direction, Offset
from core.enums.market import AssetClass, Exchange
from core.enums.order_status import OrderStatus
from core.enums.order_type import OrderType as CoreOrderType
from core.models.bar import Bar
from core.models.order import Order
from core.models.position import Position
from core.models.trade import Trade


def test_direction_and_offset_str_enum_values() -> None:
    assert Direction.LONG == "LONG"
    assert Offset.CLOSE_TODAY == "CLOSE_TODAY"


def test_exchange_and_asset_class_values() -> None:
    assert Exchange.BINANCE.value == "BINANCE"
    assert AssetClass.CRYPTO_PERP == "CRYPTO_PERP"


def test_order_status_includes_failed() -> None:
    assert OrderStatus.FAILED == "FAILED"


def test_core_order_type_lowercase_values() -> None:
    assert CoreOrderType.LIMIT == "limit"
    assert CoreOrderType.STOP_LIMIT == "stop_limit"


def test_order_remaining_volume() -> None:
    o = Order(
        strategy_id="s1",
        symbol="rb2505",
        exchange=Exchange.SHFE,
        direction=Direction.LONG,
        offset=Offset.OPEN,
        price=Decimal("3000"),
        volume=10,
        filled_volume=3,
    )
    assert o.remaining == 7


def test_position_frozen_volume() -> None:
    p = Position(
        symbol="BTC-USDT",
        exchange=Exchange.BINANCE,
        direction=Direction.LONG,
        volume=5,
        available=2,
    )
    assert p.frozen == 3


def test_bar_model_fields() -> None:
    ts = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
    b = Bar(
        symbol="rb2505",
        datetime=ts,
        open=Decimal("1"),
        high=Decimal("3"),
        low=Decimal("0.5"),
        close=Decimal("2"),
        volume=100,
    )
    assert b.duration_seconds == 60
    assert b.open_interest is None


def test_trade_round_trip_ids() -> None:
    t = Trade(
        order_id="oid-1",
        strategy_id="s1",
        symbol="rb2505",
        exchange=Exchange.SHFE,
        direction=Direction.SHORT,
        offset=Offset.CLOSE,
        price=Decimal("4000"),
        volume=1,
    )
    assert len(t.trade_id) == 32
    assert t.commission == Decimal("0")


@pytest.mark.parametrize(
    ("high", "low", "expect"),
    [
        (Decimal("2"), Decimal("1"), None),
        (Decimal("1"), Decimal("2"), ValueError),
    ],
)
def test_bar_high_must_not_be_below_low(
    high: Decimal, low: Decimal, expect: type[BaseException] | None
) -> None:
    ts = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
    if expect is None:
        Bar(
            symbol="x",
            datetime=ts,
            open=Decimal("1"),
            high=high,
            low=low,
            close=Decimal("1.5"),
            volume=1,
        )
    else:
        with pytest.raises(expect):
            Bar(
                symbol="x",
                datetime=ts,
                open=Decimal("1"),
                high=high,
                low=low,
                close=Decimal("1.5"),
                volume=1,
            )
