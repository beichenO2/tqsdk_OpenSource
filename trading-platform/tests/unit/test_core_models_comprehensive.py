"""Comprehensive unit tests for core domain models, enums, events, and strategy schemas."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest

_repo = Path(__file__).resolve().parents[2]
for p in [_repo, _repo / "packages" / "core", _repo / "packages"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core.enums.direction import Direction, Offset
from core.enums.market import AssetClass, Exchange
from core.enums.order_status import OrderStatus
from core.events.base import DomainEvent, OrderEvent, PositionEvent, TradeEvent
from core.models.bar import Bar
from core.models.order import Order
from core.models.position import Position
from core.models.tick import Tick
from core.models.trade import Trade
from core.schemas.strategy import StrategyConfig, StrategyMeta


# ---------------------------------------------------------------------------
# Bar (10)
# ---------------------------------------------------------------------------


def test_bar_create_valid() -> None:
    dt = datetime(2026, 4, 15, 9, 30, tzinfo=UTC)
    bar = Bar(
        symbol="rb2505",
        datetime=dt,
        open=Decimal("3500"),
        high=Decimal("3510"),
        low=Decimal("3495"),
        close=Decimal("3505"),
        volume=12_000,
    )
    assert bar.symbol == "rb2505"
    assert bar.datetime == dt
    assert bar.close == Decimal("3505")
    assert bar.volume == 12_000


def test_bar_high_less_than_low_raises_value_error() -> None:
    with pytest.raises(ValueError, match="high must be greater than or equal to low"):
        Bar(
            symbol="rb2505",
            datetime=datetime.now(UTC),
            open=Decimal("1"),
            high=Decimal("1"),
            low=Decimal("2"),
            close=Decimal("1"),
            volume=1,
        )


def test_bar_high_equals_low_is_valid() -> None:
    bar = Bar(
        symbol="x",
        datetime=datetime.now(UTC),
        open=Decimal("10"),
        high=Decimal("10"),
        low=Decimal("10"),
        close=Decimal("10"),
        volume=1,
    )
    assert bar.high == bar.low


def test_bar_default_duration_seconds_is_60() -> None:
    bar = Bar(
        symbol="x",
        datetime=datetime.now(UTC),
        open=Decimal("1"),
        high=Decimal("2"),
        low=Decimal("1"),
        close=Decimal("2"),
        volume=1,
    )
    assert bar.duration_seconds == 60


def test_bar_explicit_duration_seconds_overrides_default() -> None:
    bar = Bar(
        symbol="x",
        datetime=datetime.now(UTC),
        open=Decimal("1"),
        high=Decimal("2"),
        low=Decimal("1"),
        close=Decimal("2"),
        volume=1,
        duration_seconds=300,
    )
    assert bar.duration_seconds == 300


def test_bar_open_interest_optional_defaults_to_none() -> None:
    bar = Bar(
        symbol="x",
        datetime=datetime.now(UTC),
        open=Decimal("1"),
        high=Decimal("2"),
        low=Decimal("1"),
        close=Decimal("2"),
        volume=1,
    )
    assert bar.open_interest is None


def test_bar_open_interest_can_be_set() -> None:
    bar = Bar(
        symbol="x",
        datetime=datetime.now(UTC),
        open=Decimal("1"),
        high=Decimal("2"),
        low=Decimal("1"),
        close=Decimal("2"),
        volume=1,
        open_interest=99_999,
    )
    assert bar.open_interest == 99_999


def test_bar_ohlc_fields_are_decimal_instances() -> None:
    bar = Bar(
        symbol="x",
        datetime=datetime.now(UTC),
        open=Decimal("1.5"),
        high=Decimal("2.5"),
        low=Decimal("1.25"),
        close=Decimal("2.0"),
        volume=10,
    )
    for name in ("open", "high", "low", "close"):
        assert isinstance(getattr(bar, name), Decimal)


def test_bar_model_dump_contains_expected_keys_and_values() -> None:
    dt = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    bar = Bar(
        symbol="cu2506",
        datetime=dt,
        open=Decimal("10"),
        high=Decimal("11"),
        low=Decimal("9"),
        close=Decimal("10.5"),
        volume=100,
        open_interest=500,
        duration_seconds=60,
    )
    data = bar.model_dump()
    assert data["symbol"] == "cu2506"
    assert data["datetime"] == dt
    assert data["open"] == Decimal("10")
    assert data["high"] == Decimal("11")
    assert data["low"] == Decimal("9")
    assert data["close"] == Decimal("10.5")
    assert data["volume"] == 100
    assert data["open_interest"] == 500
    assert data["duration_seconds"] == 60


def test_bar_model_dump_mode_json_serializes_datetimes() -> None:
    bar = Bar(
        symbol="x",
        datetime=datetime(2026, 4, 15, 0, 0, tzinfo=UTC),
        open=Decimal("1"),
        high=Decimal("2"),
        low=Decimal("1"),
        close=Decimal("2"),
        volume=1,
    )
    dumped = bar.model_dump(mode="json")
    assert isinstance(dumped["datetime"], str)
    assert dumped["symbol"] == "x"


# ---------------------------------------------------------------------------
# Order (10)
# ---------------------------------------------------------------------------


def test_order_create_valid() -> None:
    order = Order(
        strategy_id="s1",
        symbol="rb2505",
        exchange=Exchange.SHFE,
        direction=Direction.LONG,
        offset=Offset.OPEN,
        price=Decimal("3500"),
        volume=5,
    )
    assert order.strategy_id == "s1"
    assert order.symbol == "rb2505"
    assert order.exchange is Exchange.SHFE
    assert order.direction is Direction.LONG
    assert order.offset is Offset.OPEN
    assert order.price == Decimal("3500")
    assert order.volume == 5


def test_order_default_status_is_pending() -> None:
    order = Order(
        strategy_id="s",
        symbol="x",
        exchange=Exchange.DCE,
        direction=Direction.SHORT,
        offset=Offset.CLOSE,
        price=Decimal("1"),
        volume=1,
    )
    assert order.status is OrderStatus.PENDING


def test_order_remaining_is_volume_minus_filled_volume() -> None:
    order = Order(
        strategy_id="s",
        symbol="x",
        exchange=Exchange.CZCE,
        direction=Direction.LONG,
        offset=Offset.OPEN,
        price=Decimal("1"),
        volume=10,
        filled_volume=3,
    )
    assert order.remaining == 7


def test_order_remaining_when_fully_filled_is_zero() -> None:
    order = Order(
        strategy_id="s",
        symbol="x",
        exchange=Exchange.CFFEX,
        direction=Direction.LONG,
        offset=Offset.CLOSE_TODAY,
        price=Decimal("1"),
        volume=4,
        filled_volume=4,
    )
    assert order.remaining == 0


def test_order_id_is_auto_generated_hex_string() -> None:
    order = Order(
        strategy_id="s",
        symbol="x",
        exchange=Exchange.INE,
        direction=Direction.LONG,
        offset=Offset.OPEN,
        price=Decimal("1"),
        volume=1,
    )
    assert isinstance(order.order_id, str)
    assert len(order.order_id) == 32


def test_order_ids_are_unique_across_instances() -> None:
    kwargs = dict(
        strategy_id="s",
        symbol="x",
        exchange=Exchange.GFEX,
        direction=Direction.LONG,
        offset=Offset.OPEN,
        price=Decimal("1"),
        volume=1,
    )
    a = Order(**kwargs)
    b = Order(**kwargs)
    assert a.order_id != b.order_id


def test_order_model_is_not_frozen_mutable_fields() -> None:
    order = Order(
        strategy_id="s",
        symbol="x",
        exchange=Exchange.BINANCE,
        direction=Direction.SHORT,
        offset=Offset.OPEN,
        price=Decimal("1"),
        volume=2,
    )
    assert order.model_config.get("frozen") is False
    order.filled_volume = 1
    assert order.filled_volume == 1


def test_order_filled_volume_defaults_to_zero() -> None:
    order = Order(
        strategy_id="s",
        symbol="x",
        exchange=Exchange.OKX,
        direction=Direction.LONG,
        offset=Offset.OPEN,
        price=Decimal("1"),
        volume=3,
    )
    assert order.filled_volume == 0


def test_order_avg_fill_price_optional() -> None:
    pending = Order(
        strategy_id="s",
        symbol="x",
        exchange=Exchange.SHFE,
        direction=Direction.LONG,
        offset=Offset.OPEN,
        price=Decimal("10"),
        volume=1,
    )
    assert pending.avg_fill_price is None
    filled = Order(
        strategy_id="s",
        symbol="x",
        exchange=Exchange.SHFE,
        direction=Direction.LONG,
        offset=Offset.OPEN,
        price=Decimal("10"),
        volume=1,
        avg_fill_price=Decimal("10.25"),
    )
    assert filled.avg_fill_price == Decimal("10.25")


def test_order_model_dump_preserves_enums_and_decimals() -> None:
    order = Order(
        strategy_id="s",
        symbol="x",
        exchange=Exchange.DCE,
        direction=Direction.SHORT,
        offset=Offset.CLOSE,
        price=Decimal("2.5"),
        volume=2,
        status=OrderStatus.SUBMITTED,
    )
    data = order.model_dump()
    assert data["exchange"] is Exchange.DCE
    assert data["status"] is OrderStatus.SUBMITTED
    assert data["price"] == Decimal("2.5")


# ---------------------------------------------------------------------------
# Position (8)
# ---------------------------------------------------------------------------


def test_position_create_valid() -> None:
    pos = Position(
        symbol="rb2505",
        exchange=Exchange.SHFE,
        direction=Direction.LONG,
        volume=10,
        available=4,
        avg_price=Decimal("3500"),
        margin=Decimal("100"),
        float_pnl=Decimal("12.5"),
        close_pnl=Decimal("-3"),
    )
    assert pos.symbol == "rb2505"
    assert pos.exchange is Exchange.SHFE
    assert pos.direction is Direction.LONG


def test_position_frozen_is_volume_minus_available() -> None:
    pos = Position(
        symbol="x",
        exchange=Exchange.DCE,
        direction=Direction.SHORT,
        volume=10,
        available=3,
    )
    assert pos.frozen == 7


def test_position_frozen_zero_when_fully_available() -> None:
    pos = Position(
        symbol="x",
        exchange=Exchange.CZCE,
        direction=Direction.LONG,
        volume=5,
        available=5,
    )
    assert pos.frozen == 0


def test_position_volume_and_available_default_to_zero() -> None:
    pos = Position(symbol="x", exchange=Exchange.CFFEX, direction=Direction.LONG)
    assert pos.volume == 0
    assert pos.available == 0


def test_position_decimal_fields_default_to_zero_decimal() -> None:
    pos = Position(symbol="x", exchange=Exchange.INE, direction=Direction.SHORT)
    assert pos.avg_price == Decimal("0")
    assert pos.margin == Decimal("0")
    assert pos.float_pnl == Decimal("0")
    assert pos.close_pnl == Decimal("0")


def test_position_frozen_negative_when_available_exceeds_volume() -> None:
    pos = Position(
        symbol="x",
        exchange=Exchange.GFEX,
        direction=Direction.LONG,
        volume=2,
        available=5,
    )
    assert pos.frozen == -3


def test_position_model_dump_roundtrip_core_fields() -> None:
    pos = Position(
        symbol="x",
        exchange=Exchange.BINANCE,
        direction=Direction.LONG,
        volume=1,
        available=0,
        avg_price=Decimal("100"),
    )
    d = pos.model_dump()
    assert d["symbol"] == "x"
    assert d["exchange"] is Exchange.BINANCE
    assert d["avg_price"] == Decimal("100")


def test_position_pnl_fields_can_be_non_zero() -> None:
    pos = Position(
        symbol="x",
        exchange=Exchange.OKX,
        direction=Direction.SHORT,
        volume=1,
        available=1,
        float_pnl=Decimal("0.01"),
        close_pnl=Decimal("-0.02"),
    )
    assert pos.float_pnl == Decimal("0.01")
    assert pos.close_pnl == Decimal("-0.02")


# ---------------------------------------------------------------------------
# Trade (6)
# ---------------------------------------------------------------------------


def test_trade_create_valid() -> None:
    tr = Trade(
        order_id="oid",
        strategy_id="sid",
        symbol="rb2505",
        exchange=Exchange.SHFE,
        direction=Direction.LONG,
        offset=Offset.OPEN,
        price=Decimal("3500"),
        volume=1,
    )
    assert tr.order_id == "oid"
    assert tr.strategy_id == "sid"
    assert tr.volume == 1


def test_trade_id_auto_generated() -> None:
    tr = Trade(
        order_id="o",
        strategy_id="s",
        symbol="x",
        exchange=Exchange.DCE,
        direction=Direction.LONG,
        offset=Offset.CLOSE,
        price=Decimal("1"),
        volume=1,
    )
    assert isinstance(tr.trade_id, str)
    assert len(tr.trade_id) == 32


def test_trade_commission_defaults_to_zero_decimal() -> None:
    tr = Trade(
        order_id="o",
        strategy_id="s",
        symbol="x",
        exchange=Exchange.CZCE,
        direction=Direction.SHORT,
        offset=Offset.OPEN,
        price=Decimal("1"),
        volume=1,
    )
    assert tr.commission == Decimal("0")


def test_trade_traded_at_defaults_to_now_utc() -> None:
    before = datetime.now(UTC)
    tr = Trade(
        order_id="o",
        strategy_id="s",
        symbol="x",
        exchange=Exchange.CFFEX,
        direction=Direction.LONG,
        offset=Offset.OPEN,
        price=Decimal("1"),
        volume=1,
    )
    after = datetime.now(UTC)
    assert tr.traded_at.tzinfo is UTC
    assert before <= tr.traded_at <= after


def test_trade_model_dump_includes_ids_and_enums() -> None:
    tr = Trade(
        order_id="o1",
        strategy_id="s1",
        symbol="x",
        exchange=Exchange.INE,
        direction=Direction.LONG,
        offset=Offset.CLOSE_TODAY,
        price=Decimal("3"),
        volume=2,
        commission=Decimal("1.25"),
    )
    d = tr.model_dump()
    assert d["order_id"] == "o1"
    assert d["exchange"] is Exchange.INE
    assert d["commission"] == Decimal("1.25")


def test_trade_ids_unique_per_instance() -> None:
    kwargs = dict(
        order_id="o",
        strategy_id="s",
        symbol="x",
        exchange=Exchange.GFEX,
        direction=Direction.LONG,
        offset=Offset.OPEN,
        price=Decimal("1"),
        volume=1,
    )
    assert Trade(**kwargs).trade_id != Trade(**kwargs).trade_id


# ---------------------------------------------------------------------------
# Tick (6)
# ---------------------------------------------------------------------------


def test_tick_create_valid_minimal_optional_fields_none() -> None:
    tick = Tick(
        symbol="rb2505",
        datetime=datetime.now(UTC),
        last_price=Decimal("3500"),
        highest=Decimal("3510"),
        lowest=Decimal("3490"),
        volume=1_000_000,
        amount=Decimal("1234567.89"),
    )
    assert tick.symbol == "rb2505"
    assert tick.open_interest is None
    assert tick.bid_price1 is None


def test_tick_open_interest_optional_set() -> None:
    tick = Tick(
        symbol="x",
        datetime=datetime.now(UTC),
        last_price=Decimal("1"),
        highest=Decimal("2"),
        lowest=Decimal("1"),
        volume=1,
        amount=Decimal("1"),
        open_interest=42,
    )
    assert tick.open_interest == 42


def test_tick_bid_ask_levels_optional() -> None:
    tick = Tick(
        symbol="x",
        datetime=datetime.now(UTC),
        last_price=Decimal("10"),
        highest=Decimal("11"),
        lowest=Decimal("9"),
        volume=100,
        amount=Decimal("1000"),
        bid_price1=Decimal("9.9"),
        bid_volume1=5,
        ask_price1=Decimal("10.1"),
        ask_volume1=7,
    )
    assert tick.bid_price1 == Decimal("9.9")
    assert tick.bid_volume1 == 5
    assert tick.ask_price1 == Decimal("10.1")
    assert tick.ask_volume1 == 7


def test_tick_decimal_price_fields_are_decimal() -> None:
    tick = Tick(
        symbol="x",
        datetime=datetime.now(UTC),
        last_price=Decimal("1.1"),
        highest=Decimal("2.2"),
        lowest=Decimal("0.9"),
        volume=1,
        amount=Decimal("3.3"),
    )
    assert isinstance(tick.last_price, Decimal)
    assert isinstance(tick.amount, Decimal)


def test_tick_model_dump_json_serializes_decimals() -> None:
    tick = Tick(
        symbol="x",
        datetime=datetime(2026, 4, 15, 0, 0, tzinfo=UTC),
        last_price=Decimal("1"),
        highest=Decimal("2"),
        lowest=Decimal("1"),
        volume=1,
        amount=Decimal("1.5"),
    )
    dumped = tick.model_dump(mode="json")
    assert dumped["last_price"] == "1"
    assert dumped["amount"] == "1.5"


def test_tick_volume_is_int_amount_decimal() -> None:
    tick = Tick(
        symbol="x",
        datetime=datetime.now(UTC),
        last_price=Decimal("1"),
        highest=Decimal("2"),
        lowest=Decimal("1"),
        volume=999,
        amount=Decimal("0"),
    )
    assert isinstance(tick.volume, int)
    assert tick.amount == Decimal("0")


# ---------------------------------------------------------------------------
# Enums (15)
# ---------------------------------------------------------------------------


def test_direction_long_and_short_exist() -> None:
    assert Direction.LONG.value == "LONG"
    assert Direction.SHORT.value == "SHORT"


def test_direction_strenum_compares_equal_to_equivalent_string() -> None:
    assert Direction.LONG == "LONG"
    assert Direction.SHORT == "SHORT"


def test_direction_membership_in_collection() -> None:
    assert Direction.LONG in (Direction.LONG, Direction.SHORT)
    members = {m.value for m in Direction}
    assert members == {"LONG", "SHORT"}


@pytest.mark.parametrize(
    "offset",
    [Offset.OPEN, Offset.CLOSE, Offset.CLOSE_TODAY],
)
def test_offset_enum_values_exist(offset: Offset) -> None:
    assert offset in Offset


@pytest.mark.parametrize(
    "offset_str",
    ["OPEN", "CLOSE", "CLOSE_TODAY"],
)
def test_offset_strenum_string_equality(offset_str: str) -> None:
    assert Offset(offset_str) == offset_str


@pytest.mark.parametrize(
    "exchange",
    [
        Exchange.SHFE,
        Exchange.DCE,
        Exchange.CZCE,
        Exchange.CFFEX,
        Exchange.INE,
        Exchange.GFEX,
        Exchange.BINANCE,
        Exchange.OKX,
    ],
)
def test_exchange_enum_members_exist(exchange: Exchange) -> None:
    assert exchange in Exchange
    assert isinstance(exchange.value, str)


def test_asset_class_all_expected_values_exist() -> None:
    expected = {"FUTURES", "OPTIONS", "CRYPTO_SPOT", "CRYPTO_PERP"}
    assert {a.value for a in AssetClass} == expected
    for ac in AssetClass:
        assert ac == ac.value


def test_asset_class_membership() -> None:
    assert AssetClass.FUTURES in AssetClass
    assert "CRYPTO_PERP" == AssetClass.CRYPTO_PERP


def test_order_status_all_expected_values_exist() -> None:
    expected = {
        "PENDING",
        "SUBMITTED",
        "PARTIAL_FILLED",
        "FILLED",
        "CANCELLED",
        "REJECTED",
        "FAILED",
    }
    assert {s.value for s in OrderStatus} == expected


@pytest.mark.parametrize(
    "status",
    [
        OrderStatus.PENDING,
        OrderStatus.SUBMITTED,
        OrderStatus.PARTIAL_FILLED,
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
        OrderStatus.REJECTED,
        OrderStatus.FAILED,
    ],
)
def test_order_status_each_member_in_enum(status: OrderStatus) -> None:
    assert status in OrderStatus
    assert status == status.value


def test_order_status_string_compare_pending() -> None:
    assert OrderStatus.PENDING == "PENDING"


# ---------------------------------------------------------------------------
# Events (8)
# ---------------------------------------------------------------------------


def test_domain_event_requires_event_type() -> None:
    ev = DomainEvent(event_type="custom")
    assert ev.event_type == "custom"


def test_domain_event_timestamp_auto_populated() -> None:
    fixed = datetime(2026, 4, 15, 8, 0, 0, tzinfo=UTC)
    with patch("core.events.base.datetime") as mock_dt:
        mock_dt.now.return_value = fixed
        mock_dt.UTC = UTC
        ev = DomainEvent(event_type="t")
    assert ev.timestamp == fixed


def test_domain_event_payload_defaults_to_empty_dict() -> None:
    ev = DomainEvent(event_type="t")
    assert ev.payload == {}


def test_domain_event_payload_can_be_provided() -> None:
    ev = DomainEvent(event_type="t", payload={"a": 1})
    assert ev.payload == {"a": 1}


def test_order_event_has_default_event_type_and_ids() -> None:
    ev = OrderEvent()
    assert ev.event_type == "order"
    assert ev.order_id == ""
    assert ev.strategy_id == ""


def test_trade_event_has_default_event_type_and_ids() -> None:
    ev = TradeEvent()
    assert ev.event_type == "trade"
    assert ev.trade_id == ""
    assert ev.order_id == ""


def test_position_event_has_default_event_type_and_symbol() -> None:
    ev = PositionEvent()
    assert ev.event_type == "position"
    assert ev.symbol == ""


def test_events_model_dump_includes_base_fields() -> None:
    ev = OrderEvent(order_id="o1", strategy_id="s1", payload={"x": True})
    d = ev.model_dump()
    assert d["event_type"] == "order"
    assert d["order_id"] == "o1"
    assert d["payload"] == {"x": True}
    assert "timestamp" in d


# ---------------------------------------------------------------------------
# StrategyConfig / StrategyMeta (6)
# ---------------------------------------------------------------------------


def test_strategy_config_create_required_fields() -> None:
    cfg = StrategyConfig(
        strategy_id="s1",
        name="momentum",
        symbols=["rb2505", "hc2505"],
    )
    assert cfg.strategy_id == "s1"
    assert cfg.name == "momentum"
    assert cfg.symbols == ["rb2505", "hc2505"]


def test_strategy_config_defaults_enabled_max_position_capital_params() -> None:
    cfg = StrategyConfig(strategy_id="s", name="n", symbols=["x"])
    assert cfg.enabled is True
    assert cfg.max_position == 10
    assert cfg.capital == 1_000_000.0
    assert cfg.params == {}


def test_strategy_config_params_override() -> None:
    cfg = StrategyConfig(
        strategy_id="s",
        name="n",
        symbols=["x"],
        params={"threshold": 0.5, "flag": True, "label": "a"},
        enabled=False,
        max_position=3,
        capital=50_000.0,
    )
    assert cfg.params["threshold"] == 0.5
    assert cfg.enabled is False
    assert cfg.max_position == 3
    assert cfg.capital == 50_000.0


def test_strategy_config_model_dump_mode_python() -> None:
    cfg = StrategyConfig(strategy_id="s", name="n", symbols=["a", "b"])
    d = cfg.model_dump()
    assert d["symbols"] == ["a", "b"]
    assert d["strategy_id"] == "s"


def test_strategy_meta_create_and_defaults() -> None:
    meta = StrategyMeta(name="n", version="1.0.0")
    assert meta.author == ""
    assert meta.description == ""
    assert meta.asset_class == "FUTURES"
    assert meta.tags == []


def test_strategy_meta_model_dump_with_tags_and_description() -> None:
    meta = StrategyMeta(
        name="n",
        version="1.0.0",
        author="a",
        description="d",
        asset_class="CRYPTO_PERP",
        tags=["t1", "t2"],
    )
    d = meta.model_dump()
    assert d["tags"] == ["t1", "t2"]
    assert d["asset_class"] == "CRYPTO_PERP"
    assert d["description"] == "d"

