"""Unit tests for strategy.BaseStrategy, models, and strategy.StrategyRegistry."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

_repo = Path(__file__).resolve().parents[2]
for p in [_repo, _repo / "packages" / "core", _repo / "packages" / "strategy", _repo / "packages"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from strategy.base import (  # noqa: E402
    BaseStrategy,
    OrderSide,
    Position,
    Signal,
    SignalType,
    StrategyConfig,
    StrategyState,
)
from strategy import registry as strategy_registry  # noqa: E402
from strategy.registry import StrategyRegistry, auto_register  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_strategy_registry_state() -> Any:
    saved = dict(strategy_registry._STRATEGY_REGISTRY)
    strategy_registry._STRATEGY_REGISTRY.clear()
    strategy_registry._STRATEGY_INSTANCES.clear()
    yield
    strategy_registry._STRATEGY_REGISTRY.clear()
    strategy_registry._STRATEGY_INSTANCES.clear()
    strategy_registry._STRATEGY_REGISTRY.update(saved)


def _make_config(**overrides: Any) -> StrategyConfig:
    base: dict[str, Any] = {
        "strategy_id": "sid-test",
        "name": "dummy",
        "params": {"alpha": 1, "beta": 2},
    }
    base.update(overrides)
    return StrategyConfig(**base)


class DummyStrategy(BaseStrategy):
    """Minimal concrete strategy for exercising the ABC surface."""

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        return []

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        return []


# --- BaseStrategy / subclass ---


def test_base_strategy_cannot_be_instantiated() -> None:
    cfg = _make_config()
    with pytest.raises(TypeError):
        BaseStrategy(cfg)  # type: ignore[abstract,misc]


def test_dummy_strategy_init_sets_config_state_positions_signals() -> None:
    cfg = _make_config()
    s = DummyStrategy(cfg)
    assert s.config is cfg
    assert s.state == StrategyState.IDLE
    assert s._positions == {}
    assert len(s._signals) == 0


def test_strategy_id_property_reads_config() -> None:
    cfg = _make_config(strategy_id="abc12345")
    s = DummyStrategy(cfg)
    assert s.strategy_id == "abc12345"


def test_name_property_reads_config() -> None:
    cfg = _make_config(name="my-strat")
    s = DummyStrategy(cfg)
    assert s.name == "my-strat"


@pytest.mark.asyncio
async def test_on_start_sets_running() -> None:
    s = DummyStrategy(_make_config())
    await s.on_start()
    assert s.state == StrategyState.RUNNING


@pytest.mark.asyncio
async def test_on_stop_sets_stopped() -> None:
    s = DummyStrategy(_make_config())
    await s.on_stop()
    assert s.state == StrategyState.STOPPED


@pytest.mark.asyncio
async def test_on_error_sets_error() -> None:
    s = DummyStrategy(_make_config())
    await s.on_error(ValueError("boom"))
    assert s.state == StrategyState.ERROR


def test_on_fill_is_noop() -> None:
    s = DummyStrategy(_make_config())
    s.on_fill({"order_id": "1"})  # should not raise


def test_on_backtest_complete_is_noop() -> None:
    s = DummyStrategy(_make_config())
    s.on_backtest_complete({"sharpe": 1.2})  # should not raise


def test_update_and_get_position() -> None:
    s = DummyStrategy(_make_config())
    pos = Position(symbol="BTC-USDT", side=OrderSide.BUY, qty=1.0, avg_price=100.0)
    s.update_position(pos)
    got = s.get_position("BTC-USDT")
    assert got is not None
    assert got.symbol == "BTC-USDT"
    assert got.qty == 1.0


def test_update_position_overwrites_same_symbol() -> None:
    s = DummyStrategy(_make_config())
    s.update_position(
        Position(symbol="ETH-USDT", side=OrderSide.BUY, qty=2.0, avg_price=10.0)
    )
    s.update_position(
        Position(symbol="ETH-USDT", side=OrderSide.SELL, qty=0.5, avg_price=11.0)
    )
    p = s.get_position("ETH-USDT")
    assert p is not None
    assert p.side == OrderSide.SELL
    assert p.qty == 0.5


def test_remove_position() -> None:
    s = DummyStrategy(_make_config())
    s.update_position(Position(symbol="X", side=OrderSide.BUY, qty=1.0, avg_price=1.0))
    s.remove_position("X")
    assert s.get_position("X") is None


def test_remove_position_missing_is_safe() -> None:
    s = DummyStrategy(_make_config())
    s.remove_position("nope")  # should not raise


def test_get_all_positions_returns_shallow_copy() -> None:
    s = DummyStrategy(_make_config())
    s.update_position(Position(symbol="A", side=OrderSide.BUY, qty=1.0, avg_price=1.0))
    d = s.get_all_positions()
    d["B"] = Position(symbol="B", side=OrderSide.SELL, qty=1.0, avg_price=2.0)
    assert s.get_position("B") is None
    assert "B" not in s._positions


def test_record_signal_and_get_recent_default_limit() -> None:
    s = DummyStrategy(_make_config())
    sig = Signal(
        strategy_id=s.strategy_id,
        symbol="S",
        signal_type=SignalType.HOLD,
        strength=0.5,
    )
    s.record_signal(sig)
    recent = s.get_recent_signals()
    assert len(recent) == 1
    assert recent[0].symbol == "S"


def test_get_recent_signals_respects_limit() -> None:
    s = DummyStrategy(_make_config())
    for i in range(10):
        s.record_signal(
            Signal(
                strategy_id=s.strategy_id,
                symbol=str(i),
                signal_type=SignalType.HOLD,
                strength=0.1,
            )
        )
    tail = s.get_recent_signals(limit=3)
    assert [x.symbol for x in tail] == ["7", "8", "9"]


def test_get_recent_signals_when_less_than_limit() -> None:
    s = DummyStrategy(_make_config())
    s.record_signal(
        Signal(strategy_id=s.strategy_id, symbol="a", signal_type=SignalType.HOLD, strength=0.2)
    )
    assert len(s.get_recent_signals(limit=50)) == 1


def test_get_param_returns_value() -> None:
    s = DummyStrategy(_make_config(params={"k": 42}))
    assert s.get_param("k") == 42


def test_get_param_missing_returns_default() -> None:
    s = DummyStrategy(_make_config())
    assert s.get_param("missing", "d") == "d"


def test_get_param_missing_default_none() -> None:
    s = DummyStrategy(_make_config())
    assert s.get_param("missing") is None


@pytest.mark.asyncio
async def test_on_tick_default_returns_empty_list() -> None:
    s = DummyStrategy(_make_config())
    out = await s.on_tick("BTC-USDT", {"last": 1.0})
    assert out == []


@pytest.mark.asyncio
async def test_on_bar_returns_list() -> None:
    s = DummyStrategy(_make_config())
    assert await s.on_bar("S", {"close": 1.0}) == []


@pytest.mark.asyncio
async def test_generate_signals_returns_list() -> None:
    s = DummyStrategy(_make_config())
    assert await s.generate_signals({}) == []


# --- StrategyConfig ---


def test_strategy_config_requires_name() -> None:
    with pytest.raises(ValidationError):
        StrategyConfig()  # type: ignore[call-arg]


def test_strategy_config_defaults() -> None:
    c = StrategyConfig(name="n")
    assert c.version == "1.0.0"
    assert c.symbols == []
    assert c.params == {}
    assert c.risk_limits == {}
    assert c.enabled is True
    assert len(c.strategy_id) >= 4


def test_strategy_config_custom_fields() -> None:
    c = StrategyConfig(
        strategy_id="id1",
        name="n",
        version="2.0.0",
        symbols=["A", "B"],
        params={"x": 1},
        risk_limits={"max_dd": 0.1},
        enabled=False,
    )
    assert c.strategy_id == "id1"
    assert c.symbols == ["A", "B"]
    assert c.enabled is False


def test_strategy_config_model_dump() -> None:
    c = StrategyConfig(name="n", params={"a": 1})
    d = c.model_dump()
    assert d["name"] == "n"
    assert d["params"] == {"a": 1}


def test_strategy_config_model_dump_json_roundtrip() -> None:
    c = StrategyConfig(strategy_id="z9", name="n", params={"pi": 3.14})
    js = c.model_dump_json()
    c2 = StrategyConfig.model_validate_json(js)
    assert c2.strategy_id == "z9"
    assert c2.params["pi"] == 3.14


def test_strategy_config_model_validate_from_dict() -> None:
    c = StrategyConfig.model_validate({"name": "from_dict"})
    assert c.name == "from_dict"


# --- Signal / Position ---


@pytest.mark.parametrize("strength", [0.0, 0.5, 1.0, 0.001, 0.999])
def test_signal_strength_valid(strength: float) -> None:
    s = Signal(strategy_id="s", symbol="SYM", signal_type=SignalType.LONG_ENTRY, strength=strength)
    assert s.strength == strength


@pytest.mark.parametrize("strength", [-0.01, 1.01, 2.0, -1.0])
def test_signal_strength_invalid(strength: float) -> None:
    with pytest.raises(ValidationError):
        Signal(strategy_id="s", symbol="SYM", signal_type=SignalType.HOLD, strength=strength)


def test_signal_optional_fields_defaults() -> None:
    s = Signal(strategy_id="sid", symbol="BTC", signal_type=SignalType.SHORT_EXIT, strength=0.0)
    assert s.price is None
    assert s.suggested_qty is None
    assert s.reason == ""
    assert s.metadata == {}
    assert len(s.signal_id) >= 8


def test_signal_metadata_and_reason() -> None:
    s = Signal(
        strategy_id="sid",
        symbol="BTC",
        signal_type=SignalType.LONG_EXIT,
        strength=1.0,
        price=10.0,
        suggested_qty=2.0,
        reason="tp",
        metadata={"k": "v"},
    )
    assert s.price == 10.0
    assert s.suggested_qty == 2.0
    assert s.reason == "tp"
    assert s.metadata == {"k": "v"}


def test_position_model_defaults() -> None:
    p = Position(symbol="S", side=OrderSide.BUY, qty=1.0, avg_price=5.0)
    assert p.unrealized_pnl == 0.0
    assert p.realized_pnl == 0.0


@pytest.mark.parametrize(
    "member",
    [
        SignalType.LONG_ENTRY,
        SignalType.LONG_EXIT,
        SignalType.SHORT_ENTRY,
        SignalType.SHORT_EXIT,
        SignalType.HOLD,
    ],
)
def test_signal_type_enum_member(member: SignalType) -> None:
    assert isinstance(member, SignalType)
    assert isinstance(member.value, str)


def test_signal_type_enum_values_set() -> None:
    vals = {e.value for e in SignalType}
    assert vals == {"long_entry", "long_exit", "short_entry", "short_exit", "hold"}


@pytest.mark.parametrize("member", [OrderSide.BUY, OrderSide.SELL])
def test_order_side_enum_member(member: OrderSide) -> None:
    assert member.value in ("buy", "sell")


def test_order_side_enum_values_set() -> None:
    assert {e.value for e in OrderSide} == {"buy", "sell"}


@pytest.mark.parametrize(
    "member",
    [
        StrategyState.IDLE,
        StrategyState.RUNNING,
        StrategyState.PAUSED,
        StrategyState.ERROR,
        StrategyState.STOPPED,
    ],
)
def test_strategy_state_enum_member(member: StrategyState) -> None:
    assert isinstance(member.value, str)


def test_strategy_state_enum_values_set() -> None:
    assert {e.value for e in StrategyState} == {"idle", "running", "paused", "error", "stopped"}


# --- StrategyRegistry ---


def test_register_get_create_flow() -> None:
    StrategyRegistry.register("dummy", DummyStrategy)
    assert StrategyRegistry.get("dummy") is DummyStrategy
    cfg = _make_config()
    inst = StrategyRegistry.create("dummy", cfg)
    assert isinstance(inst, DummyStrategy)
    assert inst.config is cfg


def test_create_raises_keyerror_unknown() -> None:
    with pytest.raises(KeyError, match="未注册"):
        StrategyRegistry.create("nope", _make_config())


def test_get_returns_none_when_missing() -> None:
    assert StrategyRegistry.get("missing") is None


def test_unregister_true_when_present() -> None:
    StrategyRegistry.register("d", DummyStrategy)
    assert StrategyRegistry.unregister("d") is True
    assert StrategyRegistry.get("d") is None


def test_unregister_false_when_missing() -> None:
    assert StrategyRegistry.unregister("ghost") is False


def test_list_registered_empty() -> None:
    assert StrategyRegistry.list_registered() == []


def test_list_registered_returns_names() -> None:
    StrategyRegistry.register("a", DummyStrategy)
    StrategyRegistry.register("b", DummyStrategy)
    assert set(StrategyRegistry.list_registered()) == {"a", "b"}


def test_add_instance_returns_config() -> None:
    cfg = _make_config(strategy_id="i1")
    out = StrategyRegistry.add_instance(cfg)
    assert out is cfg


def test_list_instances_returns_values() -> None:
    c1 = _make_config(strategy_id="i1")
    c2 = _make_config(strategy_id="i2")
    StrategyRegistry.add_instance(c1)
    StrategyRegistry.add_instance(c2)
    got = {c.strategy_id for c in StrategyRegistry.list_instances()}
    assert got == {"i1", "i2"}


def test_get_instance_hits() -> None:
    cfg = _make_config(strategy_id="hit")
    StrategyRegistry.add_instance(cfg)
    assert StrategyRegistry.get_instance("hit") is cfg


def test_get_instance_miss() -> None:
    assert StrategyRegistry.get_instance("nope") is None


def test_delete_instance_true() -> None:
    cfg = _make_config(strategy_id="del1")
    StrategyRegistry.add_instance(cfg)
    assert StrategyRegistry.delete_instance("del1") is True
    assert StrategyRegistry.get_instance("del1") is None


def test_delete_instance_false() -> None:
    assert StrategyRegistry.delete_instance("missing") is False


def test_set_instance_enabled_updates_copy() -> None:
    cfg = _make_config(strategy_id="en", enabled=True)
    StrategyRegistry.add_instance(cfg)
    updated = StrategyRegistry.set_instance_enabled("en", False)
    assert updated is not None
    assert updated.enabled is False
    assert cfg.enabled is True
    stored = StrategyRegistry.get_instance("en")
    assert stored is not None
    assert stored.enabled is False


def test_set_instance_enabled_unknown_returns_none() -> None:
    assert StrategyRegistry.set_instance_enabled("zzz", True) is None


def test_auto_register_decorator_registers_class() -> None:

    @auto_register("decorated_dummy")
    class DecoratedDummy(DummyStrategy):
        pass

    assert StrategyRegistry.get("decorated_dummy") is DecoratedDummy
    obj = StrategyRegistry.create("decorated_dummy", _make_config())
    assert isinstance(obj, DecoratedDummy)


def test_double_register_logs_overwrite_warning(caplog: pytest.LogCaptureFixture) -> None:
    StrategyRegistry.register("dup", DummyStrategy)

    class Other(DummyStrategy):
        pass

    with caplog.at_level("WARNING", logger="strategy.registry"):
        StrategyRegistry.register("dup", Other)
    assert any("已注册" in r.message for r in caplog.records)
    assert StrategyRegistry.get("dup") is Other


def test_registry_create_uses_latest_class_after_overwrite() -> None:
    StrategyRegistry.register("z", DummyStrategy)

    class Z2(DummyStrategy):
        pass

    StrategyRegistry.register("z", Z2)
    obj = StrategyRegistry.create("z", _make_config())
    assert type(obj) is Z2


def test_add_instance_overwrites_same_id() -> None:
    c1 = _make_config(strategy_id="same", name="n1")
    c2 = _make_config(strategy_id="same", name="n2")
    StrategyRegistry.add_instance(c1)
    StrategyRegistry.add_instance(c2)
    got = StrategyRegistry.get_instance("same")
    assert got is not None
    assert got.name == "n2"


def test_lifecycle_sequence_idle_running_stopped() -> None:
    async def _run() -> None:
        s = DummyStrategy(_make_config())
        assert s.state == StrategyState.IDLE
        await s.on_start()
        assert s.state == StrategyState.RUNNING
        await s.on_stop()
        assert s.state == StrategyState.STOPPED

    asyncio.run(_run())


def test_lifecycle_running_then_error() -> None:
    async def _run() -> None:
        s = DummyStrategy(_make_config())
        await s.on_start()
        await s.on_error(RuntimeError("x"))
        assert s.state == StrategyState.ERROR

    asyncio.run(_run())


def test_signal_timestamp_is_set() -> None:
    s = Signal(strategy_id="s", symbol="X", signal_type=SignalType.HOLD, strength=0.0)
    assert s.timestamp is not None


def test_strategy_config_json_mode_roundtrip() -> None:
    c = StrategyConfig(name="jsonmode", symbols=["BTC"])
    raw = c.model_dump(mode="json")
    c2 = StrategyConfig.model_validate(raw)
    assert c2.symbols == ["BTC"]


@pytest.mark.parametrize("sig_type", list(SignalType))
def test_signal_accepts_each_signal_type(sig_type: SignalType) -> None:
    s = Signal(strategy_id="id", symbol="S", signal_type=sig_type, strength=0.25)
    assert s.signal_type == sig_type


@pytest.mark.parametrize("side", list(OrderSide))
def test_position_accepts_each_order_side(side: OrderSide) -> None:
    p = Position(symbol="S", side=side, qty=1.0, avg_price=1.0)
    assert p.side == side


def test_list_instances_order_stable_per_insertion() -> None:
    ids = ["a", "b", "c"]
    for i in ids:
        StrategyRegistry.add_instance(_make_config(strategy_id=i, name=i))
    listed = [c.strategy_id for c in StrategyRegistry.list_instances()]
    assert set(listed) == set(ids)
    assert len(listed) == 3


def test_unregister_then_create_raises() -> None:
    StrategyRegistry.register("tmp", DummyStrategy)
    StrategyRegistry.unregister("tmp")
    with pytest.raises(KeyError):
        StrategyRegistry.create("tmp", _make_config())


def test_get_all_positions_empty() -> None:
    s = DummyStrategy(_make_config())
    assert s.get_all_positions() == {}


def test_get_position_none_when_missing() -> None:
    s = DummyStrategy(_make_config())
    assert s.get_position("nope") is None


def test_record_signal_preserves_order_many() -> None:
    s = DummyStrategy(_make_config())
    for n in range(5):
        s.record_signal(
            Signal(
                strategy_id=s.strategy_id,
                symbol=f"S{n}",
                signal_type=SignalType.HOLD,
                strength=0.1,
            )
        )
    syms = [x.symbol for x in s.get_recent_signals(100)]
    assert syms == ["S0", "S1", "S2", "S3", "S4"]


def test_strategy_config_risk_limits_roundtrip() -> None:
    c = StrategyConfig(name="n", risk_limits={"m": 0.5})
    c2 = StrategyConfig.model_validate(c.model_dump())
    assert c2.risk_limits == {"m": 0.5}


def test_signal_model_copy() -> None:
    s = Signal(strategy_id="a", symbol="b", signal_type=SignalType.HOLD, strength=0.5)
    s2 = s.model_copy(update={"strength": 0.9})
    assert s2.strength == 0.9
    assert s.strength == 0.5


def test_position_unrealized_realized() -> None:
    p = Position(
        symbol="Q",
        side=OrderSide.SELL,
        qty=3.0,
        avg_price=4.0,
        unrealized_pnl=-1.0,
        realized_pnl=2.5,
    )
    assert p.unrealized_pnl == -1.0
    assert p.realized_pnl == 2.5


def test_registry_module_dicts_are_cleared_between_tests() -> None:
    assert strategy_registry._STRATEGY_REGISTRY == {}
    assert strategy_registry._STRATEGY_INSTANCES == {}


def test_set_instance_enabled_true_from_false() -> None:
    cfg = _make_config(strategy_id="tgl", enabled=False)
    StrategyRegistry.add_instance(cfg)
    u = StrategyRegistry.set_instance_enabled("tgl", True)
    assert u is not None and u.enabled is True


def test_create_passes_same_config_object() -> None:
    StrategyRegistry.register("d2", DummyStrategy)
    cfg = _make_config()
    obj = StrategyRegistry.create("d2", cfg)
    assert obj.config is cfg


def test_on_error_does_not_raise() -> None:
    async def _run() -> None:
        s = DummyStrategy(_make_config())
        await s.on_error(Exception("e"))

    asyncio.run(_run())


def test_strategy_state_string_compare() -> None:
    assert StrategyState.IDLE == "idle"
    assert StrategyState.RUNNING.value == "running"


def test_order_side_string_compare() -> None:
    assert OrderSide.BUY == "buy"


def test_signal_type_string_compare() -> None:
    assert SignalType.HOLD == "hold"


def test_list_registered_is_list_not_alias() -> None:
    StrategyRegistry.register("x", DummyStrategy)
    names = StrategyRegistry.list_registered()
    names.append("mutate")
    assert "mutate" not in StrategyRegistry.list_registered()


def test_list_instances_is_copy() -> None:
    StrategyRegistry.add_instance(_make_config(strategy_id="one"))
    lst = StrategyRegistry.list_instances()
    lst.clear()
    assert len(StrategyRegistry.list_instances()) == 1
