"""Tests for VolatilityCircuitBreaker recovery behavior."""

import sys
from decimal import Decimal
from types import ModuleType
from unittest.mock import MagicMock


class _RiskLimitStub:
    """Minimal RiskLimit base class stub."""
    pass


class _OffsetStub:
    OPEN = "OPEN"


class _DirectionStub:
    LONG = "LONG"
    SHORT = "SHORT"


class _OrderTypeStub:
    MARKET = "MARKET"
    LIMIT = "LIMIT"


def _setup_stubs():
    """Create minimal module stubs so btc.risk_limits can import."""
    core_mod = ModuleType("core")
    core_enums = ModuleType("core.enums")
    core_dir = ModuleType("core.enums.direction")
    core_dir.Direction = _DirectionStub
    core_dir.Offset = _OffsetStub
    core_ot = ModuleType("core.enums.order_type")
    core_ot.OrderType = _OrderTypeStub
    core_os = ModuleType("core.enums.order_status")
    core_os.OrderStatus = MagicMock()
    core_models = ModuleType("core.models")
    core_order = ModuleType("core.models.order")
    core_order.Order = MagicMock()
    core_pos = ModuleType("core.models.position")
    core_pos.Position = MagicMock()
    exec_mod = ModuleType("execution")
    exec_om = ModuleType("execution.order_manager")
    exec_om.OrderRequest = MagicMock()
    exec_ba = ModuleType("execution.broker_adapter")
    exec_ba.BrokerAdapter = MagicMock()
    risk_mod = ModuleType("risk")
    risk_limits = ModuleType("risk.limits")
    risk_limits.RiskLimit = _RiskLimitStub
    risk_limits.RiskContext = MagicMock()

    for name, mod in [
        ("core", core_mod), ("core.enums", core_enums),
        ("core.enums.direction", core_dir), ("core.enums.order_type", core_ot),
        ("core.enums.order_status", core_os), ("core.models", core_models),
        ("core.models.order", core_order), ("core.models.position", core_pos),
        ("execution", exec_mod), ("execution.order_manager", exec_om),
        ("execution.broker_adapter", exec_ba),
        ("risk", risk_mod), ("risk.limits", risk_limits),
    ]:
        sys.modules[name] = mod


_setup_stubs()

from strategy.btc.risk_limits import VolatilityCircuitBreaker  # noqa: E402


def _make_open_request(symbol: str = "BTCUSDT"):
    req = MagicMock()
    req.symbol = symbol
    req.offset = _OffsetStub.OPEN
    return req


def _ctx():
    return MagicMock()


def test_breaker_trips_on_high_volatility():
    cb = VolatilityCircuitBreaker(
        max_volatility_pct=Decimal("0.02"),
        lookback_bars=5,
        cooldown_bars=3,
    )
    for p in [100, 110, 90, 120, 80, 130, 70]:
        cb.feed_price("BTCUSDT", Decimal(str(p)))

    req = _make_open_request()
    ok, msg = cb.check(req, _ctx())
    assert not ok, f"Should trip on high volatility, msg={msg}"


def test_breaker_allows_when_calm():
    cb = VolatilityCircuitBreaker(
        max_volatility_pct=Decimal("0.50"),
        lookback_bars=5,
        cooldown_bars=3,
    )
    for p in [100.0, 100.1, 100.2, 100.1, 100.0, 100.15, 100.05]:
        cb.feed_price("BTCUSDT", Decimal(str(p)))

    req = _make_open_request()
    ok, _ = cb.check(req, _ctx())
    assert ok, "Should allow trading when volatility is low"


def test_breaker_recovery_mechanism():
    """After tripping, breaker should recover once vol drops and cooldown elapses."""
    cb = VolatilityCircuitBreaker(
        max_volatility_pct=Decimal("0.05"),
        lookback_bars=5,
        cooldown_bars=2,
        recovery_pct=Decimal("0.03"),
    )

    for p in [100, 115, 85, 120, 80, 125]:
        cb.feed_price("BTCUSDT", Decimal(str(p)))

    req = _make_open_request()
    ok1, _ = cb.check(req, _ctx())
    assert not ok1, "Should trip initially"

    for p in [100.0, 100.1, 100.05, 100.02, 100.08, 100.03]:
        cb.feed_price("BTCUSDT", Decimal(str(p)))

    recovered = False
    for _ in range(5):
        ok, _ = cb.check(req, _ctx())
        if ok:
            recovered = True
            break

    assert recovered, "Should eventually recover after calm prices and cooldown"
