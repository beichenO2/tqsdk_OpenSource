"""Unit tests for futures-specific RiskGate limits and live confirm helpers."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from core.enums.direction import Direction, Offset
from execution.order_manager import OrderRequest
from risk.futures_limits import (
    DeliveryMonthLimit,
    LimitUpDownLimit,
    TradingSessionLimit,
    parse_delivery_ym,
)
from risk.gate import RiskGate, live_trading_enabled, verify_live_confirm_token
from risk.limits import RiskContext

_CST = ZoneInfo("Asia/Shanghai")


def _req(
    symbol: str = "rb2505",
    volume: int = 1,
    price: Decimal = Decimal("3500"),
    offset: Offset = Offset.OPEN,
) -> OrderRequest:
    return OrderRequest(
        symbol=symbol,
        exchange="SHFE",
        direction=Direction.LONG,
        offset=offset,
        price=price,
        volume=volume,
    )


def test_parse_delivery_ym() -> None:
    assert parse_delivery_ym("rb2505") == (2025, 5)
    assert parse_delivery_ym("SHFE.rb2505") == (2025, 5)
    assert parse_delivery_ym("DCE.i2509") == (2025, 9)
    assert parse_delivery_ym("BTCUSDT") is None


def test_delivery_month_blocks_open_in_delivery_month() -> None:
    limit = DeliveryMonthLimit(clock=lambda: datetime(2025, 5, 10, 10, 0, tzinfo=_CST))
    ctx = RiskContext(positions=[])
    ok, reason = limit.check(_req(symbol="rb2505"), ctx)
    assert ok is False
    assert "delivery month" in reason


def test_delivery_month_allows_close_in_delivery_month() -> None:
    limit = DeliveryMonthLimit(clock=lambda: datetime(2025, 5, 10, 10, 0, tzinfo=_CST))
    ctx = RiskContext(positions=[])
    ok, _ = limit.check(_req(symbol="rb2505", offset=Offset.CLOSE), ctx)
    assert ok is True


def test_delivery_month_allows_open_outside_delivery_month() -> None:
    limit = DeliveryMonthLimit(clock=lambda: datetime(2025, 4, 10, 10, 0, tzinfo=_CST))
    ctx = RiskContext(positions=[])
    ok, _ = limit.check(_req(symbol="rb2505"), ctx)
    assert ok is True


def test_limit_up_down_rejects_far_price() -> None:
    limit = LimitUpDownLimit(band_pct=Decimal("0.10"))
    ctx = RiskContext(positions=[], last_prices={"rb2505": Decimal("100")})
    ok, reason = limit.check(_req(price=Decimal("120")), ctx)
    assert ok is False
    assert "limit-up/down" in reason


def test_limit_up_down_accepts_near_price() -> None:
    limit = LimitUpDownLimit(band_pct=Decimal("0.10"))
    ctx = RiskContext(positions=[], last_prices={"rb2505": Decimal("100")})
    ok, _ = limit.check(_req(price=Decimal("105")), ctx)
    assert ok is True


def test_trading_session_rejects_weekend() -> None:
    # 2025-05-10 is Saturday
    limit = TradingSessionLimit(clock=lambda: datetime(2025, 5, 10, 10, 0, tzinfo=_CST))
    ctx = RiskContext(positions=[])
    ok, reason = limit.check(_req(), ctx)
    assert ok is False
    assert "weekend" in reason.lower()


def test_trading_session_accepts_day_session() -> None:
    # 2025-05-08 is Thursday 10:00
    limit = TradingSessionLimit(clock=lambda: datetime(2025, 5, 8, 10, 0, tzinfo=_CST))
    ctx = RiskContext(positions=[])
    ok, _ = limit.check(_req(), ctx)
    assert ok is True


def test_trading_session_rejects_off_hours() -> None:
    limit = TradingSessionLimit(clock=lambda: datetime(2025, 5, 8, 12, 0, tzinfo=_CST))
    ctx = RiskContext(positions=[])
    ok, reason = limit.check(_req(), ctx)
    assert ok is False
    assert "Outside trading session" in reason


def test_risk_gate_emits_reject_callback(monkeypatch) -> None:
    # Freeze session to daytime so only MaxOrderSize fires
    from risk.futures_limits import TradingSessionLimit as TSL
    from risk.limits import MaxOrderSizeLimit

    events: list[dict] = []
    gate = RiskGate(enable_futures_limits=False, on_reject=events.append)
    gate.engine.add_limit(MaxOrderSizeLimit(max_volume=5))
    gate.engine.add_limit(TSL(clock=lambda: datetime(2025, 5, 8, 10, 0, tzinfo=_CST)))

    verdict = gate.check(_req(volume=50))
    assert verdict.allowed is False
    assert verdict.limit_name == "MaxOrderSize"
    assert len(events) == 1
    assert events[0]["type"] == "risk_alert"
    assert events[0]["limit"] == "MaxOrderSize"


def test_live_trading_enabled_default_false(monkeypatch) -> None:
    monkeypatch.delenv("LIVE_TRADING_ENABLED", raising=False)
    assert live_trading_enabled() is False
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    assert live_trading_enabled() is True


def test_verify_live_confirm_token(monkeypatch) -> None:
    monkeypatch.delenv("LIVE_CONFIRM_TOKEN", raising=False)
    assert verify_live_confirm_token(None) is False
    assert verify_live_confirm_token("") is False
    assert verify_live_confirm_token("I_UNDERSTAND_LIVE_RISK") is True
    monkeypatch.setenv("LIVE_CONFIRM_TOKEN", "secret-token")
    assert verify_live_confirm_token("I_UNDERSTAND_LIVE_RISK") is False
    assert verify_live_confirm_token("secret-token") is True
