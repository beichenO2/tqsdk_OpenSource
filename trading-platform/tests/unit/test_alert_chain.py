"""TDD — alert chain: EventBus → WS / Feishu."""

from __future__ import annotations

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from core.enums.direction import Direction, Offset
from event_bus import EventBus
from execution.order_manager import OrderRequest
from execution.service import ExecutionService
from risk.futures_limits import TradingSessionLimit
from risk.limits import MaxOrderSizeLimit
from risk.monitor import AlertLevel, RiskAlert, RiskMonitor
from tests.conftest import MockBrokerAdapter

_CST = ZoneInfo("Asia/Shanghai")


class _RecordingBus(EventBus):
    def __init__(self) -> None:
        super().__init__()
        self.emitted: list[tuple[str, dict[str, Any]]] = []

    async def emit(self, event_type: str, data: dict[str, Any]) -> None:
        self.emitted.append((event_type, data))
        await super().emit(event_type, data)


def _open_req(symbol: str = "rb2505", volume: int = 50) -> OrderRequest:
    return OrderRequest(
        symbol=symbol,
        exchange="SHFE",
        direction=Direction.LONG,
        offset=Offset.OPEN,
        price=Decimal("3500"),
        volume=volume,
    )


@pytest.fixture
def recording_bus() -> _RecordingBus:
    bus = _RecordingBus()
    EventBus._instance = bus  # type: ignore[attr-defined]
    return bus


@pytest.fixture
def wired_service(recording_bus: _RecordingBus) -> ExecutionService:
    from app.services.tqsdk_bootstrap import wire_alert_chain

    svc = ExecutionService(MockBrokerAdapter())
    wire_alert_chain(svc, recording_bus)
    return svc


def test_live_ws_event_types_include_order_channels() -> None:
    from app.routers.ws import LIVE_WS_EVENT_TYPES

    for channel in (
        "order_rejected",
        "order_cancelled",
        "order_partially_filled",
        "risk_alert",
        "trade_fill",
    ):
        assert channel in LIVE_WS_EVENT_TYPES


@pytest.mark.asyncio
async def test_risk_gate_reject_publishes_risk_alert(
    wired_service: ExecutionService,
    recording_bus: _RecordingBus,
) -> None:
    gate = wired_service.risk_gate
    gate.engine._limits.clear()  # noqa: SLF001
    gate.engine.add_limit(MaxOrderSizeLimit(max_volume=5))
    gate.engine.add_limit(
        TradingSessionLimit(clock=lambda: datetime(2025, 5, 8, 10, 0, tzinfo=_CST))
    )

    verdict = gate.check(_open_req(volume=50))
    assert verdict.allowed is False

    await asyncio.sleep(0.05)

    alerts = [(t, d) for t, d in recording_bus.emitted if t == "risk_alert"]
    assert len(alerts) == 1
    _, data = alerts[0]
    assert data["symbol"] == "rb2505"
    assert data["limit"] == "MaxOrderSize"
    assert "reason" in data
    assert data["source"] == "RiskGate"


@pytest.mark.asyncio
async def test_risk_monitor_alert_publishes_risk_alert(
    wired_service: ExecutionService,
    recording_bus: _RecordingBus,
) -> None:
    wired_service.risk_engine.update_account(
        balance=Decimal("100000"),
        available=Decimal("20000"),
        margin_ratio=Decimal("0.85"),
    )

    monitor = wired_service.risk_monitor
    monitor._check_margin()  # noqa: SLF001

    await asyncio.sleep(0.05)

    alerts = [(t, d) for t, d in recording_bus.emitted if t == "risk_alert"]
    assert len(alerts) == 1
    _, data = alerts[0]
    assert data["source"] == "Margin"
    assert "CRITICAL" in data["message"] or "critical" in data.get("level", "").lower()
    assert "reason" in data or "message" in data


@pytest.mark.asyncio
async def test_feishu_notifier_disabled_without_webhook(
    recording_bus: _RecordingBus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FEISHU_WEBHOOK_URL", raising=False)
    from notify import FeishuNotifier

    notifier = FeishuNotifier()
    notifier.attach(recording_bus)

    await recording_bus.emit(
        "risk_alert",
        {"symbol": "rb2505", "limit": "MaxOrderSize", "reason": "too big"},
    )
    assert notifier.enabled is False


@pytest.mark.asyncio
async def test_feishu_notifier_sends_on_risk_alert(
    recording_bus: _RecordingBus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEISHU_WEBHOOK_URL", "https://feishu.test/hook")
    from notify import FeishuNotifier

    mock_response = MagicMock()
    mock_response.json.return_value = {"code": 0}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("notify.httpx.AsyncClient", return_value=mock_client):
        notifier = FeishuNotifier()
        notifier.attach(recording_bus)

        await recording_bus.emit(
            "risk_alert",
            {
                "symbol": "rb2505",
                "limit": "MaxOrderSize",
                "reason": "[MaxOrderSize] volume exceeds max",
                "source": "RiskGate",
            },
        )
        await asyncio.sleep(0.05)

    mock_client.post.assert_awaited_once()
    call_kwargs = mock_client.post.await_args
    assert call_kwargs.args[0] == "https://feishu.test/hook"
    payload = call_kwargs.kwargs.get("json") or call_kwargs.args[1]
    text = payload["content"]["text"]
    assert "rb2505" in text
    assert "MaxOrderSize" in text or "risk_alert" in text.lower()


@pytest.mark.asyncio
async def test_feishu_notifier_throttles_duplicate_within_60s(
    recording_bus: _RecordingBus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEISHU_WEBHOOK_URL", "https://feishu.test/hook")
    from notify import FeishuNotifier

    mock_response = MagicMock()
    mock_response.json.return_value = {"code": 0}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    event_data = {
        "symbol": "rb2505",
        "limit": "MaxOrderSize",
        "reason": "too big",
        "source": "RiskGate",
    }

    with patch("notify.httpx.AsyncClient", return_value=mock_client):
        notifier = FeishuNotifier(throttle_seconds=60)
        notifier.attach(recording_bus)

        await recording_bus.emit("risk_alert", event_data)
        await recording_bus.emit("risk_alert", event_data)
        await asyncio.sleep(0.05)

    assert mock_client.post.await_count == 1


@pytest.mark.asyncio
async def test_feishu_notifier_swallows_httpx_errors(
    recording_bus: _RecordingBus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEISHU_WEBHOOK_URL", "https://feishu.test/hook")
    from notify import FeishuNotifier

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=RuntimeError("network down"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("notify.httpx.AsyncClient", return_value=mock_client):
        notifier = FeishuNotifier()
        notifier.attach(recording_bus)

        await recording_bus.emit(
            "risk_alert",
            {"symbol": "cu2509", "limit": "Margin", "reason": "margin high"},
        )
        await asyncio.sleep(0.05)

    mock_client.post.assert_awaited_once()
