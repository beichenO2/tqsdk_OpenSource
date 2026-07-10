"""TqSdk startup via gateway — trading-platform never uses D-class grant."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from broker_tqsdk.gateway_client import TqGatewayBrokerClient
    from broker_tqsdk.gateway_market_adapter import TqGatewayMarketAdapter
    from event_bus import EventBus
    from execution.service import ExecutionService
    from notify import FeishuNotifier

logger = logging.getLogger(__name__)


@dataclass
class TqSdkRuntime:
    broker_client: TqGatewayBrokerClient
    market_adapter: TqGatewayMarketAdapter
    execution_service: ExecutionService
    feishu_notifier: FeishuNotifier | None = None


def wire_alert_chain(
    svc: ExecutionService,
    bus: EventBus | None = None,
) -> FeishuNotifier:
    """Connect RiskGate / RiskMonitor reject & alert callbacks to EventBus + Feishu."""
    from event_bus import EventBus as _EventBus
    from notify import FeishuNotifier

    bus = bus or _EventBus.get_instance()

    def _publish_risk_alert(payload: dict[str, Any]) -> None:
        data = {k: v for k, v in payload.items() if k != "type"}
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(bus.emit("risk_alert", data))
        except RuntimeError:
            asyncio.run(bus.emit("risk_alert", data))

    svc.risk_gate.on_reject(_publish_risk_alert)

    def _on_monitor_alert(alert: Any) -> None:
        data = {
            "source": alert.source,
            "level": str(alert.level),
            "message": alert.message,
            "reason": alert.message,
            "timestamp": alert.timestamp.isoformat(),
        }
        if alert.data:
            data.update(alert.data)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(bus.emit("risk_alert", data))
        except RuntimeError:
            asyncio.run(bus.emit("risk_alert", data))

    svc.risk_monitor.on_alert(_on_monitor_alert)

    notifier = FeishuNotifier()
    notifier.attach(bus)
    return notifier


async def init_tqsdk_runtime() -> TqSdkRuntime | None:
    """Connect to tqsdk-gateway (credentials live only in gateway process)."""
    from broker_tqsdk.gateway_client import TqGatewayBrokerClient
    from broker_tqsdk.gateway_market_adapter import TqGatewayMarketAdapter
    from execution.service import ExecutionService
    from execution.tqsdk_adapter import TqSdkBrokerAdapter

    gateway_url = os.getenv("TQSDK_GATEWAY_URL", "http://127.0.0.1:12890")
    broker_client = TqGatewayBrokerClient(base_url=gateway_url)
    adapter = TqSdkBrokerAdapter(broker_client)
    svc = ExecutionService(adapter)
    market_adapter = TqGatewayMarketAdapter(base_url=gateway_url)

    await broker_client.connect()
    notifier = wire_alert_chain(svc)
    await svc.start()
    logger.info("TqSdk runtime ready via gateway %s (no local D-class)", gateway_url)
    return TqSdkRuntime(
        broker_client=broker_client,
        market_adapter=market_adapter,
        execution_service=svc,
        feishu_notifier=notifier,
    )
