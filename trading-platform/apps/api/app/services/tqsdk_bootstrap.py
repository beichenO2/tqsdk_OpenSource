"""TqSdk startup via gateway — trading-platform never uses D-class grant."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from broker_tqsdk.gateway_client import TqGatewayBrokerClient
    from broker_tqsdk.gateway_market_adapter import TqGatewayMarketAdapter
    from execution.service import ExecutionService

logger = logging.getLogger(__name__)


@dataclass
class TqSdkRuntime:
    broker_client: TqGatewayBrokerClient
    market_adapter: TqGatewayMarketAdapter
    execution_service: ExecutionService


async def init_tqsdk_runtime() -> TqSdkRuntime | None:
    """Connect to tqsdk-gateway (credentials live only in gateway process)."""
    from broker_tqsdk.gateway_client import TqGatewayBrokerClient
    from broker_tqsdk.gateway_market_adapter import TqGatewayMarketAdapter
    from execution.service import ExecutionService
    from execution.tqsdk_adapter import TqSdkBrokerAdapter

    gateway_url = os.getenv("TQSDK_GATEWAY_URL", "http://127.0.0.1:12891")
    broker_client = TqGatewayBrokerClient(base_url=gateway_url)
    adapter = TqSdkBrokerAdapter(broker_client)
    svc = ExecutionService(adapter)
    market_adapter = TqGatewayMarketAdapter(base_url=gateway_url)

    await broker_client.connect()
    await svc.start()
    logger.info("TqSdk runtime ready via gateway %s (no local D-class)", gateway_url)
    return TqSdkRuntime(
        broker_client=broker_client,
        market_adapter=market_adapter,
        execution_service=svc,
    )
