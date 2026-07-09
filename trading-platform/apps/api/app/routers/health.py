"""健康检查路由 — liveness / readiness / 系统组件聚合。"""

from __future__ import annotations

import logging
import os
from typing import Any
from urllib.request import urlopen, Request

from fastapi import APIRouter

from app.deps import is_execution_service_ready
from core.exceptions import ServiceNotReadyError
from risk.gate import live_trading_enabled

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz() -> dict[str, str]:
    if not is_execution_service_ready():
        logger.warning("readyz: ExecutionService not initialized")
        raise ServiceNotReadyError(
            "ExecutionService not initialized",
            detail={"execution": "unavailable"},
        )
    return {"status": "ready", "execution": "ok"}


def _probe_http(url: str, timeout: float = 2.0) -> dict[str, Any]:
    try:
        req = Request(url, method="GET")
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310 — local health only
            body = resp.read().decode("utf-8", errors="replace")[:500]
            return {"ok": 200 <= resp.status < 300, "status_code": resp.status, "body": body}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/api/v1/system/health")
async def system_health() -> dict[str, Any]:
    """聚合 API / Execution / RiskGate / TqSdk Gateway 健康。"""
    components: dict[str, Any] = {}

    components["api"] = {"ok": True, "status": "ok"}

    exec_ready = is_execution_service_ready()
    components["execution"] = {
        "ok": exec_ready,
        "status": "ready" if exec_ready else "unavailable",
    }

    risk: dict[str, Any] = {"ok": True, "live_enabled": live_trading_enabled()}
    if exec_ready:
        try:
            from app.deps import get_execution_service
            svc = get_execution_service()
            risk.update(svc.risk_gate.get_status())
        except Exception as e:
            risk["ok"] = False
            risk["error"] = str(e)
    else:
        risk["ok"] = False
        risk["status"] = "execution_unavailable"
    components["risk_gate"] = risk

    gateway_url = os.getenv("TQSDK_GATEWAY_URL", "http://127.0.0.1:12890").rstrip("/")
    gw = _probe_http(f"{gateway_url}/health")
    components["tqsdk_gateway"] = {
        "ok": gw.get("ok", False),
        "url": gateway_url,
        **{k: v for k, v in gw.items() if k != "ok"},
    }

    overall = all(c.get("ok") for c in components.values())
    return {
        "status": "ok" if overall else "degraded",
        "live_enabled": live_trading_enabled(),
        "components": components,
    }
