"""健康检查路由."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from app.deps import is_execution_service_ready
from core.exceptions import ServiceNotReadyError

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
