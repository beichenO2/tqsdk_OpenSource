"""Optional API authentication dependency.

Enabled via ``TRADING_AUTH_ENABLED=true``.  When active, requests to
``/api/*`` must carry either:
  - ``Authorization: Bearer <jwt>``  header, or
  - ``X-API-Key: tpk_...``           header.

Public paths (/healthz, /readyz, /docs, /openapi.json) are always open.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

_AUTH_ENABLED: bool = os.getenv("TRADING_AUTH_ENABLED", "").lower() in ("true", "1", "yes")
_JWT_SECRET: str = os.getenv("TRADING_JWT_SECRET", "dev-secret-change-me")

_PUBLIC_PATHS: frozenset[str] = frozenset({
    "/healthz", "/readyz", "/docs", "/redoc", "/openapi.json",
})

_bearer_scheme = HTTPBearer(auto_error=False)
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

_jwt_service = None
_api_key_manager = None


def _get_jwt_service():
    global _jwt_service
    if _jwt_service is None:
        from security.jwt import JWTService
        _jwt_service = JWTService(_JWT_SECRET)
    return _jwt_service


def _get_api_key_manager():
    global _api_key_manager
    if _api_key_manager is None:
        from security.apikeys import APIKeyManager
        _api_key_manager = APIKeyManager()
    return _api_key_manager


async def require_auth(
    request: Request,
    bearer: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    api_key: Optional[str] = Depends(_api_key_header),
) -> dict:
    """FastAPI dependency that enforces authentication when enabled.

    Returns the authenticated identity dict (``{"sub": ..., "scopes": [...]}``).
    """
    if not _AUTH_ENABLED:
        return {"sub": "anonymous", "scopes": ["*"]}

    if request.url.path in _PUBLIC_PATHS:
        return {"sub": "public", "scopes": ["read"]}

    if bearer and bearer.credentials:
        try:
            payload = _get_jwt_service().verify_token(bearer.credentials)
            return {"sub": payload.get("sub", "unknown"), "scopes": payload.get("scopes", ["read"])}
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired JWT token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    if api_key:
        record = _get_api_key_manager().verify(api_key)
        if record is not None:
            return {"sub": record.owner, "scopes": record.scopes}
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required (Bearer token or X-API-Key header)",
        headers={"WWW-Authenticate": "Bearer"},
    )


def is_auth_enabled() -> bool:
    return _AUTH_ENABLED
