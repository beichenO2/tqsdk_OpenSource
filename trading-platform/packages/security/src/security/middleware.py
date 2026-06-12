"""FastAPI / Starlette middleware for sanitized HTTP logging and CORS setup."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from security.sanitizer import sanitize_log

logger = logging.getLogger("security.http")


class SanitizingLoggingMiddleware(BaseHTTPMiddleware):
    """Log basic request/response metadata with secrets redacted from URLs and headers."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        log_headers: bool = False,
        header_allowlist: Sequence[str] | None = None,
    ) -> None:
        super().__init__(app)
        self._log_headers = log_headers
        self._header_allowlist = {h.lower() for h in (header_allowlist or ("content-type", "user-agent"))}

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        safe_url = sanitize_log(str(request.url))
        extra = ""
        if self._log_headers:
            parts: list[str] = []
            for name, value in request.headers.items():
                if name.lower() not in self._header_allowlist:
                    continue
                parts.append(f"{name}={sanitize_log(value)}")
            if parts:
                extra = " " + " ".join(parts)
        logger.info("%s %s%s", request.method, safe_url, extra)
        response = await call_next(request)
        logger.debug("response %s status=%s", safe_url, response.status_code)
        return response


def cors_middleware_config(
    *,
    allow_origins: Sequence[str] | None = None,
    allow_credentials: bool = True,
    allow_methods: Sequence[str] | None = None,
    allow_headers: Sequence[str] | None = None,
    expose_headers: Sequence[str] | None = None,
    max_age: int = 600,
) -> dict:
    """Build kwargs suitable for ``CORSMiddleware`` (explicit origins; no wildcard with credentials)."""
    origins = list(allow_origins) if allow_origins is not None else []
    methods = list(allow_methods) if allow_methods is not None else ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
    headers = list(allow_headers) if allow_headers is not None else ["*"]
    exposed = list(expose_headers) if expose_headers is not None else []
    return {
        "allow_origins": origins,
        "allow_credentials": allow_credentials,
        "allow_methods": methods,
        "allow_headers": headers,
        "expose_headers": exposed,
        "max_age": max_age,
    }


def add_cors(
    app: FastAPI,
    *,
    allow_origins: Sequence[str] | None = None,
    allow_credentials: bool = True,
    allow_methods: Sequence[str] | None = None,
    allow_headers: Sequence[str] | None = None,
    expose_headers: Sequence[str] | None = None,
    max_age: int = 600,
) -> None:
    """Register ``CORSMiddleware`` on *app* with trading-friendly defaults."""
    cfg = cors_middleware_config(
        allow_origins=allow_origins,
        allow_credentials=allow_credentials,
        allow_methods=allow_methods,
        allow_headers=allow_headers,
        expose_headers=expose_headers,
        max_age=max_age,
    )
    app.add_middleware(CORSMiddleware, **cfg)


def add_sanitizing_logging_middleware(
    app: FastAPI,
    *,
    log_headers: bool = False,
    header_allowlist: Sequence[str] | None = None,
) -> None:
    """Append middleware that logs requests with :func:`security.sanitizer.sanitize_log` applied."""
    app.add_middleware(
        SanitizingLoggingMiddleware,
        log_headers=log_headers,
        header_allowlist=header_allowlist,
    )
