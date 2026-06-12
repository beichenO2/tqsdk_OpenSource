"""Shared FastAPI builders for route integration tests — no lifespan."""

from __future__ import annotations

import logging
from typing import Any, Iterable

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from core.exceptions import TradingPlatformError

logger = logging.getLogger("tests.route_harness")

_VALID_ROUTERS: frozenset[str] = frozenset(
    {
        "health",
        "orders",
        "positions",
        "strategies",
        "backtest",
        "market",
        "ml",
        "btc",
        "crypto_data",
        "explain",
    }
)

_DEFAULT_ORDER: tuple[str, ...] = (
    "health",
    "orders",
    "positions",
    "strategies",
    "backtest",
    "market",
    "ml",
    "btc",
    "crypto_data",
    "explain",
)


def register_platform_exception_handlers(app: FastAPI) -> None:
    """Mirror ``app.main.create_app`` exception handlers (without lifespan)."""
    log = logging.getLogger("tests.route_harness")

    @app.exception_handler(TradingPlatformError)
    async def platform_error_handler(request: object, exc: TradingPlatformError):
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    @app.exception_handler(ValueError)
    async def value_error_handler(request: object, exc: ValueError):
        return JSONResponse(
            status_code=422,
            content={"error": "VALIDATION_ERROR", "message": str(exc)},
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: object, exc: Exception):
        log.exception("Unhandled error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": "INTERNAL_ERROR", "message": "Internal server error"},
        )


def _include_router(app: FastAPI, name: str) -> None:
    """Import only the router module needed — avoids pulling optional deps (e.g. explain → core.db)."""
    if name == "health":
        from app.routers import health

        app.include_router(health.router)
        return
    if name == "orders":
        from app.routers import orders

        app.include_router(orders.router, prefix="/api/v1")
        return
    if name == "positions":
        from app.routers import positions

        app.include_router(positions.router, prefix="/api/v1")
        return
    if name == "strategies":
        from app.routers import strategies

        app.include_router(strategies.router, prefix="/api/v1")
        return
    if name == "backtest":
        from app.routers import backtest

        app.include_router(backtest.router, prefix="/api/v1")
        return
    if name == "market":
        from app.routers import market

        app.include_router(market.router, prefix="/api/v1")
        return
    if name == "ml":
        from app.routers import ml

        app.include_router(ml.router, prefix="/api/v1")
        return
    if name == "btc":
        from app.routers import btc

        app.include_router(btc.router, prefix="/api/v1/btc", tags=["btc"])
        return
    if name == "crypto_data":
        from app.routers import crypto_data

        app.include_router(crypto_data.router, prefix="/api/v1")
        return
    if name == "explain":
        from app.routers import explain

        app.include_router(explain.router, prefix="/api/v1")
        return
    raise ValueError(f"Unknown router name: {name!r}")


def build_test_app(
    *,
    routers: Iterable[str] | None = None,
    execution_service: Any | None = None,
    **kwargs: Any,
) -> FastAPI:
    """Assemble routers like ``create_app`` but without CORS/lifespan/TqSdk startup.

    This mirrors ``app.main.create_app`` routing and exception handlers only; it does **not**
    call ``create_app()`` (which attaches ``lifespan`` and bootstraps TqSdk).

    ``routers`` names: health, orders, positions, strategies, backtest, market, ml,
    btc, crypto_data, explain. ``None`` includes every router that imports successfully
    (optional routers such as ``explain`` are skipped on ``ImportError``).

    Extra keyword arguments are accepted for forward-compatible test overrides and ignored.
    """
    _ = kwargs  # reserved for future overrides (e.g. dependency injection hooks)

    from app.deps import get_execution_service

    if routers is None:
        selected = list(_DEFAULT_ORDER)
        explicit = False
    else:
        names = list(routers)
        unknown = [n for n in names if n not in _VALID_ROUTERS]
        if unknown:
            raise ValueError(f"Unknown router(s): {unknown}; valid: {sorted(_VALID_ROUTERS)}")
        selected = names
        explicit = True

    app = FastAPI()
    register_platform_exception_handlers(app)

    for name in selected:
        try:
            _include_router(app, name)
        except ImportError:
            if explicit:
                raise
            logger.warning("Skipping router %r (import failed)", name, exc_info=True)

    if execution_service is not None:
        app.dependency_overrides[get_execution_service] = lambda: execution_service

    return app
