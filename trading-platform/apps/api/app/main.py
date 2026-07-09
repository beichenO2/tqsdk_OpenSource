"""FastAPI 应用入口."""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


def _bootstrap_repo_import_paths() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    extra_paths = (
        repo_root,
        repo_root / "apps" / "api",
        repo_root / "packages" / "core",
        repo_root / "packages" / "backtest",
        repo_root / "packages" / "broker_tqsdk",
        repo_root / "packages" / "broker_crypto" / "src",
        repo_root / "packages" / "risk",
        repo_root / "packages" / "factor",
        repo_root / "packages" / "features",
        repo_root / "packages" / "sim_live",
        repo_root / "packages" / "security" / "src",
        repo_root / "packages",
    )
    for path in reversed(extra_paths):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_bootstrap_repo_import_paths()

from core.exceptions import TradingPlatformError
from core.logging_config import setup_logging
from app.deps import set_btc_broker_manager, set_execution_service, set_market_adapter
from app.services.polarprivate_bootstrap import init_btc_broker
from app.services.tqsdk_bootstrap import init_tqsdk_runtime
from app.routers import btc, crypto_data, deploy, factors, health, live_trading, market, optimizer, orders, paper_trading, platform, positions, strategies, ws

try:
    from app.routers import mcp
except (ImportError, ModuleNotFoundError):
    mcp = None  # type: ignore[assignment]

try:
    from app.routers import settings
except (ImportError, ModuleNotFoundError):
    settings = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_OPTIONAL_ROUTERS = ("backtest", "ml", "explain", "research")
_optional: dict[str, object] = {}
for _name in _OPTIONAL_ROUTERS:
    try:
        _optional[_name] = __import__(f"app.routers.{_name}", fromlist=[_name])
    except (ImportError, ModuleNotFoundError, AttributeError):
        _optional[_name] = None
        logger.debug("Optional router %s not available", _name)

backtest = _optional["backtest"]
ml = _optional["ml"]
explain = _optional["explain"]
research = _optional["research"]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging(level="INFO")

    tqsdk_runtime = None

    try:
        tqsdk_runtime = await init_tqsdk_runtime()
        set_execution_service(tqsdk_runtime.execution_service)
        set_market_adapter(tqsdk_runtime.market_adapter)
        logger.info("TqTrader 启动完成")
    except Exception:
        logger.warning("TqSdk 初始化失败 — 研究模式启动（交易路由返回 503）", exc_info=True)

    btc_manager = await init_btc_broker()
    set_btc_broker_manager(btc_manager)

    try:
        yield
    finally:
        if btc_manager is not None:
            try:
                await btc_manager.disconnect_all()
            except Exception:
                logger.warning("BTCBrokerManager disconnect error", exc_info=True)
        set_btc_broker_manager(None)
        if tqsdk_runtime is not None:
            await tqsdk_runtime.broker_client.disconnect()
            await tqsdk_runtime.market_adapter.aclose()
            await tqsdk_runtime.execution_service.stop()
        set_market_adapter(None)
        logger.info("TqTrader 关闭")


def create_app() -> FastAPI:
    app = FastAPI(
        title="TqTrader API",
        description="量化交易平台 — 期货 & 加密货币",
        version="0.2.0",
        lifespan=lifespan,
    )

    import os
    cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:5174").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in cors_origins],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    try:
        from security.middleware import add_sanitizing_logging_middleware
        add_sanitizing_logging_middleware(app)
    except ImportError:
        logger.debug("security.middleware not available, skipping sanitizing log middleware")

    # ── 统一异常处理 ──
    @app.exception_handler(TradingPlatformError)
    async def platform_error_handler(request, exc: TradingPlatformError):
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    @app.exception_handler(ValueError)
    async def value_error_handler(request, exc: ValueError):
        return JSONResponse(
            status_code=422,
            content={"error": "VALIDATION_ERROR", "message": str(exc)},
        )

    @app.exception_handler(ConnectionRefusedError)
    async def db_connection_handler(request, exc: ConnectionRefusedError):
        return JSONResponse(
            status_code=503,
            content={"error": "SERVICE_NOT_READY", "message": "Database connection unavailable"},
        )

    @app.exception_handler(OSError)
    async def os_error_handler(request, exc: OSError):
        if "connect" in str(exc).lower() or "refused" in str(exc).lower():
            return JSONResponse(
                status_code=503,
                content={"error": "SERVICE_NOT_READY", "message": "Backend service unavailable"},
            )
        logger.exception("OS error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": "INTERNAL_ERROR", "message": "Internal server error"},
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request, exc: Exception):
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": "INTERNAL_ERROR", "message": "Internal server error"},
        )

    # ── 认证依赖（env TRADING_AUTH_ENABLED=true 启用）──
    from app.middleware.auth import is_auth_enabled, require_auth
    auth_deps = [Depends(require_auth)] if is_auth_enabled() else []

    # ── 路由注册 ──
    app.include_router(health.router)
    app.include_router(orders.router, prefix="/api/v1", dependencies=auth_deps)
    app.include_router(positions.router, prefix="/api/v1", dependencies=auth_deps)
    app.include_router(strategies.router, prefix="/api/v1", dependencies=auth_deps)
    if backtest is not None:
        app.include_router(backtest.router, prefix="/api/v1", dependencies=auth_deps)
    app.include_router(market.router, prefix="/api/v1", dependencies=auth_deps)
    if ml is not None:
        app.include_router(ml.router, prefix="/api/v1", dependencies=auth_deps)
    app.include_router(btc.router, prefix="/api/v1/btc", tags=["btc"], dependencies=auth_deps)
    app.include_router(crypto_data.router, prefix="/api/v1", dependencies=auth_deps)
    if explain is not None:
        app.include_router(explain.router, prefix="/api/v1", dependencies=auth_deps)
    if research is not None:
        app.include_router(research.router, prefix="/api/v1", dependencies=auth_deps)
    app.include_router(paper_trading.router, prefix="/api/v1", dependencies=auth_deps)
    app.include_router(live_trading.router, prefix="/api/v1", dependencies=auth_deps)
    app.include_router(deploy.router, prefix="/api/v1", dependencies=auth_deps)
    app.include_router(optimizer.router, prefix="/api/v1", dependencies=auth_deps)
    app.include_router(platform.router, prefix="/api/v1", dependencies=auth_deps)
    app.include_router(factors.router, prefix="/api/v1", dependencies=auth_deps)
    app.include_router(ws.router)
    if mcp is not None:
        app.include_router(mcp.router, prefix="/api/v1", dependencies=auth_deps)
    if settings is not None:
        app.include_router(settings.router, prefix="/api/v1", dependencies=auth_deps)

    return app


app = create_app()
