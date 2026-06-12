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
from app.routers import backtest, btc, crypto_data, deploy, health, live_trading, market, ml, orders, paper_trading, positions, strategies, ws

try:
    from app.routers import explain
except (ImportError, ModuleNotFoundError):
    explain = None  # type: ignore[assignment]

try:
    from app.routers import research
except (ImportError, ModuleNotFoundError):
    research = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _get_privportal_client():
    """Create a PolarPrivate client for credential retrieval."""
    try:
        from security.privportal import PrivPortalClient
        client = PrivPortalClient(service_name="tqsdk-api")
        client.ensure_session()
        return client
    except Exception:
        logger.warning("PolarPrivate unavailable, falling back to env vars")
        return None


async def _init_btc_broker() -> object | None:
    """Initialize BTCBrokerManager. Reads credentials from PolarPrivate (preferred)
    or environment variables (fallback). Always registers Binance in public-only
    mode for market data."""
    import os
    try:
        from broker_crypto import BTCBrokerManager, ExchangeCredentials, Exchange

        manager = BTCBrokerManager()
        is_testnet = os.getenv("CRYPTO_TESTNET", "false").lower() == "true"

        pp = _get_privportal_client()
        authenticated = 0

        _EXCHANGE_NAMES = {
            Exchange.BINANCE: "binance",
            Exchange.OKX: "okx",
            Exchange.WEEX: "weex",
        }

        for exchange, name in _EXCHANGE_NAMES.items():
            api_key = api_secret = passphrase = None
            if pp:
                try:
                    keys = pp.get_exchange_keys(name)
                    api_key = keys.api_key
                    api_secret = keys.api_secret
                    passphrase = keys.passphrase
                    if keys.testnet:
                        is_testnet = True
                    logger.info("BTCBrokerManager: %s credentials from PolarPrivate", name)
                except (KeyError, Exception) as exc:
                    logger.debug("PolarPrivate: no %s credentials (%s)", name, exc)

            if not api_key:
                env_prefix = name.upper()
                api_key = os.getenv(f"{env_prefix}_API_KEY")
                api_secret = os.getenv(f"{env_prefix}_API_SECRET")
                passphrase = os.getenv(f"{env_prefix}_PASSPHRASE")

            if api_key and api_secret:
                creds = ExchangeCredentials(
                    exchange=exchange,
                    api_key=api_key,
                    api_secret=api_secret,
                    passphrase=passphrase,
                    testnet=is_testnet,
                )
                await manager.add_exchange(creds)
                authenticated += 1

        if pp:
            pp.close()

        if Exchange.BINANCE not in manager.exchanges:
            public_creds = ExchangeCredentials(
                exchange=Exchange.BINANCE,
                api_key="",
                api_secret="",
                testnet=is_testnet,
            )
            await manager.add_exchange(public_creds)
            logger.info("BTCBrokerManager: Binance connected (public market data only)")

        if authenticated > 0:
            logger.info("BTCBrokerManager: %d exchange(s) with trading credentials", authenticated)
        else:
            logger.info("BTCBrokerManager: public market data mode (no trading credentials)")

        return manager
    except Exception:
        logger.warning("BTCBrokerManager init failed, BTC routes will return 503", exc_info=True)
        return None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging(level="INFO")

    svc = None
    broker_client = None

    try:
        from broker_tqsdk.client import TqBrokerClient
        from execution.tqsdk_adapter import TqSdkBrokerAdapter
        from execution.service import ExecutionService
        from app.services.market import create_market_adapter

        tq_kwargs: dict[str, str] = {}
        pp = _get_privportal_client()
        if pp:
            try:
                from security.privportal import TqSdkKeys  # noqa: F401
                keys = pp.get_tqsdk_keys()
                tq_kwargs = {
                    "auth_email": keys.auth_user,
                    "auth_password": keys.auth_password,
                    "broker_id": keys.broker,
                    "account_id": keys.account,
                }
                logger.info("TqSdk credentials from PolarPrivate")
            except (KeyError, Exception) as exc:
                logger.warning("PolarPrivate: no TqSdk credentials (%s)", exc)
            finally:
                pp.close()

        broker_client = TqBrokerClient(**tq_kwargs)
        adapter = TqSdkBrokerAdapter(broker_client)
        svc = ExecutionService(adapter)
        market_adapter = create_market_adapter()

        await svc.start()
        market_adapter.set_api(broker_client.tqsdk_api)
        set_execution_service(svc)
        set_market_adapter(market_adapter)
        logger.info("TqTrader 启动完成")
    except Exception:
        logger.warning("TqSdk 初始化失败 — 研究模式启动（交易路由返回 503）", exc_info=True)

    btc_manager = await _init_btc_broker()
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
        if svc is not None:
            await svc.stop()
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
    app.include_router(backtest.router, prefix="/api/v1", dependencies=auth_deps)
    app.include_router(market.router, prefix="/api/v1", dependencies=auth_deps)
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
    app.include_router(ws.router)

    return app


app = create_app()
