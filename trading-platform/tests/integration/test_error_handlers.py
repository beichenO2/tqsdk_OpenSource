"""Integration-style tests for global exception handlers (mirrors ``app.main``)."""

from __future__ import annotations

import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
for p in (
    _repo,
    _repo / "apps" / "api",
    _repo / "packages" / "core",
    _repo / "packages" / "backtest",
    _repo / "packages" / "security" / "src",
    _repo / "packages",
):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from core.exceptions import (
    AuthenticationError,
    BacktestError,
    BacktestUnavailableError,
    BrokerConnectionError,
    BrokerError,
    ConfigError,
    DataError,
    DataNotAvailableError,
    ExchangeError,
    ExchangeNotConnectedError,
    ExecutionError,
    InvalidBarsError,
    MLUnavailableError,
    ModelNotFoundError,
    MLError,
    OrderCancelFailedError,
    OrderNotFoundError,
    OrderRejectedError,
    PermissionDeniedError,
    ProvidersUnavailableError,
    RiskError,
    RiskLimitExceeded,
    ServiceNotReadyError,
    StrategyAlreadyExistsError,
    StrategyError,
    StrategyNotFoundError,
    StrategyParamError,
    TradingPlatformError,
    ValidationError,
)
from tests.integration.route_harness import register_platform_exception_handlers


def _client(app: FastAPI) -> TestClient:
    # Starlette TestClient re-raises unhandled server exceptions by default; the generic
    # ``Exception`` handler still returns a 500 JSON body, but the client propagates the
    # original ``RuntimeError`` unless this flag is disabled.
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def sandbox_app() -> FastAPI:
    app = FastAPI()
    register_platform_exception_handlers(app)

    @app.get("/tp/custom")
    async def tp_custom():
        raise TradingPlatformError("custom", code="CUSTOM", status_code=418)

    @app.get("/tp/detail")
    async def tp_detail():
        raise TradingPlatformError("with detail", code="DET", status_code=400, detail={"a": 1})

    @app.get("/tp/base-defaults")
    async def tp_base_defaults():
        raise TradingPlatformError("base defaults")

    @app.get("/strategy-not-found")
    async def snf():
        raise StrategyNotFoundError("missing")

    @app.get("/strategy-exists")
    async def sae():
        raise StrategyAlreadyExistsError("dup")

    @app.get("/strategy-param")
    async def spe():
        raise StrategyParamError("bad param")

    @app.get("/strategy-base")
    async def sbe():
        raise StrategyError("generic strategy")

    @app.get("/backtest")
    async def be():
        raise BacktestError("bt fail")

    @app.get("/backtest-unavail")
    async def bu():
        raise BacktestUnavailableError("off")

    @app.get("/invalid-bars")
    async def ib():
        raise InvalidBarsError("bars")

    @app.get("/risk")
    async def re():
        raise RiskError("risk")

    @app.get("/risk-limit")
    async def rl():
        raise RiskLimitExceeded("cap")

    @app.get("/data-base")
    async def dbase():
        raise DataError("data")

    @app.get("/data-na")
    async def dna():
        raise DataNotAvailableError("na")

    @app.get("/providers")
    async def pu():
        raise ProvidersUnavailableError("providers")

    @app.get("/broker")
    async def br():
        raise BrokerError("br")

    @app.get("/broker-conn")
    async def bc():
        raise BrokerConnectionError("conn")

    @app.get("/exchange-nc")
    async def enc():
        raise ExchangeNotConnectedError("nc")

    @app.get("/exchange")
    async def ex():
        raise ExchangeError("ex")

    @app.get("/order-rej")
    async def orej():
        raise OrderRejectedError("rej")

    @app.get("/order-missing")
    async def onf():
        raise OrderNotFoundError("nf")

    @app.get("/order-cancel")
    async def ocf():
        raise OrderCancelFailedError("cancel")

    @app.get("/execution")
    async def ee():
        raise ExecutionError("exec")

    @app.get("/not-ready")
    async def snr():
        raise ServiceNotReadyError("wait")

    @app.get("/ml-unavail")
    async def mu():
        raise MLUnavailableError("ml off")

    @app.get("/model-missing")
    async def mm():
        raise ModelNotFoundError("no model")

    @app.get("/ml-base")
    async def mb():
        raise MLError("ml")

    @app.get("/config")
    async def ce():
        raise ConfigError("cfg")

    @app.get("/auth")
    async def ae():
        raise AuthenticationError("auth")

    @app.get("/perm")
    async def pd():
        raise PermissionDeniedError("denied")

    @app.get("/validation")
    async def ve():
        raise ValidationError("dates")

    @app.get("/value")
    async def val():
        raise ValueError("plain value error")

    @app.get("/boom")
    async def boom():
        raise RuntimeError("boom")

    return app


def test_trading_platform_error_custom_status_and_envelope(sandbox_app: FastAPI) -> None:
    r = _client(sandbox_app).get("/tp/custom")
    assert r.status_code == 418
    assert r.json() == {"error": "CUSTOM", "message": "custom"}


def test_trading_platform_error_includes_detail(sandbox_app: FastAPI) -> None:
    r = _client(sandbox_app).get("/tp/detail")
    assert r.status_code == 400
    body = r.json()
    assert body["error"] == "DET"
    assert body["message"] == "with detail"
    assert body["detail"] == {"a": 1}


def test_trading_platform_error_default_code_and_status(sandbox_app: FastAPI) -> None:
    r = _client(sandbox_app).get("/tp/base-defaults")
    assert r.status_code == 500
    body = r.json()
    assert body["error"] == "INTERNAL_ERROR"
    assert body["message"] == "base defaults"


def test_value_error_returns_422_envelope(sandbox_app: FastAPI) -> None:
    r = _client(sandbox_app).get("/value")
    assert r.status_code == 422
    assert r.json() == {"error": "VALIDATION_ERROR", "message": "plain value error"}


def test_generic_exception_returns_500(sandbox_app: FastAPI) -> None:
    r = _client(sandbox_app).get("/boom")
    assert r.status_code == 500
    assert r.json() == {"error": "INTERNAL_ERROR", "message": "Internal server error"}


def test_strategy_not_found_status(sandbox_app: FastAPI) -> None:
    r = _client(sandbox_app).get("/strategy-not-found")
    assert r.status_code == 404
    assert r.json()["error"] == "STRATEGY_NOT_FOUND"


def test_strategy_already_exists_status(sandbox_app: FastAPI) -> None:
    r = _client(sandbox_app).get("/strategy-exists")
    assert r.status_code == 409
    assert r.json()["error"] == "STRATEGY_ALREADY_EXISTS"


def test_strategy_param_invalid_status(sandbox_app: FastAPI) -> None:
    r = _client(sandbox_app).get("/strategy-param")
    assert r.status_code == 422
    assert r.json()["error"] == "STRATEGY_PARAM_INVALID"


def test_strategy_error_base_status(sandbox_app: FastAPI) -> None:
    r = _client(sandbox_app).get("/strategy-base")
    assert r.status_code == 400
    assert r.json()["error"] == "STRATEGY_ERROR"


def test_backtest_error_status(sandbox_app: FastAPI) -> None:
    r = _client(sandbox_app).get("/backtest")
    assert r.status_code == 400
    assert r.json()["error"] == "BACKTEST_ERROR"


def test_backtest_unavailable_status(sandbox_app: FastAPI) -> None:
    r = _client(sandbox_app).get("/backtest-unavail")
    assert r.status_code == 503
    assert r.json()["error"] == "BACKTEST_UNAVAILABLE"


def test_invalid_bars_status(sandbox_app: FastAPI) -> None:
    r = _client(sandbox_app).get("/invalid-bars")
    assert r.status_code == 422
    assert r.json()["error"] == "INVALID_BARS"


def test_risk_error_status(sandbox_app: FastAPI) -> None:
    r = _client(sandbox_app).get("/risk")
    assert r.status_code == 400
    assert r.json()["error"] == "RISK_ERROR"


def test_risk_limit_exceeded_status(sandbox_app: FastAPI) -> None:
    r = _client(sandbox_app).get("/risk-limit")
    assert r.status_code == 403
    assert r.json()["error"] == "RISK_LIMIT_EXCEEDED"


def test_data_error_base_status(sandbox_app: FastAPI) -> None:
    r = _client(sandbox_app).get("/data-base")
    assert r.status_code == 500
    assert r.json()["error"] == "DATA_ERROR"


def test_data_not_available_status(sandbox_app: FastAPI) -> None:
    r = _client(sandbox_app).get("/data-na")
    assert r.status_code == 404
    assert r.json()["error"] == "DATA_NOT_AVAILABLE"


def test_providers_unavailable_status(sandbox_app: FastAPI) -> None:
    r = _client(sandbox_app).get("/providers")
    assert r.status_code == 503
    assert r.json()["error"] == "PROVIDERS_UNAVAILABLE"


def test_broker_error_status(sandbox_app: FastAPI) -> None:
    r = _client(sandbox_app).get("/broker")
    assert r.status_code == 502
    assert r.json()["error"] == "BROKER_ERROR"


def test_broker_connection_error_status(sandbox_app: FastAPI) -> None:
    r = _client(sandbox_app).get("/broker-conn")
    assert r.status_code == 503
    assert r.json()["error"] == "BROKER_CONN_FAILED"


def test_exchange_not_connected_status(sandbox_app: FastAPI) -> None:
    r = _client(sandbox_app).get("/exchange-nc")
    assert r.status_code == 503
    assert r.json()["error"] == "EXCHANGE_NOT_CONNECTED"


def test_exchange_error_status(sandbox_app: FastAPI) -> None:
    r = _client(sandbox_app).get("/exchange")
    assert r.status_code == 502
    assert r.json()["error"] == "EXCHANGE_ERROR"


def test_order_rejected_status(sandbox_app: FastAPI) -> None:
    r = _client(sandbox_app).get("/order-rej")
    assert r.status_code == 400
    assert r.json()["error"] == "ORDER_REJECTED"


def test_order_not_found_status(sandbox_app: FastAPI) -> None:
    r = _client(sandbox_app).get("/order-missing")
    assert r.status_code == 404
    assert r.json()["error"] == "ORDER_NOT_FOUND"


def test_order_cancel_failed_status(sandbox_app: FastAPI) -> None:
    r = _client(sandbox_app).get("/order-cancel")
    assert r.status_code == 400
    assert r.json()["error"] == "ORDER_CANCEL_FAILED"


def test_execution_error_status(sandbox_app: FastAPI) -> None:
    r = _client(sandbox_app).get("/execution")
    assert r.status_code == 500
    assert r.json()["error"] == "EXECUTION_ERROR"


def test_service_not_ready_status(sandbox_app: FastAPI) -> None:
    r = _client(sandbox_app).get("/not-ready")
    assert r.status_code == 503
    assert r.json()["error"] == "SERVICE_NOT_READY"


def test_ml_unavailable_status(sandbox_app: FastAPI) -> None:
    r = _client(sandbox_app).get("/ml-unavail")
    assert r.status_code == 503
    assert r.json()["error"] == "ML_UNAVAILABLE"


def test_model_not_found_status(sandbox_app: FastAPI) -> None:
    r = _client(sandbox_app).get("/model-missing")
    assert r.status_code == 404
    assert r.json()["error"] == "MODEL_NOT_FOUND"


def test_ml_error_base_status(sandbox_app: FastAPI) -> None:
    r = _client(sandbox_app).get("/ml-base")
    assert r.status_code == 500
    assert r.json()["error"] == "ML_ERROR"


def test_config_error_status(sandbox_app: FastAPI) -> None:
    r = _client(sandbox_app).get("/config")
    assert r.status_code == 500
    assert r.json()["error"] == "CONFIG_ERROR"


def test_authentication_error_status(sandbox_app: FastAPI) -> None:
    r = _client(sandbox_app).get("/auth")
    assert r.status_code == 401
    assert r.json()["error"] == "AUTH_ERROR"


def test_permission_denied_status(sandbox_app: FastAPI) -> None:
    r = _client(sandbox_app).get("/perm")
    assert r.status_code == 403
    assert r.json()["error"] == "PERMISSION_DENIED"


def test_core_validation_error_status(sandbox_app: FastAPI) -> None:
    r = _client(sandbox_app).get("/validation")
    assert r.status_code == 422
    assert r.json()["error"] == "VALIDATION_ERROR"


def test_platform_handler_used_not_fastapi_default_for_domain_validation(
    sandbox_app: FastAPI,
) -> None:
    """Core ``ValidationError`` must hit ``TradingPlatformError`` handler (JSON envelope)."""
    r = _client(sandbox_app).get("/validation")
    body = r.json()
    assert body["error"] == "VALIDATION_ERROR"
    assert body["message"] == "dates"


def test_value_error_json_content_type(sandbox_app: FastAPI) -> None:
    r = _client(sandbox_app).get("/value")
    assert "application/json" in r.headers.get("content-type", "")


def test_trading_platform_error_json_content_type(sandbox_app: FastAPI) -> None:
    r = _client(sandbox_app).get("/strategy-not-found")
    assert "application/json" in r.headers.get("content-type", "")
