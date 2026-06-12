"""Unit tests for ``TradingPlatformError`` hierarchy (``core.exceptions``)."""

from __future__ import annotations

import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
for p in [_repo, _repo / "apps" / "api", _repo / "packages" / "core", _repo / "packages" / "security" / "src", _repo / "packages"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytest

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
    MLError,
    MLUnavailableError,
    ModelNotFoundError,
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


def test_trading_platform_error_default_code() -> None:
    err = TradingPlatformError()
    assert err.code == "INTERNAL_ERROR"


def test_trading_platform_error_default_status_code() -> None:
    err = TradingPlatformError()
    assert err.status_code == 500


def test_trading_platform_error_custom_code() -> None:
    err = TradingPlatformError(code="CUSTOM")
    assert err.code == "CUSTOM"


def test_trading_platform_error_custom_status_code() -> None:
    err = TradingPlatformError(status_code=418)
    assert err.status_code == 418


def test_trading_platform_error_message_str() -> None:
    err = TradingPlatformError("boom")
    assert str(err) == "boom"


def test_trading_platform_error_empty_message() -> None:
    err = TradingPlatformError("")
    assert str(err) == ""


def test_trading_platform_error_to_dict_basic() -> None:
    err = TradingPlatformError("hello")
    assert err.to_dict() == {"error": "INTERNAL_ERROR", "message": "hello"}


def test_trading_platform_error_to_dict_includes_detail_when_non_empty() -> None:
    err = TradingPlatformError("x", detail={"a": 1})
    assert err.to_dict() == {
        "error": "INTERNAL_ERROR",
        "message": "x",
        "detail": {"a": 1},
    }


def test_trading_platform_error_to_dict_omits_detail_when_empty() -> None:
    err = TradingPlatformError("x", detail={})
    d = err.to_dict()
    assert "detail" not in d


def test_trading_platform_error_to_dict_omits_detail_when_none_normalized() -> None:
    err = TradingPlatformError("x", detail=None)
    d = err.to_dict()
    assert "detail" not in d


def test_trading_platform_error_repr() -> None:
    err = TradingPlatformError("m", code="C")
    assert repr(err) == "TradingPlatformError(code='C', message='m')"


def test_trading_platform_error_detail_default_empty() -> None:
    err = TradingPlatformError()
    assert err.detail == {}


def test_trading_platform_error_custom_code_and_status_together() -> None:
    err = TradingPlatformError("z", code="Z", status_code=400)
    assert err.code == "Z"
    assert err.status_code == 400
    assert err.to_dict()["error"] == "Z"


def test_strategy_error_default_code() -> None:
    assert StrategyError().code == "STRATEGY_ERROR"


def test_strategy_error_default_status_code() -> None:
    assert StrategyError().status_code == 400


def test_strategy_not_found_error_default_code() -> None:
    assert StrategyNotFoundError("n").code == "STRATEGY_NOT_FOUND"


def test_strategy_not_found_error_default_status_code() -> None:
    assert StrategyNotFoundError("n").status_code == 404


def test_strategy_already_exists_error_default_code() -> None:
    assert StrategyAlreadyExistsError("e").code == "STRATEGY_ALREADY_EXISTS"


def test_strategy_already_exists_error_default_status_code() -> None:
    assert StrategyAlreadyExistsError("e").status_code == 409


def test_strategy_param_error_default_code() -> None:
    assert StrategyParamError("p").code == "STRATEGY_PARAM_INVALID"


def test_strategy_param_error_default_status_code() -> None:
    assert StrategyParamError("p").status_code == 422


def test_backtest_error_default_code() -> None:
    assert BacktestError().code == "BACKTEST_ERROR"


def test_backtest_error_default_status_code() -> None:
    assert BacktestError().status_code == 400


def test_backtest_unavailable_error_default_code() -> None:
    assert BacktestUnavailableError().code == "BACKTEST_UNAVAILABLE"


def test_backtest_unavailable_error_default_status_code() -> None:
    assert BacktestUnavailableError().status_code == 503


def test_invalid_bars_error_default_code() -> None:
    assert InvalidBarsError().code == "INVALID_BARS"


def test_invalid_bars_error_default_status_code() -> None:
    assert InvalidBarsError().status_code == 422


def test_risk_error_default_code() -> None:
    assert RiskError().code == "RISK_ERROR"


def test_risk_error_default_status_code() -> None:
    assert RiskError().status_code == 400


def test_risk_limit_exceeded_default_code() -> None:
    assert RiskLimitExceeded().code == "RISK_LIMIT_EXCEEDED"


def test_risk_limit_exceeded_default_status_code() -> None:
    assert RiskLimitExceeded().status_code == 403


def test_data_error_default_code() -> None:
    assert DataError().code == "DATA_ERROR"


def test_data_error_default_status_code() -> None:
    assert DataError().status_code == 500


def test_data_not_available_error_default_code() -> None:
    assert DataNotAvailableError().code == "DATA_NOT_AVAILABLE"


def test_data_not_available_error_default_status_code() -> None:
    assert DataNotAvailableError().status_code == 404


def test_providers_unavailable_error_default_code() -> None:
    assert ProvidersUnavailableError().code == "PROVIDERS_UNAVAILABLE"


def test_providers_unavailable_error_default_status_code() -> None:
    assert ProvidersUnavailableError().status_code == 503


def test_broker_error_default_code() -> None:
    assert BrokerError().code == "BROKER_ERROR"


def test_broker_error_default_status_code() -> None:
    assert BrokerError().status_code == 502


def test_broker_connection_error_default_code() -> None:
    assert BrokerConnectionError().code == "BROKER_CONN_FAILED"


def test_broker_connection_error_default_status_code() -> None:
    assert BrokerConnectionError().status_code == 503


def test_exchange_not_connected_error_default_code() -> None:
    assert ExchangeNotConnectedError().code == "EXCHANGE_NOT_CONNECTED"


def test_exchange_not_connected_error_default_status_code() -> None:
    assert ExchangeNotConnectedError().status_code == 503


def test_exchange_error_default_code() -> None:
    assert ExchangeError().code == "EXCHANGE_ERROR"


def test_exchange_error_default_status_code() -> None:
    assert ExchangeError().status_code == 502


def test_order_rejected_error_default_code() -> None:
    assert OrderRejectedError().code == "ORDER_REJECTED"


def test_order_rejected_error_default_status_code() -> None:
    assert OrderRejectedError().status_code == 400


def test_order_not_found_error_default_code() -> None:
    assert OrderNotFoundError().code == "ORDER_NOT_FOUND"


def test_order_not_found_error_default_status_code() -> None:
    assert OrderNotFoundError().status_code == 404


def test_order_cancel_failed_error_default_code() -> None:
    assert OrderCancelFailedError().code == "ORDER_CANCEL_FAILED"


def test_order_cancel_failed_error_default_status_code() -> None:
    assert OrderCancelFailedError().status_code == 400


def test_execution_error_default_code() -> None:
    assert ExecutionError().code == "EXECUTION_ERROR"


def test_execution_error_default_status_code() -> None:
    assert ExecutionError().status_code == 500


def test_service_not_ready_error_default_code() -> None:
    assert ServiceNotReadyError().code == "SERVICE_NOT_READY"


def test_service_not_ready_error_default_status_code() -> None:
    assert ServiceNotReadyError().status_code == 503


def test_ml_error_default_code() -> None:
    assert MLError().code == "ML_ERROR"


def test_ml_error_default_status_code() -> None:
    assert MLError().status_code == 500


def test_ml_unavailable_error_default_code() -> None:
    assert MLUnavailableError().code == "ML_UNAVAILABLE"


def test_ml_unavailable_error_default_status_code() -> None:
    assert MLUnavailableError().status_code == 503


def test_model_not_found_error_default_code() -> None:
    assert ModelNotFoundError().code == "MODEL_NOT_FOUND"


def test_model_not_found_error_default_status_code() -> None:
    assert ModelNotFoundError().status_code == 404


def test_config_error_default_code() -> None:
    assert ConfigError().code == "CONFIG_ERROR"


def test_config_error_default_status_code() -> None:
    assert ConfigError().status_code == 500


def test_authentication_error_default_code() -> None:
    assert AuthenticationError().code == "AUTH_ERROR"


def test_authentication_error_default_status_code() -> None:
    assert AuthenticationError().status_code == 401


def test_permission_denied_error_default_code() -> None:
    assert PermissionDeniedError().code == "PERMISSION_DENIED"


def test_permission_denied_error_default_status_code() -> None:
    assert PermissionDeniedError().status_code == 403


def test_validation_error_default_code() -> None:
    assert ValidationError().code == "VALIDATION_ERROR"


def test_validation_error_default_status_code() -> None:
    assert ValidationError().status_code == 422


def test_strategy_not_found_is_strategy_error() -> None:
    err = StrategyNotFoundError("x")
    assert isinstance(err, StrategyError)


def test_strategy_not_found_is_trading_platform_error() -> None:
    err = StrategyNotFoundError("x")
    assert isinstance(err, TradingPlatformError)


def test_order_not_found_is_broker_error() -> None:
    err = OrderNotFoundError()
    assert isinstance(err, BrokerError)


def test_order_not_found_is_trading_platform_error() -> None:
    err = OrderNotFoundError()
    assert isinstance(err, TradingPlatformError)


def test_invalid_bars_is_backtest_error() -> None:
    err = InvalidBarsError()
    assert isinstance(err, BacktestError)


def test_catch_strategy_not_found_as_strategy_error() -> None:
    with pytest.raises(StrategyError):
        raise StrategyNotFoundError("missing")


def test_catch_order_not_found_as_broker_error() -> None:
    with pytest.raises(BrokerError):
        raise OrderNotFoundError()


def test_catch_service_not_ready_as_trading_platform_error() -> None:
    with pytest.raises(TradingPlatformError):
        raise ServiceNotReadyError("no")


def test_catch_validation_error_as_trading_platform_error() -> None:
    with pytest.raises(TradingPlatformError):
        raise ValidationError("bad")


def test_strategy_error_custom_code_and_status_override() -> None:
    err = StrategyError("s", code="S_CUSTOM", status_code=499)
    assert err.code == "S_CUSTOM"
    assert err.status_code == 499


def test_broker_error_custom_code_override() -> None:
    err = BrokerError(code="B_CUSTOM")
    assert err.code == "B_CUSTOM"
    assert err.status_code == 502


def test_data_error_custom_status_override() -> None:
    err = DataError(status_code=503)
    assert err.status_code == 503
    assert err.code == "DATA_ERROR"


def test_subclass_to_dict_includes_detail() -> None:
    err = RiskLimitExceeded("cap", detail={"limit": "size"})
    d = err.to_dict()
    assert d["error"] == "RISK_LIMIT_EXCEEDED"
    assert d["detail"] == {"limit": "size"}


def test_exception_is_base_exception_subclass() -> None:
    assert issubclass(TradingPlatformError, BaseException)
    assert issubclass(TradingPlatformError, Exception)


def test_strategy_already_exists_is_strategy_error() -> None:
    assert isinstance(StrategyAlreadyExistsError(), StrategyError)


def test_backtest_unavailable_is_backtest_error() -> None:
    assert isinstance(BacktestUnavailableError(), BacktestError)


def test_providers_unavailable_is_data_error() -> None:
    assert isinstance(ProvidersUnavailableError(), DataError)


def test_ml_unavailable_is_ml_error() -> None:
    assert isinstance(MLUnavailableError(), MLError)


def test_model_not_found_is_ml_error() -> None:
    assert isinstance(ModelNotFoundError(), MLError)


def test_trading_platform_error_message_only_kw_none_uses_empty() -> None:
    err = TradingPlatformError()
    assert str(err) == ""


def test_repr_reflects_runtime_code_after_subclass_defaults() -> None:
    err = OrderRejectedError("rej")
    assert "ORDER_REJECTED" in repr(err)
    assert "rej" in repr(err)


def test_strategy_param_error_is_strategy_error() -> None:
    assert isinstance(StrategyParamError("p"), StrategyError)


def test_strategy_error_is_trading_platform_error() -> None:
    assert isinstance(StrategyError(), TradingPlatformError)


def test_backtest_error_is_trading_platform_error() -> None:
    assert isinstance(BacktestError(), TradingPlatformError)


def test_risk_limit_exceeded_is_risk_error() -> None:
    assert isinstance(RiskLimitExceeded(), RiskError)


def test_risk_error_is_trading_platform_error() -> None:
    assert isinstance(RiskError(), TradingPlatformError)


def test_exchange_error_is_broker_error() -> None:
    assert isinstance(ExchangeError(), BrokerError)


def test_broker_connection_error_is_broker_error() -> None:
    assert isinstance(BrokerConnectionError(), BrokerError)


def test_execution_error_is_trading_platform_error() -> None:
    assert isinstance(ExecutionError(), TradingPlatformError)


def test_config_error_is_trading_platform_error() -> None:
    assert isinstance(ConfigError(), TradingPlatformError)


def test_catch_backtest_unavailable_as_backtest_error() -> None:
    with pytest.raises(BacktestError):
        raise BacktestUnavailableError("down")


def test_catch_risk_limit_as_risk_error() -> None:
    with pytest.raises(RiskError):
        raise RiskLimitExceeded("too big")


def test_catch_model_not_found_as_ml_error() -> None:
    with pytest.raises(MLError):
        raise ModelNotFoundError("m")


def test_trading_platform_error_to_dict_nested_detail() -> None:
    err = TradingPlatformError("e", detail={"outer": {"inner": 1}})
    assert err.to_dict()["detail"] == {"outer": {"inner": 1}}


def test_validation_error_custom_code_override() -> None:
    err = ValidationError("bad", code="V_CUSTOM")
    assert err.code == "V_CUSTOM"
    assert err.status_code == 422


def test_authentication_error_custom_status_override() -> None:
    err = AuthenticationError("nope", status_code=440)
    assert err.status_code == 440
    assert err.code == "AUTH_ERROR"


def test_permission_denied_can_be_caught_as_exception() -> None:
    with pytest.raises(Exception):
        raise PermissionDeniedError("denied")


def test_order_rejected_is_broker_error() -> None:
    assert isinstance(OrderRejectedError(), BrokerError)


def test_data_not_available_is_data_error() -> None:
    assert isinstance(DataNotAvailableError(), DataError)


def test_exchange_not_connected_is_broker_error() -> None:
    assert isinstance(ExchangeNotConnectedError(), BrokerError)


def test_providers_unavailable_is_trading_platform_error() -> None:
    assert isinstance(ProvidersUnavailableError(), TradingPlatformError)


def test_invalid_bars_is_trading_platform_error() -> None:
    assert isinstance(InvalidBarsError(), TradingPlatformError)


def test_ml_unavailable_is_trading_platform_error() -> None:
    assert isinstance(MLUnavailableError(), TradingPlatformError)


def test_service_not_ready_is_trading_platform_error() -> None:
    assert isinstance(ServiceNotReadyError(), TradingPlatformError)


def test_order_cancel_failed_is_broker_error() -> None:
    assert isinstance(OrderCancelFailedError(), BrokerError)


def test_catch_config_error_as_trading_platform_error() -> None:
    with pytest.raises(TradingPlatformError):
        raise ConfigError("cfg")


def test_catch_execution_error_as_trading_platform_error() -> None:
    with pytest.raises(TradingPlatformError):
        raise ExecutionError("exec")


def test_catch_data_not_available_as_data_error() -> None:
    with pytest.raises(DataError):
        raise DataNotAvailableError("gone")


def test_strategy_not_found_mro_includes_strategy_error() -> None:
    assert StrategyError in StrategyNotFoundError.__mro__


def test_broker_error_custom_status_override() -> None:
    err = BrokerError("x", status_code=599)
    assert err.status_code == 599
    assert err.code == "BROKER_ERROR"


def test_trading_platform_error_repr_contains_message_and_code() -> None:
    err = TradingPlatformError("say 'hi'", code="Q")
    r = repr(err)
    assert r.startswith("TradingPlatformError(")
    assert "code='Q'" in r
    assert "hi" in r


def test_subclass_detail_empty_omitted_in_to_dict() -> None:
    err = StrategyNotFoundError("n", detail={})
    assert "detail" not in err.to_dict()
