"""Trading Platform 统一异常体系

所有业务异常继承 ``TradingPlatformError``，携带机器可读的 ``code`` 和 HTTP ``status_code``；
API 层通过全局 exception_handler 统一转换为 ``{"error": code, "message": ...}``。
"""
from __future__ import annotations

from typing import Any


class TradingPlatformError(Exception):
    """项目根异常 — 所有业务错误的基类"""

    code: str = "INTERNAL_ERROR"
    status_code: int = 500

    def __init__(
        self,
        message: str = "",
        *,
        code: str | None = None,
        status_code: int | None = None,
        detail: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code
        self.detail = detail or {}

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"error": self.code, "message": str(self)}
        if self.detail:
            payload["detail"] = self.detail
        return payload

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.code!r}, message={str(self)!r})"


# ── 策略层 ──────────────────────────────────────────────────
class StrategyError(TradingPlatformError):
    code = "STRATEGY_ERROR"
    status_code = 400


class StrategyNotFoundError(StrategyError):
    code = "STRATEGY_NOT_FOUND"
    status_code = 404


class StrategyAlreadyExistsError(StrategyError):
    code = "STRATEGY_ALREADY_EXISTS"
    status_code = 409


class StrategyParamError(StrategyError):
    code = "STRATEGY_PARAM_INVALID"
    status_code = 422


# ── 回测层 ──────────────────────────────────────────────────
class BacktestError(TradingPlatformError):
    code = "BACKTEST_ERROR"
    status_code = 400


class BacktestUnavailableError(BacktestError):
    code = "BACKTEST_UNAVAILABLE"
    status_code = 503


class InvalidBarsError(BacktestError):
    code = "INVALID_BARS"
    status_code = 422


# ── 风控层 ──────────────────────────────────────────────────
class RiskError(TradingPlatformError):
    code = "RISK_ERROR"
    status_code = 400


class RiskLimitExceeded(RiskError):
    code = "RISK_LIMIT_EXCEEDED"
    status_code = 403


# ── 数据层 ──────────────────────────────────────────────────
class DataError(TradingPlatformError):
    code = "DATA_ERROR"
    status_code = 500


class DataNotAvailableError(DataError):
    code = "DATA_NOT_AVAILABLE"
    status_code = 404


class ProvidersUnavailableError(DataError):
    code = "PROVIDERS_UNAVAILABLE"
    status_code = 503


# ── 经纪/执行层 ────────────────────────────────────────────
class BrokerError(TradingPlatformError):
    code = "BROKER_ERROR"
    status_code = 502


class BrokerConnectionError(BrokerError):
    code = "BROKER_CONN_FAILED"
    status_code = 503


class ExchangeNotConnectedError(BrokerError):
    code = "EXCHANGE_NOT_CONNECTED"
    status_code = 503


class ExchangeError(BrokerError):
    code = "EXCHANGE_ERROR"
    status_code = 502


class OrderRejectedError(BrokerError):
    code = "ORDER_REJECTED"
    status_code = 400


class OrderNotFoundError(BrokerError):
    code = "ORDER_NOT_FOUND"
    status_code = 404


class OrderCancelFailedError(BrokerError):
    code = "ORDER_CANCEL_FAILED"
    status_code = 400


class ExecutionError(TradingPlatformError):
    code = "EXECUTION_ERROR"
    status_code = 500


class ServiceNotReadyError(TradingPlatformError):
    code = "SERVICE_NOT_READY"
    status_code = 503


# ── ML 层 ───────────────────────────────────────────────────
class MLError(TradingPlatformError):
    code = "ML_ERROR"
    status_code = 500


class MLUnavailableError(MLError):
    code = "ML_UNAVAILABLE"
    status_code = 503


class ModelNotFoundError(MLError):
    code = "MODEL_NOT_FOUND"
    status_code = 404


# ── 配置/认证 ──────────────────────────────────────────────
class ConfigError(TradingPlatformError):
    code = "CONFIG_ERROR"
    status_code = 500


class AuthenticationError(TradingPlatformError):
    code = "AUTH_ERROR"
    status_code = 401


class PermissionDeniedError(TradingPlatformError):
    code = "PERMISSION_DENIED"
    status_code = 403


# ── 输入验证 ───────────────────────────────────────────────
class ValidationError(TradingPlatformError):
    code = "VALIDATION_ERROR"
    status_code = 422
