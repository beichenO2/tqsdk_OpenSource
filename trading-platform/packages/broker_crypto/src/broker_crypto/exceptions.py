"""broker_crypto 异常体系"""

from __future__ import annotations


class BrokerCryptoError(Exception):
    """broker_crypto 基础异常"""


class ExchangeConnectionError(BrokerCryptoError):
    """交易所连接异常"""


class ExchangeAPIError(BrokerCryptoError):
    """交易所 API 返回错误"""

    def __init__(self, exchange_id: str, code: int | str, message: str) -> None:
        self.exchange_id = exchange_id
        self.code = code
        super().__init__(f"[{exchange_id}] API error {code}: {message}")


class OrderRejectedError(BrokerCryptoError):
    """订单被拒绝"""


class InsufficientBalanceError(BrokerCryptoError):
    """余额不足"""


class RateLimitError(BrokerCryptoError):
    """触发交易所速率限制"""

    def __init__(self, exchange_id: str, retry_after: float | None = None) -> None:
        self.exchange_id = exchange_id
        self.retry_after = retry_after
        msg = f"[{exchange_id}] Rate limited"
        if retry_after:
            msg += f", retry after {retry_after}s"
        super().__init__(msg)


class SymbolNotFoundError(BrokerCryptoError):
    """交易对不存在"""
