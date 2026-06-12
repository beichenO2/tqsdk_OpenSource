"""加密货币交易所接入层 - 提供统一的交易所 API 抽象。

此包由 Ch32（BTC 数据管道）主导实现，Ch33 提供接口桩。
src/broker_crypto/ 包含完整实现，此处做 re-export。
"""
from __future__ import annotations

try:
    from .src.broker_crypto.manager import BTCBrokerManager
    from .src.broker_crypto.models import Exchange as CryptoExchange, ExchangeCredentials
    from .src.broker_crypto.base import ExchangeAdapter
    from .src.broker_crypto.market_adapter import CryptoMarketAdapter
except ImportError:
    from .base import (
        ExchangeConfig,
        OrderResult,
        TickerData,
        OrderBookData,
    )

    import enum as _enum

    class CryptoExchange(str, _enum.Enum):  # type: ignore[no-redef]
        """Fallback enum when src/broker_crypto is not importable."""
        BINANCE = "BINANCE"
        OKX = "OKX"
        BYBIT = "BYBIT"
        WEEX = "WEEX"

    BTCBrokerManager = None  # type: ignore[assignment,misc]
    ExchangeCredentials = None  # type: ignore[assignment,misc]
    ExchangeAdapter = None  # type: ignore[assignment,misc]
    CryptoMarketAdapter = None  # type: ignore[assignment,misc]

__all__ = [
    "BTCBrokerManager",
    "CryptoExchange",
    "ExchangeAdapter",
    "ExchangeConfig",
    "ExchangeCredentials",
    "CryptoMarketAdapter",
    "OrderBookData",
    "OrderResult",
    "TickerData",
]
