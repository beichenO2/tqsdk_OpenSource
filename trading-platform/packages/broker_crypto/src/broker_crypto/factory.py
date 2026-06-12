"""交易所适配器工厂 — 根据 Exchange 枚举创建对应的适配器实例。"""

from __future__ import annotations

from .base import ExchangeAdapter
from .binance import BinanceAdapter
from .models import Exchange, ExchangeCredentials
from .okx import OKXAdapter
from .weex import WEEXAdapter


_REGISTRY: dict[Exchange, type[ExchangeAdapter]] = {
    Exchange.BINANCE: BinanceAdapter,
    Exchange.OKX: OKXAdapter,
    Exchange.WEEX: WEEXAdapter,
}


def create_adapter(credentials: ExchangeCredentials) -> ExchangeAdapter:
    """Create an exchange adapter for the given credentials."""
    adapter_cls = _REGISTRY.get(credentials.exchange)
    if adapter_cls is None:
        raise ValueError(
            f"Unsupported exchange: {credentials.exchange}. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return adapter_cls(credentials)


def register_adapter(exchange: Exchange, adapter_cls: type[ExchangeAdapter]) -> None:
    """Register a custom adapter (e.g., for Bybit)."""
    _REGISTRY[exchange] = adapter_cls
