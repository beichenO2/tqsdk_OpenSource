"""BTC 策略包 — 向后兼容 shim。

新代码请直接使用 crypto.strategies 包：
    from crypto.strategies import FundingRateAlphaStrategy
旧代码 `from strategy.btc import ...` 仍然有效（通过此 shim 代理）。
"""

from __future__ import annotations

import warnings as _w
from typing import Any

_w.warn(
    "strategy.btc is deprecated — use crypto.strategies instead",
    DeprecationWarning,
    stacklevel=2,
)

from .funding_rate_alpha import FundingRateAlphaStrategy
from .cross_sectional_momentum import TimeSeriesMomentumStrategy
from .regime_detector import MarketRegimeDetector, MarketRegime
from .volatility_breakout_scalp import VolatilityBreakoutScalpStrategy  # noqa: F401
from .scalp_momentum import ScalpMomentumStrategy  # noqa: F401
from .funding_meta_ensemble import FundingMetaEnsembleStrategy  # noqa: F401
from .portfolio_strategy import PortfolioStrategy  # noqa: F401
from .funding_rate_arb import FundingRateArbitrage  # noqa: F401

__all__ = [
    "FundingRateAlphaStrategy",
    "TimeSeriesMomentumStrategy",
    "MarketRegimeDetector",
    "MarketRegime",
    "VolatilityBreakoutScalpStrategy",
    "ScalpMomentumStrategy",
    "FundingMetaEnsembleStrategy",
    "PortfolioStrategy",
    "FundingRateArbitrage",
    "BacktestStrategyAdapter",
    "CryptoBrokerAdapter",
    "VolatilityCircuitBreaker",
    "SpreadLimit",
    "FundingRateLimit",
    "CryptoPositionValueLimit",
    "LeverageLimit",
    "LiquidationGuard",
    "SessionAwareFilter",
    "SessionType",
    "get_current_session",
]


_LAZY_RISK = {
    "VolatilityCircuitBreaker",
    "SpreadLimit",
    "FundingRateLimit",
    "CryptoPositionValueLimit",
    "LeverageLimit",
    "LiquidationGuard",
}

_LAZY_SESSION = {
    "SessionAwareFilter",
    "SessionType",
    "get_current_session",
}


def __getattr__(name: str) -> Any:
    """Lazy imports for BTC adapter / risk / session modules."""
    if name == "BacktestStrategyAdapter":
        from .backtest_adapter import BacktestStrategyAdapter

        return BacktestStrategyAdapter
    if name == "CryptoBrokerAdapter":
        from .broker_adapter import CryptoBrokerAdapter

        return CryptoBrokerAdapter
    if name in _LAZY_RISK:
        from . import risk_limits as _rl

        return getattr(_rl, name)
    if name in _LAZY_SESSION:
        from . import session as _sess

        return getattr(_sess, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
