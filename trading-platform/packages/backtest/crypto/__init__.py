"""BTC / crypto backtesting sub-package.

Provides pluggable cost models, slippage models, an LOB simulator,
and a 24/7-aware backtest engine built on top of the standard
futures backtest framework.
"""

from .cost_model import (
    CostBreakdown,
    CostModel,
    FeeRole,
    FeeTier,
    FlatRateCostModel,
    MakerTakerCostModel,
    TieredCostModel,
)
from .engine import (
    CryptoBacktestConfig,
    CryptoBacktestEngine,
    CryptoBacktestResult,
    CryptoFill,
    CryptoOrder,
    CryptoPosition,
)
from .orderbook import (
    BookFill,
    BookOrder,
    CryptoOrderBook,
    LimitLevel,
    MatchResult,
    Side,
)
from .slippage import (
    FixedBpsSlippage,
    SlippageModel,
    SlippageResult,
    VolatilityAdaptiveSlippage,
    VolumeImpactSlippage,
)

__all__ = [
    "CostBreakdown",
    "CostModel",
    "CryptoBacktestConfig",
    "CryptoBacktestEngine",
    "CryptoBacktestResult",
    "CryptoFill",
    "CryptoOrder",
    "CryptoOrderBook",
    "CryptoPosition",
    "FeeRole",
    "FeeTier",
    "FixedBpsSlippage",
    "FlatRateCostModel",
    "MakerTakerCostModel",
    "BookFill",
    "BookOrder",
    "LimitLevel",
    "MatchResult",
    "Side",
    "SlippageModel",
    "SlippageResult",
    "TieredCostModel",
    "VolatilityAdaptiveSlippage",
    "VolumeImpactSlippage",
]
