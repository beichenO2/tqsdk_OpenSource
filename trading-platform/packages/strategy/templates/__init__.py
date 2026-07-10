"""通用参数化策略模板 — 用于批量生成 SOTA 策略变体。"""

from .adaptive_trend import AdaptiveTrendStrategy
from .channel_breakout import ChannelBreakoutStrategy
from .ema_derivative import EMADerivativeStrategy
from .factor_strategy import FactorStrategy
from .kalman_trend import KalmanTrendStrategy
from .momentum_rotation import MomentumRotationStrategy
from .multi_indicator import MultiIndicatorStrategy
from .orderflow import OrderFlowStrategy
from .regime_filter import RegimeFilterStrategy
from .stat_arb import StatArbStrategy
from .supertrend import SupertrendStrategy
from .trix_alpha import TrixAlphaStrategy
from .vol_target import VolTargetStrategy

__all__ = [
    "AdaptiveTrendStrategy",
    "ChannelBreakoutStrategy",
    "EMADerivativeStrategy",
    "FactorStrategy",
    "KalmanTrendStrategy",
    "MomentumRotationStrategy",
    "MultiIndicatorStrategy",
    "OrderFlowStrategy",
    "RegimeFilterStrategy",
    "StatArbStrategy",
    "SupertrendStrategy",
    "TrixAlphaStrategy",
    "VolTargetStrategy",
]
