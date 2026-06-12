"""Crypto trading strategies — eager imports trigger @auto_register."""

from __future__ import annotations

from . import adaptive_trend as _adaptive_trend  # noqa: F401 — @auto_register
from . import atr_channel_breakout as _atr_channel_breakout  # noqa: F401
from . import crypto_pairs as _crypto_pairs  # noqa: F401
from . import dual_momentum as _dual_momentum  # noqa: F401
from . import engulfing_pattern as _engulfing_pattern  # noqa: F401
from . import ensemble_strategy as _ensemble_strategy  # noqa: F401
from . import extreme_reversal as _extreme_reversal  # noqa: F401
from . import fibonacci_pullback as _fibonacci_pullback  # noqa: F401
from . import funding_meta_ensemble as _funding_meta_ensemble  # noqa: F401
from . import funding_rate_alpha as _funding_rate_alpha  # noqa: F401
from . import funding_rate_arb as _funding_rate_arb  # noqa: F401
from . import funding_rate_v2 as _funding_rate_v2  # noqa: F401
from . import grid_v2 as _grid_v2  # noqa: F401
from . import hurst_regime_switch as _hurst_regime_switch  # noqa: F401
from . import ichimoku_cloud as _ichimoku_cloud  # noqa: F401
from . import kalman_trend as _kalman_trend  # noqa: F401
from . import keltner_pullback as _keltner_pullback  # noqa: F401
from . import liquidation_reversal as _liquidation_reversal  # noqa: F401
from . import macd_divergence as _macd_divergence  # noqa: F401
from . import momentum_rotation as _momentum_rotation  # noqa: F401
from . import mtf_confluence as _mtf_confluence  # noqa: F401
from . import ou_mean_reversion as _ou_mean_reversion  # noqa: F401
from . import portfolio_strategy as _portfolio_strategy  # noqa: F401
from . import range_breakout as _range_breakout  # noqa: F401
from . import regime_adaptive as _regime_adaptive  # noqa: F401
from . import rsi_divergence as _rsi_divergence  # noqa: F401
from . import scalp_momentum as _scalp_momentum  # noqa: F401
from . import session_momentum as _session_momentum  # noqa: F401
from . import signal_consensus as _signal_consensus  # noqa: F401
from . import smart_money_fvg as _smart_money_fvg  # noqa: F401
from . import squeeze_breakout as _squeeze_breakout  # noqa: F401
from . import supertrend as _supertrend  # noqa: F401
from . import taker_imbalance as _taker_imbalance  # noqa: F401
from . import trend_following_v2 as _trend_following_v2  # noqa: F401
from . import triple_ema_crossover as _triple_ema_crossover  # noqa: F401
from . import turtle_system as _turtle_system  # noqa: F401
from . import volatility_breakout_scalp as _volatility_breakout_scalp  # noqa: F401
from . import volume_profile_flow as _volume_profile_flow  # noqa: F401
from . import vwap_reversion as _vwap_reversion  # noqa: F401
from . import williams_fractal as _williams_fractal  # noqa: F401
from . import wyckoff_phases as _wyckoff_phases  # noqa: F401
from . import cross_sectional_momentum as _cross_sectional_momentum  # noqa: F401

from .funding_rate_alpha import FundingRateAlphaStrategy
from .cross_sectional_momentum import TimeSeriesMomentumStrategy
from .regime_detector import MarketRegimeDetector, MarketRegime
from .volatility_breakout_scalp import VolatilityBreakoutScalpStrategy
from .scalp_momentum import ScalpMomentumStrategy
from .funding_meta_ensemble import FundingMetaEnsembleStrategy
from .portfolio_strategy import PortfolioStrategy
from .funding_rate_arb import FundingRateArbitrage

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
]
