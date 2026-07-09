"""国内期货策略集 — CTA / DL / 缠论 / 新闻联动 / 日内策略 + 交易时段管理。"""

from . import spread_arb as _spread_arb  # noqa: F401 — @auto_register
from . import vol_breakout as _vol_breakout  # noqa: F401
from . import adaptive_bollinger as _adaptive_bollinger  # noqa: F401
from . import regime_momentum as _regime_momentum  # noqa: F401
from . import chan_theory as _chan_theory  # noqa: F401 — @auto_register
from . import news_sentiment as _news_sentiment  # noqa: F401 — @auto_register
from . import orderflow_imbalance as _orderflow_imbalance  # noqa: F401 — @auto_register
from . import cross_momentum as _cross_momentum  # noqa: F401 — @auto_register
from . import intraday_reversal as _intraday_reversal  # noqa: F401 — @auto_register
from . import intraday_reversal_v2_a as _intraday_reversal_v2_a  # noqa: F401 — @auto_register (hybrid A)
from . import intraday_reversal_v2_b as _intraday_reversal_v2_b  # noqa: F401 — @auto_register (hybrid B)
from . import intraday_reversal_v2_c as _intraday_reversal_v2_c  # noqa: F401 — @auto_register (hybrid C)
from . import kalman_trend as _kalman_trend  # noqa: F401 — @auto_register
from . import har_volatility as _har_volatility  # noqa: F401 — @auto_register
from . import ensemble_futures as _ensemble_futures  # noqa: F401 — @auto_register
from . import wavelet_trend as _wavelet_trend  # noqa: F401 — @auto_register
from . import hurst_adaptive as _hurst_adaptive  # noqa: F401 — @auto_register
from . import tick_microstructure as _tick_microstructure  # noqa: F401 — @auto_register
from . import fractal_dimension as _fractal_dimension  # noqa: F401 — @auto_register
from . import ou_stat_arb as _ou_stat_arb  # noqa: F401 — @auto_register
from . import supply_demand_zone as _supply_demand_zone  # noqa: F401 — @auto_register
from . import lunch_gap as _lunch_gap  # noqa: F401 — @auto_register
from . import nighttime_linkage as _nighttime_linkage  # noqa: F401 — @auto_register
from . import closing_pressure as _closing_pressure  # noqa: F401 — @auto_register
from . import market_profile as _market_profile  # noqa: F401 — @auto_register
from . import commodity_chain as _commodity_chain  # noqa: F401 — @auto_register
from . import pivot_point as _pivot_point  # noqa: F401 — @auto_register
from . import fibonacci_levels as _fibonacci_levels  # noqa: F401 — @auto_register
from . import supertrend as _supertrend  # noqa: F401 — @auto_register
from . import keltner_channel as _keltner_channel  # noqa: F401 — @auto_register
from . import rsi_divergence as _rsi_divergence  # noqa: F401 — @auto_register
from . import macd_histogram as _macd_histogram  # noqa: F401 — @auto_register
from . import donchian_breakout as _donchian_breakout  # noqa: F401 — @auto_register
from . import stochastic_oscillator as _stochastic_oscillator  # noqa: F401 — @auto_register
from . import williams_r as _williams_r  # noqa: F401 — @auto_register
from . import ichimoku_futures as _ichimoku_futures  # noqa: F401 — @auto_register
from . import parabolic_sar as _parabolic_sar  # noqa: F401 — @auto_register
from . import adx_trend_strength as _adx_trend_strength  # noqa: F401 — @auto_register
from . import opening_rush as _opening_rush  # noqa: F401 — @auto_register
from . import mamba_trend as _mamba_trend  # noqa: F401 — @auto_register
from . import regime_ensemble as _regime_ensemble  # noqa: F401 — @auto_register
from . import cross_asset_momentum as _cross_asset_momentum  # noqa: F401 — @auto_register
from . import attack_defense as _attack_defense  # noqa: F401 — @auto_register
from .adaptive_bollinger import AdaptiveBollingerStrategy
from .bollinger_mr import BollingerMRStrategy
from .chan_theory import ChanTheoryStrategy
from .cta_trend import CTATrendStrategy
from .dl_strategy import DLTimeseriesStrategy
from .dual_ma import FuturesDualMAStrategy
from .news_sentiment import NewsSentimentStrategy
from .pairs_trading import PairsTradingStrategy
from .rbreaker import RBreakerStrategy
from .regime_momentum import RegimeMomentumStrategy
from .spread_arb import SpreadArbitrage
from .vol_breakout import VolBreakoutStrategy
from .volume_price import VolumePriceStrategy
from .session import (
    FuturesSessionType,
    IntradayAction,
    IntradayGuard,
    FuturesSessionFilter,
    get_session,
    is_trading_hours,
    minutes_to_session_close,
)

__all__ = [
    "AdaptiveBollingerStrategy",
    "BollingerMRStrategy",
    "ChanTheoryStrategy",
    "CTATrendStrategy",
    "DLTimeseriesStrategy",
    "FuturesDualMAStrategy",
    "FuturesSessionFilter",
    "FuturesSessionType",
    "IntradayAction",
    "IntradayGuard",
    "NewsSentimentStrategy",
    "PairsTradingStrategy",
    "RBreakerStrategy",
    "RegimeMomentumStrategy",
    "SpreadArbitrage",
    "VolBreakoutStrategy",
    "VolumePriceStrategy",
    "get_session",
    "is_trading_hours",
    "minutes_to_session_close",
]
