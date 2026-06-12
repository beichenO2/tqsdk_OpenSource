"""BTC 24 小时交易时段管理。

加密货币市场 7x24 不间断运行，但不同时段流动性和波动性差异显著。
本模块提供时段划分、流动性评估和时段感知的交易信号过滤。
"""

from __future__ import annotations

import enum
import logging
from datetime import datetime, time, timezone
from typing import Any

logger = logging.getLogger(__name__)


class SessionType(str, enum.Enum):
    ASIA = "asia"
    EUROPE = "europe"
    US = "us"
    OVERLAP_ASIA_EU = "overlap_asia_eu"
    OVERLAP_EU_US = "overlap_eu_us"
    LOW_LIQUIDITY = "low_liquidity"


_SESSION_RANGES: list[tuple[time, time, SessionType]] = [
    (time(1, 0), time(3, 0), SessionType.ASIA),
    (time(3, 0), time(8, 0), SessionType.OVERLAP_ASIA_EU),
    (time(8, 0), time(13, 0), SessionType.EUROPE),
    (time(13, 0), time(14, 0), SessionType.OVERLAP_EU_US),
    (time(14, 0), time(21, 0), SessionType.US),
    (time(21, 0), time(23, 59), SessionType.LOW_LIQUIDITY),
    (time(0, 0), time(1, 0), SessionType.LOW_LIQUIDITY),
]


SESSION_PROPERTIES: dict[SessionType, dict[str, Any]] = {
    SessionType.ASIA: {
        "liquidity": 0.6,
        "volatility_scale": 0.8,
        "description": "亚洲盘 (09:00-11:00 UTC+8)",
    },
    SessionType.EUROPE: {
        "liquidity": 0.8,
        "volatility_scale": 1.0,
        "description": "欧洲盘 (16:00-21:00 UTC+8)",
    },
    SessionType.US: {
        "liquidity": 1.0,
        "volatility_scale": 1.2,
        "description": "美国盘 (22:00-05:00+1 UTC+8)",
    },
    SessionType.OVERLAP_ASIA_EU: {
        "liquidity": 0.7,
        "volatility_scale": 0.9,
        "description": "亚欧重叠 (11:00-16:00 UTC+8)",
    },
    SessionType.OVERLAP_EU_US: {
        "liquidity": 1.0,
        "volatility_scale": 1.3,
        "description": "欧美重叠 (21:00-22:00 UTC+8) — 最高波动",
    },
    SessionType.LOW_LIQUIDITY: {
        "liquidity": 0.3,
        "volatility_scale": 0.6,
        "description": "低流动性 (05:00-09:00 UTC+8)",
    },
}


def get_current_session(utc_now: datetime | None = None) -> SessionType:
    """Return the current trading session based on UTC time."""
    if utc_now is None:
        utc_now = datetime.now(timezone.utc)
    t = utc_now.time()
    for start, end, session in _SESSION_RANGES:
        if start <= end:
            if start <= t < end:
                return session
        else:
            if t >= start or t < end:
                return session
    return SessionType.LOW_LIQUIDITY


def get_session_liquidity(session: SessionType | None = None) -> float:
    if session is None:
        session = get_current_session()
    return SESSION_PROPERTIES[session]["liquidity"]


def get_volatility_scale(session: SessionType | None = None) -> float:
    if session is None:
        session = get_current_session()
    return SESSION_PROPERTIES[session]["volatility_scale"]


class SessionAwareFilter:
    """根据交易时段属性调整信号强度和风控参数。

    - 低流动性时段降低信号强度、收紧止损
    - 高波动重叠时段放宽入场要求但减小仓位
    """

    def __init__(
        self,
        min_liquidity: float = 0.4,
        low_liq_strength_scale: float = 0.6,
        high_vol_position_scale: float = 0.7,
    ) -> None:
        self._min_liquidity = min_liquidity
        self._low_liq_scale = low_liq_strength_scale
        self._high_vol_pos_scale = high_vol_position_scale

    def should_trade(self, session: SessionType | None = None) -> bool:
        liq = get_session_liquidity(session)
        return liq >= self._min_liquidity

    def adjust_signal_strength(
        self, strength: float, session: SessionType | None = None
    ) -> float:
        liq = get_session_liquidity(session)
        if liq < 0.5:
            return round(strength * self._low_liq_scale, 4)
        return strength

    def adjust_position_size(
        self, base_qty: float, session: SessionType | None = None
    ) -> float:
        vol_scale = get_volatility_scale(session)
        if vol_scale > 1.1:
            return round(base_qty * self._high_vol_pos_scale, 8)
        return base_qty

    def get_session_report(self, utc_now: datetime | None = None) -> dict[str, Any]:
        session = get_current_session(utc_now)
        props = SESSION_PROPERTIES[session]
        return {
            "session": session.value,
            "description": props["description"],
            "liquidity": props["liquidity"],
            "volatility_scale": props["volatility_scale"],
            "tradeable": self.should_trade(session),
        }
