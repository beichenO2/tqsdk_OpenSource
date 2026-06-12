"""国内期货交易时段管理 — 日内交易框架核心。

中国期货市场有严格的交易时段：
- 日盘：09:00-10:15, 10:30-11:30, 13:30-15:00
- 夜盘：21:00-23:00 (大多数品种) / 01:00 (铜铝锌等) / 02:30 (黄金白银)
- 无夜盘品种：苹果(AP)、红枣(CJ)、花生(PK)等

本模块提供：
1. 交易时段识别与状态机
2. 强制日终平仓逻辑（收盘前 N 分钟禁止开仓 + 平仓信号）
3. 开盘 rush / 午间 / 夜盘等时段特征参数
4. 与 BaseStrategy 集成的 mixin
"""

from __future__ import annotations

import enum
import logging
from datetime import datetime, time, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

CST = timezone(timedelta(hours=8))


class FuturesSessionType(str, enum.Enum):
    MORNING_OPEN = "morning_open"
    MORNING_MID = "morning_mid"
    MORNING_BREAK = "morning_break"
    MORNING_LATE = "morning_late"
    LUNCH_BREAK = "lunch_break"
    AFTERNOON = "afternoon"
    AFTERNOON_CLOSE = "afternoon_close"
    NIGHT_OPEN = "night_open"
    NIGHT_MID = "night_mid"
    NIGHT_LATE = "night_late"
    CLOSED = "closed"


class NightSessionEnd(str, enum.Enum):
    """夜盘收盘时间按品种分类。"""
    H23 = "23:00"
    H01 = "01:00"
    H230 = "02:30"
    NONE = "none"


NIGHT_SESSION_MAP: dict[str, NightSessionEnd] = {
    "cu": NightSessionEnd.H01, "al": NightSessionEnd.H01,
    "zn": NightSessionEnd.H01, "pb": NightSessionEnd.H01,
    "sn": NightSessionEnd.H01, "ni": NightSessionEnd.H01,
    "ss": NightSessionEnd.H01, "bc": NightSessionEnd.H01,
    "au": NightSessionEnd.H230, "ag": NightSessionEnd.H230,
    "sc": NightSessionEnd.H230,
    "AP": NightSessionEnd.NONE, "CJ": NightSessionEnd.NONE,
    "PK": NightSessionEnd.NONE, "UR": NightSessionEnd.NONE,
    "SA": NightSessionEnd.NONE, "LH": NightSessionEnd.NONE,
}

_DAY_SESSIONS: list[tuple[time, time, FuturesSessionType]] = [
    (time(9, 0), time(9, 15), FuturesSessionType.MORNING_OPEN),
    (time(9, 15), time(10, 0), FuturesSessionType.MORNING_MID),
    (time(10, 0), time(10, 15), FuturesSessionType.MORNING_MID),
    (time(10, 15), time(10, 30), FuturesSessionType.MORNING_BREAK),
    (time(10, 30), time(11, 30), FuturesSessionType.MORNING_LATE),
    (time(11, 30), time(13, 30), FuturesSessionType.LUNCH_BREAK),
    (time(13, 30), time(14, 45), FuturesSessionType.AFTERNOON),
    (time(14, 45), time(15, 0), FuturesSessionType.AFTERNOON_CLOSE),
]

_NIGHT_SESSIONS: list[tuple[time, time, FuturesSessionType]] = [
    (time(21, 0), time(21, 15), FuturesSessionType.NIGHT_OPEN),
    (time(21, 15), time(22, 30), FuturesSessionType.NIGHT_MID),
    (time(22, 30), time(23, 59), FuturesSessionType.NIGHT_LATE),
    (time(0, 0), time(2, 30), FuturesSessionType.NIGHT_LATE),
]


SESSION_PROPERTIES: dict[FuturesSessionType, dict[str, Any]] = {
    FuturesSessionType.MORNING_OPEN: {
        "liquidity": 1.0, "volatility_scale": 1.4,
        "description": "早盘开盘冲击 09:00-09:15 — 流动性最高，波动剧烈",
        "allow_new_entry": True,
    },
    FuturesSessionType.MORNING_MID: {
        "liquidity": 0.9, "volatility_scale": 1.0,
        "description": "早盘中段 09:15-10:15 — 趋势延续主时段",
        "allow_new_entry": True,
    },
    FuturesSessionType.MORNING_BREAK: {
        "liquidity": 0.0, "volatility_scale": 0.0,
        "description": "早盘休息 10:15-10:30 — 不交易",
        "allow_new_entry": False,
    },
    FuturesSessionType.MORNING_LATE: {
        "liquidity": 0.7, "volatility_scale": 0.8,
        "description": "早盘后段 10:30-11:30 — 波动收敛",
        "allow_new_entry": True,
    },
    FuturesSessionType.LUNCH_BREAK: {
        "liquidity": 0.0, "volatility_scale": 0.0,
        "description": "午间休息 11:30-13:30 — 不交易",
        "allow_new_entry": False,
    },
    FuturesSessionType.AFTERNOON: {
        "liquidity": 0.8, "volatility_scale": 0.9,
        "description": "午盘 13:30-14:45 — 稳定交易段",
        "allow_new_entry": True,
    },
    FuturesSessionType.AFTERNOON_CLOSE: {
        "liquidity": 0.6, "volatility_scale": 1.1,
        "description": "收盘前 14:45-15:00 — 禁止新开仓，仅允许平仓",
        "allow_new_entry": False,
    },
    FuturesSessionType.NIGHT_OPEN: {
        "liquidity": 0.9, "volatility_scale": 1.3,
        "description": "夜盘开盘 21:00-21:15 — 受外盘影响大，波动高",
        "allow_new_entry": True,
    },
    FuturesSessionType.NIGHT_MID: {
        "liquidity": 0.7, "volatility_scale": 0.9,
        "description": "夜盘中段 21:15-22:30",
        "allow_new_entry": True,
    },
    FuturesSessionType.NIGHT_LATE: {
        "liquidity": 0.5, "volatility_scale": 0.7,
        "description": "夜盘尾段 22:30-收盘 — 仅有色/贵金属/原油延长",
        "allow_new_entry": True,
    },
    FuturesSessionType.CLOSED: {
        "liquidity": 0.0, "volatility_scale": 0.0,
        "description": "非交易时段",
        "allow_new_entry": False,
    },
}


def get_night_session_end(symbol: str) -> NightSessionEnd:
    """获取品种的夜盘收盘时间类型。"""
    base = "".join(c for c in symbol if c.isalpha()).upper()
    base_lower = base.lower()
    return NIGHT_SESSION_MAP.get(base, NIGHT_SESSION_MAP.get(base_lower, NightSessionEnd.H23))


def get_session(cst_now: datetime | None = None) -> FuturesSessionType:
    """根据北京时间返回当前交易时段。"""
    if cst_now is None:
        cst_now = datetime.now(CST)
    t = cst_now.time()

    for start, end, session in _DAY_SESSIONS:
        if start <= t < end:
            return session

    for start, end, session in _NIGHT_SESSIONS:
        if start <= end:
            if start <= t < end:
                return session
        else:
            if t >= start or t < end:
                return session

    return FuturesSessionType.CLOSED


def is_trading_hours(cst_now: datetime | None = None) -> bool:
    """当前是否在交易时段内（排除休息时间）。"""
    session = get_session(cst_now)
    return session not in (
        FuturesSessionType.CLOSED,
        FuturesSessionType.MORNING_BREAK,
        FuturesSessionType.LUNCH_BREAK,
    )


def minutes_to_session_close(
    cst_now: datetime | None = None,
    symbol: str = "rb",
) -> float | None:
    """距本交易时段收盘的分钟数。日盘固定 15:00，夜盘按品种。

    非交易时段返回 None。
    """
    if cst_now is None:
        cst_now = datetime.now(CST)
    t = cst_now.time()

    if time(9, 0) <= t < time(15, 0):
        close_dt = cst_now.replace(hour=15, minute=0, second=0, microsecond=0)
        return max(0.0, (close_dt - cst_now).total_seconds() / 60.0)

    night_end = get_night_session_end(symbol)
    if night_end == NightSessionEnd.NONE:
        return None

    close_map = {
        NightSessionEnd.H23: time(23, 0),
        NightSessionEnd.H01: time(1, 0),
        NightSessionEnd.H230: time(2, 30),
    }
    close_time = close_map[night_end]

    if t >= time(21, 0) or t < time(3, 0):
        close_dt = cst_now.replace(
            hour=close_time.hour, minute=close_time.minute,
            second=0, microsecond=0,
        )
        if close_time < time(12, 0) and t >= time(21, 0):
            close_dt += timedelta(days=1)
        diff = (close_dt - cst_now).total_seconds() / 60.0
        return max(0.0, diff)

    return None


class IntradayGuard:
    """日内交易守卫 — 确保收盘前强制平仓、禁止新开仓。

    用法：
        guard = IntradayGuard(close_minutes=15, warn_minutes=30)

        # 在策略 on_bar 中：
        action = guard.check(bar_time, symbol, has_position=True)
        if action == IntradayAction.FORCE_CLOSE:
            # 立即市价平仓
        elif action == IntradayAction.ENTRY_BLOCKED:
            # 跳过开仓信号
        elif action == IntradayAction.WARN:
            # 收紧止损
    """

    def __init__(
        self,
        close_minutes: float = 5.0,
        block_entry_minutes: float = 15.0,
        warn_minutes: float = 30.0,
    ) -> None:
        self.close_minutes = close_minutes
        self.block_entry_minutes = block_entry_minutes
        self.warn_minutes = warn_minutes

    def check(
        self,
        bar_time: datetime,
        symbol: str = "rb",
        has_position: bool = False,
    ) -> IntradayAction:
        """检查当前时间的日内约束状态。

        日内休息（10:15-10:30 和 11:30-13:30）期间持仓保留，仅禁止新开仓。
        仅在真正的收盘时段（15:00 前/夜盘结束前）才触发强制平仓。
        """
        cst_time = bar_time.astimezone(CST) if bar_time.tzinfo else bar_time
        session = get_session(cst_time)

        if session in (FuturesSessionType.MORNING_BREAK, FuturesSessionType.LUNCH_BREAK):
            return IntradayAction.ENTRY_BLOCKED

        if session == FuturesSessionType.CLOSED:
            if has_position:
                return IntradayAction.FORCE_CLOSE
            return IntradayAction.ENTRY_BLOCKED

        remaining = minutes_to_session_close(cst_time, symbol)
        if remaining is None:
            return IntradayAction.NORMAL

        if remaining <= self.close_minutes:
            if has_position:
                return IntradayAction.FORCE_CLOSE
            return IntradayAction.ENTRY_BLOCKED

        if remaining <= self.block_entry_minutes:
            return IntradayAction.ENTRY_BLOCKED

        if remaining <= self.warn_minutes:
            return IntradayAction.WARN

        return IntradayAction.NORMAL


class IntradayAction(str, enum.Enum):
    NORMAL = "normal"
    WARN = "warn"
    ENTRY_BLOCKED = "entry_blocked"
    FORCE_CLOSE = "force_close"


class FuturesSessionFilter:
    """时段感知的信号过滤器，类似 BTC 的 SessionAwareFilter。"""

    def __init__(
        self,
        min_liquidity: float = 0.3,
        open_rush_position_scale: float = 0.6,
    ) -> None:
        self._min_liquidity = min_liquidity
        self._open_rush_scale = open_rush_position_scale

    def should_trade(self, session: FuturesSessionType | None = None) -> bool:
        if session is None:
            session = get_session()
        props = SESSION_PROPERTIES.get(session, {})
        return props.get("allow_new_entry", False)

    def adjust_position_size(
        self, base_qty: float, session: FuturesSessionType | None = None,
    ) -> float:
        if session is None:
            session = get_session()
        if session in (FuturesSessionType.MORNING_OPEN, FuturesSessionType.NIGHT_OPEN):
            return round(base_qty * self._open_rush_scale, 4)
        return base_qty

    def adjust_signal_strength(
        self, strength: float, session: FuturesSessionType | None = None,
    ) -> float:
        if session is None:
            session = get_session()
        props = SESSION_PROPERTIES.get(session, {})
        vol_scale = props.get("volatility_scale", 1.0)
        if vol_scale > 1.2:
            return round(strength * 1.1, 4)
        if vol_scale < 0.5:
            return round(strength * 0.7, 4)
        return strength

    def get_session_report(self, cst_now: datetime | None = None) -> dict[str, Any]:
        session = get_session(cst_now)
        props = SESSION_PROPERTIES.get(session, {})
        return {
            "session": session.value,
            "description": props.get("description", ""),
            "liquidity": props.get("liquidity", 0.0),
            "volatility_scale": props.get("volatility_scale", 0.0),
            "allow_new_entry": props.get("allow_new_entry", False),
            "tradeable": self.should_trade(session),
        }
