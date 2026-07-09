"""国内期货专项风控 — 涨跌停 / 交割月 / 交易时段。

挂到 RiskEngine 作为下单前置闸，与通用限额（MaxOrderSize 等）并列。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime, time
from decimal import Decimal
from typing import TYPE_CHECKING, Optional
from zoneinfo import ZoneInfo

from risk.limits import RiskLimit

if TYPE_CHECKING:
    from execution.order_manager import OrderRequest
    from risk.limits import RiskContext

_CST = ZoneInfo("Asia/Shanghai")

# 合约代码末尾 4 位年月：rb2505 → 2025-05
_YM_RE = re.compile(r"(\d{4})$")

# 日盘 / 夜盘粗粒度窗口（覆盖主流商品；精确到品种可后续扩展）
_DAY_SESSIONS: list[tuple[time, time]] = [
    (time(9, 0), time(11, 30)),
    (time(13, 30), time(15, 15)),
]
_NIGHT_SESSIONS: list[tuple[time, time]] = [
    (time(21, 0), time(23, 59, 59)),
    (time(0, 0), time(2, 30)),
]


def _in_windows(now_t: time, windows: list[tuple[time, time]]) -> bool:
    for start, end in windows:
        if start <= end:
            if start <= now_t <= end:
                return True
        else:  # wraps midnight — not used here but keep safe
            if now_t >= start or now_t <= end:
                return True
    return False


def parse_delivery_ym(symbol: str) -> Optional[tuple[int, int]]:
    """从合约代码解析交割年月。支持 rb2505 / SHFE.rb2505 / DCE.i2509。"""
    bare = symbol.split(".")[-1]
    m = _YM_RE.search(bare)
    if not m:
        return None
    yyymm = m.group(1)
    year = 2000 + int(yyymm[:2])
    month = int(yyymm[2:])
    if not (1 <= month <= 12):
        return None
    return year, month


class LimitUpDownLimit(RiskLimit):
    """涨跌停近似：相对昨收/最新价偏离超过阈值则拒单。

    国内期货涨跌停多为 ±4%~±10%。默认用 last_price ± band_pct；
    若 context 提供 limit_up/limit_down 则优先用硬边界。
    """

    def __init__(self, band_pct: Decimal = Decimal("0.10")) -> None:
        self._band = band_pct

    @property
    def name(self) -> str:
        return "LimitUpDown"

    def check(self, request: OrderRequest, context: RiskContext) -> tuple[bool, str]:
        if request.price <= 0:
            return True, ""  # 市价单跳过价格带

        # 硬边界（可选，由上游注入到 last_prices 旁路字段）
        limit_up = getattr(context, "limit_up", {}).get(request.symbol) if hasattr(context, "limit_up") else None
        limit_down = getattr(context, "limit_down", {}).get(request.symbol) if hasattr(context, "limit_down") else None
        if limit_up is not None and request.price > Decimal(str(limit_up)):
            return False, f"Price {request.price} above limit-up {limit_up}"
        if limit_down is not None and request.price < Decimal(str(limit_down)):
            return False, f"Price {request.price} below limit-down {limit_down}"

        last = context.last_prices.get(request.symbol)
        if last is None or last == 0:
            return True, ""
        upper = last * (1 + self._band)
        lower = last * (1 - self._band)
        if request.price > upper or request.price < lower:
            return (
                False,
                f"Price {request.price} outside ±{self._band:.0%} band of last {last} "
                f"[{lower:.2f}, {upper:.2f}] (limit-up/down proxy)",
            )
        return True, ""


class DeliveryMonthLimit(RiskLimit):
    """交割月限制：进入交割月后禁止新开仓（默认），平仓放行。"""

    def __init__(
        self,
        block_open_in_delivery_month: bool = True,
        *,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self._block_open = block_open_in_delivery_month
        self._clock = clock or (lambda: datetime.now(_CST))

    @property
    def name(self) -> str:
        return "DeliveryMonth"

    def check(self, request: OrderRequest, context: RiskContext) -> tuple[bool, str]:
        from core.enums.direction import Offset

        if not self._block_open:
            return True, ""
        if request.offset != Offset.OPEN:
            return True, ""

        ym = parse_delivery_ym(request.symbol)
        if ym is None:
            return True, ""  # 非标准合约代码，跳过

        now = self._clock()
        if now.tzinfo is None:
            now = now.replace(tzinfo=_CST)
        else:
            now = now.astimezone(_CST)
        year, month = ym
        if now.year == year and now.month == month:
            return False, (
                f"Symbol {request.symbol} is in delivery month {year}-{month:02d}; "
                "new open orders blocked"
            )
        return True, ""


class TradingSessionLimit(RiskLimit):
    """交易时段闸：非日盘/夜盘窗口拒单（周末全拒）。"""

    def __init__(
        self,
        allow_night: bool = True,
        *,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self._allow_night = allow_night
        self._clock = clock or (lambda: datetime.now(_CST))

    @property
    def name(self) -> str:
        return "TradingSession"

    def check(self, request: OrderRequest, context: RiskContext) -> tuple[bool, str]:
        now = self._clock()
        if now.tzinfo is None:
            now = now.replace(tzinfo=_CST)
        else:
            now = now.astimezone(_CST)

        if now.weekday() >= 5:  # Sat/Sun
            # 周五夜盘可跨到周六凌晨 — 仅允许 00:00-02:30
            if now.weekday() == 5 and self._allow_night and _in_windows(now.time(), [(time(0, 0), time(2, 30))]):
                return True, ""
            return False, f"Market closed on weekend ({now.strftime('%A')})"

        t = now.time()
        if _in_windows(t, _DAY_SESSIONS):
            return True, ""
        if self._allow_night and _in_windows(t, _NIGHT_SESSIONS):
            return True, ""
        return False, f"Outside trading session at {now.strftime('%H:%M:%S')} CST"
