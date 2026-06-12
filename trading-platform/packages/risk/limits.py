"""Pluggable risk limit definitions for pre-trade and real-time checks."""

from __future__ import annotations

import abc
import time
from collections import defaultdict, deque
from decimal import Decimal
from typing import TYPE_CHECKING, Optional

from core.enums.direction import Direction, Offset
from core.models.position import Position

if TYPE_CHECKING:
    from execution.order_manager import OrderRequest


class RiskContext:
    """Snapshot of current risk state passed to limit checks."""

    def __init__(
        self,
        positions: list[Position],
        balance: Decimal = Decimal("0"),
        available: Decimal = Decimal("0"),
        margin_ratio: Decimal = Decimal("0"),
        last_prices: Optional[dict[str, Decimal]] = None,
    ) -> None:
        self.positions = {f"{p.symbol}:{p.direction.value}": p for p in positions}
        self.balance = balance
        self.available = available
        self.margin_ratio = margin_ratio
        self.last_prices = last_prices or {}

    def get_position(self, symbol: str, direction: Direction) -> Optional[Position]:
        return self.positions.get(f"{symbol}:{direction.value}")

    def get_total_volume(self, symbol: str) -> int:
        long = self.positions.get(f"{symbol}:LONG")
        short = self.positions.get(f"{symbol}:SHORT")
        return (long.volume if long else 0) + (short.volume if short else 0)


class RiskLimit(abc.ABC):
    @abc.abstractmethod
    def check(self, request: OrderRequest, context: RiskContext) -> tuple[bool, str]: ...

    @property
    @abc.abstractmethod
    def name(self) -> str: ...


class MaxOrderSizeLimit(RiskLimit):
    """Reject orders exceeding maximum volume per order."""

    def __init__(self, max_volume: int = 100) -> None:
        self._max = max_volume

    @property
    def name(self) -> str:
        return "MaxOrderSize"

    def check(self, request, context) -> tuple[bool, str]:
        if request.volume > self._max:
            return False, f"Volume {request.volume} exceeds max {self._max}"
        return True, ""


class MaxPositionLimit(RiskLimit):
    """Reject opening orders that would exceed position limit."""

    def __init__(self, max_position: int = 500, per_symbol: Optional[dict[str, int]] = None) -> None:
        self._max = max_position
        self._per_symbol = per_symbol or {}

    @property
    def name(self) -> str:
        return "MaxPosition"

    def check(self, request, context) -> tuple[bool, str]:
        if request.offset != Offset.OPEN:
            return True, ""
        current = context.get_total_volume(request.symbol)
        limit = self._per_symbol.get(request.symbol, self._max)
        if current + request.volume > limit:
            return False, f"Position would be {current + request.volume}, limit {limit}"
        return True, ""


class PriceBandLimit(RiskLimit):
    """Reject orders deviating too far from the last price."""

    def __init__(self, max_deviation_pct: Decimal = Decimal("0.05")) -> None:
        self._max_dev = max_deviation_pct

    @property
    def name(self) -> str:
        return "PriceBand"

    def check(self, request, context) -> tuple[bool, str]:
        last = context.last_prices.get(request.symbol)
        if last is None or last == 0:
            return True, ""
        dev = abs(request.price - last) / last
        if dev > self._max_dev:
            return False, f"Price deviation {dev:.2%} exceeds {self._max_dev:.2%}"
        return True, ""


class OrderFrequencyLimit(RiskLimit):
    """Rate-limit order submissions per symbol."""

    def __init__(self, max_orders: int = 20, window_seconds: float = 60.0) -> None:
        self._max = max_orders
        self._window = window_seconds
        self._times: dict[str, deque[float]] = defaultdict(deque)

    @property
    def name(self) -> str:
        return "OrderFrequency"

    def check(self, request, context) -> tuple[bool, str]:
        now = time.monotonic()
        q = self._times[request.symbol]
        while q and now - q[0] > self._window:
            q.popleft()
        if len(q) >= self._max:
            return False, f"Frequency {len(q)}/{self._window}s exceeds {self._max}"
        q.append(now)
        return True, ""


class MarginUtilizationLimit(RiskLimit):
    """Reject opening orders when margin utilization is too high."""

    def __init__(self, max_ratio: Decimal = Decimal("0.8")) -> None:
        self._max = max_ratio

    @property
    def name(self) -> str:
        return "MarginUtilization"

    def check(self, request, context) -> tuple[bool, str]:
        if request.offset != Offset.OPEN:
            return True, ""
        if context.margin_ratio >= self._max:
            return False, f"Margin ratio {context.margin_ratio:.2%} exceeds {self._max:.2%}"
        return True, ""


class DailyLossLimit(RiskLimit):
    """Circuit-breaker halting trading when daily loss exceeds threshold."""

    def __init__(self, max_loss_pct: Decimal = Decimal("0.05")) -> None:
        self.max_loss_pct = max_loss_pct
        self._tripped = False

    @property
    def name(self) -> str:
        return "DailyLoss"

    def trip(self) -> None:
        self._tripped = True

    def reset(self) -> None:
        self._tripped = False

    @property
    def is_tripped(self) -> bool:
        return self._tripped

    def check(self, request, context) -> tuple[bool, str]:
        if self._tripped:
            return False, "Daily loss circuit-breaker tripped"
        return True, ""
