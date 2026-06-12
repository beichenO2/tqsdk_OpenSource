"""BTC-specific limit order book simulator for backtest matching.

Provides a simplified LOB that can be seeded from OHLCV bars or tick
snapshots. Designed for T34-2: 24-hour crypto order book simulation.

Key differences from the standard futures matching engine:
- No session breaks (crypto trades 24/7/365)
- Decimal quantities (fractional BTC)
- Configurable depth and liquidity profiles
"""

from __future__ import annotations

import bisect
import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any

_ZERO = Decimal(0)

logger = logging.getLogger(__name__)


class Side(str, Enum):
    BID = "bid"
    ASK = "ask"


@dataclass
class LimitLevel:
    """Single price level in the book."""
    price: Decimal
    orders: deque[BookOrder] = field(default_factory=deque)

    @property
    def total_qty(self) -> Decimal:
        return sum(o.remaining for o in self.orders)

    def __lt__(self, other: LimitLevel) -> bool:
        return self.price < other.price


@dataclass
class BookOrder:
    """An order resting on the book."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    side: Side = Side.BID
    price: Decimal = _ZERO
    quantity: Decimal = _ZERO
    remaining: Decimal = _ZERO
    timestamp: Any = None

    def __post_init__(self) -> None:
        if self.remaining == _ZERO:
            self.remaining = self.quantity


@dataclass
class MatchResult:
    """Result of matching an aggressive order against the book."""
    fills: list[BookFill] = field(default_factory=list)
    remaining_qty: Decimal = _ZERO

    @property
    def filled_qty(self) -> Decimal:
        return sum(f.quantity for f in self.fills)

    @property
    def avg_price(self) -> Decimal:
        if not self.fills:
            return _ZERO
        total_notional = sum(f.price * f.quantity for f in self.fills)
        total_qty = self.filled_qty
        return total_notional / total_qty if total_qty else _ZERO


@dataclass(frozen=True)
class BookFill:
    """A single fill from book matching."""
    price: Decimal
    quantity: Decimal
    maker_order_id: str
    aggressor_is_buy: bool


class CryptoOrderBook:
    """Simple limit order book for BTC backtesting.

    Supports:
    - Add / cancel passive orders
    - Aggressive matching (market orders eat through the book)
    - Synthetic depth generation from OHLCV bars
    - 24-hour operation (no session boundaries)
    """

    def __init__(
        self,
        symbol: str,
        tick_size: Decimal = Decimal("0.01"),
        lot_size: Decimal = Decimal("0.00001"),
        max_depth_levels: int = 50,
    ) -> None:
        self._symbol = symbol
        self._tick_size = tick_size
        self._lot_size = lot_size
        self._max_depth = max_depth_levels

        self._bids: list[LimitLevel] = []
        self._asks: list[LimitLevel] = []

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def best_bid(self) -> Decimal | None:
        return self._bids[-1].price if self._bids else None

    @property
    def best_ask(self) -> Decimal | None:
        return self._asks[0].price if self._asks else None

    @property
    def spread(self) -> Decimal | None:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    @property
    def mid_price(self) -> Decimal | None:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None

    def add_order(self, order: BookOrder) -> None:
        """Place a passive order on the book (no crossing)."""
        levels = self._bids if order.side == Side.BID else self._asks
        target_level: LimitLevel | None = None
        for lv in levels:
            if lv.price == order.price:
                target_level = lv
                break

        if target_level is None:
            target_level = LimitLevel(price=order.price)
            if order.side == Side.BID:
                bisect.insort(self._bids, target_level)
            else:
                bisect.insort(self._asks, target_level)

        target_level.orders.append(order)
        self._trim_depth()

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a resting order by ID."""
        for levels in (self._bids, self._asks):
            for lv in levels:
                for i, o in enumerate(lv.orders):
                    if o.id == order_id:
                        lv.orders.remove(o)
                        if not lv.orders:
                            levels.remove(lv)
                        return True
        return False

    def match_market(
        self,
        quantity: Decimal,
        is_buy: bool,
    ) -> MatchResult:
        """Aggressively match *quantity* against the book.

        A buy eats asks (ascending); a sell eats bids (descending).
        Returns fills and any unfilled remainder.
        """
        levels = self._asks if is_buy else list(reversed(self._bids))
        fills: list[BookFill] = []
        remaining = quantity

        levels_to_remove: list[LimitLevel] = []

        for lv in levels:
            if remaining <= 0:
                break
            while lv.orders and remaining > 0:
                head = lv.orders[0]
                fill_qty = min(remaining, head.remaining)
                fills.append(BookFill(
                    price=head.price,
                    quantity=fill_qty,
                    maker_order_id=head.id,
                    aggressor_is_buy=is_buy,
                ))
                head.remaining -= fill_qty
                remaining -= fill_qty
                if head.remaining <= 0:
                    lv.orders.popleft()
            if not lv.orders:
                levels_to_remove.append(lv)

        for lv in levels_to_remove:
            if lv in self._asks:
                self._asks.remove(lv)
            if lv in self._bids:
                self._bids.remove(lv)

        return MatchResult(fills=fills, remaining_qty=remaining)

    def seed_from_bar(
        self,
        mid_price: Decimal,
        bar_volume: Decimal,
        spread_bps: Decimal = Decimal("5"),
        depth_levels: int = 20,
        decay: Decimal = Decimal("0.85"),
    ) -> None:
        """Generate synthetic book depth from a single OHLCV bar.

        Distributes volume across *depth_levels* on each side using
        exponential decay away from the mid price.
        """
        self._bids.clear()
        self._asks.clear()

        half_spread = mid_price * spread_bps / Decimal(20000)
        best_bid = mid_price - half_spread
        best_ask = mid_price + half_spread

        per_side_volume = bar_volume / 4

        for i in range(depth_levels):
            weight = decay ** i
            level_qty = per_side_volume * weight / Decimal(depth_levels)
            level_qty = max(level_qty, self._lot_size)

            bid_px = best_bid - self._tick_size * i
            ask_px = best_ask + self._tick_size * i

            self.add_order(BookOrder(
                side=Side.BID, price=bid_px,
                quantity=level_qty, remaining=level_qty,
            ))
            self.add_order(BookOrder(
                side=Side.ASK, price=ask_px,
                quantity=level_qty, remaining=level_qty,
            ))

    def snapshot(self, depth: int = 5) -> dict[str, list[tuple[Decimal, Decimal]]]:
        """Return top *depth* price levels for each side."""
        bids = [(lv.price, lv.total_qty) for lv in reversed(self._bids[-depth:])]
        asks = [(lv.price, lv.total_qty) for lv in self._asks[:depth]]
        return {"bids": bids, "asks": asks}

    def _trim_depth(self) -> None:
        if len(self._bids) > self._max_depth:
            self._bids = self._bids[-self._max_depth:]
        if len(self._asks) > self._max_depth:
            self._asks = self._asks[:self._max_depth]
