"""Price-time priority order book for simulated matching."""

from __future__ import annotations

import bisect
from collections import deque
from datetime import datetime
from decimal import Decimal

from core.enums.direction import Direction, Offset
from core.enums.order_status import OrderStatus
from core.models.order import Order

from .models import Fill


def aggressor_is_buy(order: Order) -> bool:
    """Map futures direction/offset to bid (buy) vs ask (sell) aggression."""
    if order.direction == Direction.LONG and order.offset == Offset.OPEN:
        return True
    if order.direction == Direction.SHORT and order.offset in (
        Offset.CLOSE,
        Offset.CLOSE_TODAY,
    ):
        return True
    return False


class OrderBook:
    """
    Simple limit-order book with FIFO queues at each price level.

    Bid prices are stored ascending (best bid = last); ask prices ascending (best = first).
    ``match(price, volume, aggressor_buy=...)`` walks the opposite side and returns fills.
    """

    def __init__(self, symbol: str) -> None:
        self._symbol = symbol
        self._bids: dict[Decimal, deque[Order]] = {}
        self._bid_prices_asc: list[Decimal] = []
        self._asks: dict[Decimal, deque[Order]] = {}
        self._ask_prices_asc: list[Decimal] = []

    @property
    def symbol(self) -> str:
        return self._symbol

    def best_bid(self) -> Decimal | None:
        return self._bid_prices_asc[-1] if self._bid_prices_asc else None

    def best_ask(self) -> Decimal | None:
        return self._ask_prices_asc[0] if self._ask_prices_asc else None

    def _insert_bid_level(self, price: Decimal) -> None:
        if price not in self._bids:
            self._bids[price] = deque()
            bisect.insort(self._bid_prices_asc, price)

    def _insert_ask_level(self, price: Decimal) -> None:
        if price not in self._asks:
            self._asks[price] = deque()
            bisect.insort(self._ask_prices_asc, price)

    def add_order(self, order: Order) -> None:
        """Insert a resting limit order (no immediate cross). Caller must filter."""
        if order.symbol != self._symbol:
            raise ValueError(f"Order symbol {order.symbol} != book {self._symbol}")
        price = order.price
        if aggressor_is_buy(order):
            self._insert_bid_level(price)
            self._bids[price].append(order)
        else:
            self._insert_ask_level(price)
            self._asks[price].append(order)

    def cancel_order(self, order_id: str) -> bool:
        """Remove a resting order by id. Returns True if removed."""
        for price, dq in list(self._bids.items()):
            for i, o in enumerate(dq):
                if o.order_id == order_id:
                    del dq[i]
                    o.status = OrderStatus.CANCELLED
                    if not dq:
                        del self._bids[price]
                        self._bid_prices_asc.remove(price)
                    return True
        for price, dq in list(self._asks.items()):
            for i, o in enumerate(dq):
                if o.order_id == order_id:
                    del dq[i]
                    o.status = OrderStatus.CANCELLED
                    if not dq:
                        del self._asks[price]
                        self._ask_prices_asc.remove(price)
                    return True
        return False

    def match(
        self,
        price: Decimal,
        volume: int,
        *,
        aggressor_buy: bool,
        commission_rate: Decimal,
        tick_time: datetime,
        contract_multiplier: int = 1,
    ) -> list[Fill]:
        """
        Consume up to ``volume`` on the opposite side.

        Buy aggressor: ``price`` is max pay; asks at or below fill first.
        Sell aggressor: ``price`` is min receive; bids at or above fill first.
        """
        fills: list[Fill] = []
        remaining = volume
        if remaining <= 0:
            return fills

        if aggressor_buy:
            while remaining > 0 and self._ask_prices_asc:
                level = self._ask_prices_asc[0]
                if level > price:
                    break
                dq = self._asks[level]
                while remaining > 0 and dq:
                    ro = dq[0]
                    take = min(remaining, ro.remaining)
                    comm = level * take * contract_multiplier * commission_rate
                    ro.filled_volume += take
                    remaining -= take
                    if ro.avg_fill_price is None:
                        ro.avg_fill_price = level
                    else:
                        prev_v = ro.filled_volume - take
                        ro.avg_fill_price = (
                            ro.avg_fill_price * Decimal(prev_v) + level * Decimal(take)
                        ) / Decimal(ro.filled_volume)
                    ro.updated_at = tick_time
                    if ro.filled_volume >= ro.volume:
                        ro.status = OrderStatus.FILLED
                        dq.popleft()
                    else:
                        ro.status = OrderStatus.PARTIAL_FILLED
                    fills.append(
                        Fill(
                            order_id=ro.order_id,
                            symbol=self._symbol,
                            price=level,
                            volume=take,
                            commission=comm,
                            timestamp=tick_time,
                            aggressor_buy=True,
                        )
                    )
                if not dq:
                    del self._asks[level]
                    self._ask_prices_asc.pop(0)
        else:
            while remaining > 0 and self._bid_prices_asc:
                level = self._bid_prices_asc[-1]
                if level < price:
                    break
                dq = self._bids[level]
                while remaining > 0 and dq:
                    ro = dq[0]
                    take = min(remaining, ro.remaining)
                    comm = level * take * contract_multiplier * commission_rate
                    ro.filled_volume += take
                    remaining -= take
                    if ro.avg_fill_price is None:
                        ro.avg_fill_price = level
                    else:
                        prev_v = ro.filled_volume - take
                        ro.avg_fill_price = (
                            ro.avg_fill_price * Decimal(prev_v) + level * Decimal(take)
                        ) / Decimal(ro.filled_volume)
                    ro.updated_at = tick_time
                    if ro.filled_volume >= ro.volume:
                        ro.status = OrderStatus.FILLED
                        dq.popleft()
                    else:
                        ro.status = OrderStatus.PARTIAL_FILLED
                    fills.append(
                        Fill(
                            order_id=ro.order_id,
                            symbol=self._symbol,
                            price=level,
                            volume=take,
                            commission=comm,
                            timestamp=tick_time,
                            aggressor_buy=False,
                        )
                    )
                if not dq:
                    del self._bids[level]
                    self._bid_prices_asc.pop()

        return fills

    def consume_for_taker(
        self,
        taker: Order,
        *,
        buy: bool,
        price_cap: Decimal,
        volume: int,
        commission_rate: Decimal,
        contract_multiplier: int,
        tick_time: datetime,
    ) -> list[Fill]:
        """
        Aggressor (taker) walks the opposite side up to ``price_cap``.

        Buy taker: take asks with price <= price_cap. Sell taker: take bids with price >= price_cap.
        Each ``Fill`` uses ``taker.order_id``; resting makers are reduced or removed.
        """
        fills: list[Fill] = []
        remaining = min(volume, taker.remaining)
        if remaining <= 0:
            return fills

        if buy:
            while remaining > 0 and self._ask_prices_asc:
                level = self._ask_prices_asc[0]
                if level > price_cap:
                    break
                dq = self._asks[level]
                while remaining > 0 and dq:
                    maker = dq[0]
                    take = min(remaining, maker.remaining)
                    maker.filled_volume += take
                    remaining -= take
                    taker.filled_volume += take
                    comm = level * take * contract_multiplier * commission_rate
                    maker.updated_at = tick_time
                    if maker.avg_fill_price is None:
                        maker.avg_fill_price = level
                    else:
                        pv = maker.filled_volume - take
                        maker.avg_fill_price = (
                            maker.avg_fill_price * Decimal(pv) + level * Decimal(take)
                        ) / Decimal(maker.filled_volume)
                    if maker.filled_volume >= maker.volume:
                        maker.status = OrderStatus.FILLED
                        dq.popleft()
                    else:
                        maker.status = OrderStatus.PARTIAL_FILLED
                    fills.append(
                        Fill(
                            order_id=taker.order_id,
                            symbol=self._symbol,
                            price=level,
                            volume=take,
                            commission=comm,
                            timestamp=tick_time,
                            aggressor_buy=True,
                        )
                    )
                if not dq:
                    del self._asks[level]
                    self._ask_prices_asc.pop(0)
        else:
            while remaining > 0 and self._bid_prices_asc:
                level = self._bid_prices_asc[-1]
                if level < price_cap:
                    break
                dq = self._bids[level]
                while remaining > 0 and dq:
                    maker = dq[0]
                    take = min(remaining, maker.remaining)
                    maker.filled_volume += take
                    remaining -= take
                    taker.filled_volume += take
                    comm = level * take * contract_multiplier * commission_rate
                    maker.updated_at = tick_time
                    if maker.avg_fill_price is None:
                        maker.avg_fill_price = level
                    else:
                        pv = maker.filled_volume - take
                        maker.avg_fill_price = (
                            maker.avg_fill_price * Decimal(pv) + level * Decimal(take)
                        ) / Decimal(maker.filled_volume)
                    if maker.filled_volume >= maker.volume:
                        maker.status = OrderStatus.FILLED
                        dq.popleft()
                    else:
                        maker.status = OrderStatus.PARTIAL_FILLED
                    fills.append(
                        Fill(
                            order_id=taker.order_id,
                            symbol=self._symbol,
                            price=level,
                            volume=take,
                            commission=comm,
                            timestamp=tick_time,
                            aggressor_buy=False,
                        )
                    )
                if not dq:
                    del self._bids[level]
                    self._bid_prices_asc.pop()

        if fills:
            _update_taker_vwap(taker, fills, tick_time)
        if taker.filled_volume >= taker.volume:
            taker.status = OrderStatus.FILLED
        elif taker.filled_volume > 0:
            taker.status = OrderStatus.PARTIAL_FILLED
        return fills

    def resting_orders_snapshot(self) -> list[Order]:
        """Copy of resting orders for inspection (debug / shadow)."""
        out: list[Order] = []
        for p in reversed(self._bid_prices_asc):
            out.extend(list(self._bids[p]))
        for p in self._ask_prices_asc:
            out.extend(list(self._asks[p]))
        return out


def _update_taker_vwap(taker: Order, new_fills: list[Fill], tick_time: datetime) -> None:
    """Merge new fill slices into taker ``avg_fill_price``."""
    if not new_fills:
        return
    add_notional = sum(f.price * Decimal(f.volume) for f in new_fills)
    add_vol = sum(f.volume for f in new_fills)
    prev_filled = taker.filled_volume - add_vol
    if prev_filled <= 0:
        taker.avg_fill_price = add_notional / Decimal(add_vol) if add_vol else None
    else:
        assert taker.avg_fill_price is not None
        taker.avg_fill_price = (
            taker.avg_fill_price * Decimal(prev_filled) + add_notional
        ) / Decimal(taker.filled_volume)
    taker.updated_at = tick_time
