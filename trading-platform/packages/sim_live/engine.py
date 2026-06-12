"""Simulation matching engine: submit orders, match on ticks, optional latency."""

from __future__ import annotations

import asyncio
import heapq
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

from core.enums.order_status import OrderStatus
from core.models.order import Order
from core.models.tick import Tick

from .models import Fill, SimConfig
from .order_book import OrderBook, aggressor_is_buy


@dataclass(slots=True)
class SubmitOrderResult:
    """Outcome of ``submit_order`` (fills plus whether the order is still queued/resting)."""

    fills: list[Fill]
    queued: bool
    order: Order


def _is_market_order(order: Order) -> bool:
    """``price <= 0`` denotes a market order (``core.Order`` has no separate type field)."""
    return order.price <= Decimal(0)


class SimMatchingEngine:
    """
    Local matching engine driven by ``Tick`` snapshots.

    - Limit / market (``price <= 0``) supported via ``Direction`` / ``Offset`` → bid/ask mapping.
    - Simulated latency moves releases to ``tick.datetime``; optional wall-clock sleep.
    - Resting limits can be crossed by subsequent ticks via the order book's ``match``.
    """

    def __init__(self, config: SimConfig | None = None) -> None:
        self._config = config or SimConfig()
        self._books: dict[str, OrderBook] = {}
        self._pending: list[tuple[datetime, int, Order]] = []
        self._seq = 0
        self._last_tick: Tick | None = None

    @property
    def config(self) -> SimConfig:
        return self._config

    @property
    def last_tick(self) -> Tick | None:
        return self._last_tick

    def get_book(self, symbol: str) -> OrderBook:
        """Return the book for ``symbol`` (creates if missing)."""
        return self._book(symbol)

    def _book(self, symbol: str) -> OrderBook:
        if symbol not in self._books:
            self._books[symbol] = OrderBook(symbol)
        return self._books[symbol]

    def _latency_release_at(self, order: Order) -> datetime:
        base = self._last_tick.datetime if self._last_tick else order.created_at
        jitter = (
            random.randint(0, self._config.latency_jitter_ms)
            if self._config.latency_jitter_ms
            else 0
        )
        ms = self._config.latency_ms + jitter
        return base + timedelta(milliseconds=ms)

    def _ref_ask(self, tick: Tick) -> Decimal:
        return tick.ask_price1 if tick.ask_price1 is not None else tick.last_price

    def _ref_bid(self, tick: Tick) -> Decimal:
        return tick.bid_price1 if tick.bid_price1 is not None else tick.last_price

    def _slippage_cap_buy(self, ref_ask: Decimal) -> Decimal:
        return ref_ask + Decimal(self._config.slippage_ticks) * self._config.tick_size

    def _slippage_floor_sell(self, ref_bid: Decimal) -> Decimal:
        return ref_bid - Decimal(self._config.slippage_ticks) * self._config.tick_size

    async def submit_order(self, order: Order) -> SubmitOrderResult:
        """
        Submit an order. With zero sim latency and a known last tick, activates immediately.

        Returns fills from any immediate cross and whether the order remains delayed or resting.
        """
        if self._config.use_wall_clock_latency:
            wall_ms = self._config.latency_ms + (
                random.randint(0, self._config.latency_jitter_ms)
                if self._config.latency_jitter_ms
                else 0
            )
            await asyncio.sleep(wall_ms / 1000.0)

        order.status = OrderStatus.SUBMITTED
        fills: list[Fill] = []

        no_sim_delay = self._config.latency_ms == 0 and self._config.latency_jitter_ms == 0
        if (
            no_sim_delay
            and self._last_tick is not None
            and self._last_tick.symbol == order.symbol
        ):
            fills = await self._activate_order(order, self._last_tick)
            queued = self._order_is_queued(order)
            return SubmitOrderResult(fills=fills, queued=queued, order=order)

        order.status = OrderStatus.PENDING
        release_at = self._latency_release_at(order)
        heapq.heappush(self._pending, (release_at, self._seq, order))
        self._seq += 1
        return SubmitOrderResult(fills=[], queued=True, order=order)

    def _order_is_queued(self, order: Order) -> bool:
        if order.status == OrderStatus.CANCELLED:
            return False
        if _is_market_order(order):
            return False
        return order.remaining > 0

    async def _activate_order(self, order: Order, tick: Tick) -> list[Fill]:
        """Place or cross a single order using the snapshot ``tick``."""
        if order.symbol != tick.symbol:
            raise ValueError(f"Order symbol {order.symbol} != tick {tick.symbol}")
        book = self._book(order.symbol)
        dt = tick.datetime
        fills: list[Fill] = []
        buy = aggressor_is_buy(order)

        if _is_market_order(order):
            if buy:
                cap = self._slippage_cap_buy(self._ref_ask(tick))
                fills.extend(
                    book.consume_for_taker(
                        order,
                        buy=True,
                        price_cap=cap,
                        volume=order.remaining,
                        commission_rate=self._config.commission_rate,
                        contract_multiplier=self._config.contract_multiplier,
                        tick_time=dt,
                    )
                )
            else:
                floor = self._slippage_floor_sell(self._ref_bid(tick))
                fills.extend(
                    book.consume_for_taker(
                        order,
                        buy=False,
                        price_cap=floor,
                        volume=order.remaining,
                        commission_rate=self._config.commission_rate,
                        contract_multiplier=self._config.contract_multiplier,
                        tick_time=dt,
                    )
                )
            self._finalize_market_order(order)
            return fills

        if buy:
            ba = book.best_ask()
            if ba is not None and ba <= order.price:
                fills.extend(
                    book.consume_for_taker(
                        order,
                        buy=True,
                        price_cap=order.price,
                        volume=order.remaining,
                        commission_rate=self._config.commission_rate,
                        contract_multiplier=self._config.contract_multiplier,
                        tick_time=dt,
                    )
                )
        else:
            bb = book.best_bid()
            if bb is not None and bb >= order.price:
                fills.extend(
                    book.consume_for_taker(
                        order,
                        buy=False,
                        price_cap=order.price,
                        volume=order.remaining,
                        commission_rate=self._config.commission_rate,
                        contract_multiplier=self._config.contract_multiplier,
                        tick_time=dt,
                    )
                )

        if order.remaining > 0:
            book.add_order(order)
            order.status = (
                OrderStatus.PARTIAL_FILLED
                if order.filled_volume > 0
                else OrderStatus.SUBMITTED
            )
        elif order.filled_volume >= order.volume:
            order.status = OrderStatus.FILLED
        return fills

    def _finalize_market_order(self, order: Order) -> None:
        """IOC-style market: reject if no fill; cancel any leftover after partial."""
        if order.filled_volume == 0:
            order.status = OrderStatus.REJECTED
        elif order.remaining > 0:
            order.status = OrderStatus.CANCELLED
        else:
            order.status = OrderStatus.FILLED

    async def process_tick(self, tick: Tick) -> list[Fill]:
        """
        Advance simulation time, release delayed orders, match passive limits, return all fills.
        """
        self._last_tick = tick
        all_fills: list[Fill] = []
        dt = tick.datetime

        due_batch: list[tuple[datetime, Order]] = []
        while self._pending and self._pending[0][0] <= dt:
            release_at, _, order = heapq.heappop(self._pending)
            due_batch.append((release_at, order))

        for release_at, order in due_batch:
            if order.status in (
                OrderStatus.CANCELLED,
                OrderStatus.REJECTED,
                OrderStatus.FILLED,
            ):
                continue
            if order.symbol != tick.symbol:
                heapq.heappush(self._pending, (release_at, self._seq, order))
                self._seq += 1
                continue
            all_fills.extend(await self._activate_order(order, tick))

        all_fills.extend(self._passive_fills_from_tick(tick))
        return all_fills

    def _passive_fills_from_tick(self, tick: Tick) -> list[Fill]:
        """
        Resting limits crossed by the opposite side of the quoted spread.

        Requires at least one of ``bid_price1`` / ``ask_price1``; if both are absent, passive
        book updates are skipped (only ``last_price`` is ambiguous for two-sided matching).
        """
        book = self._book(tick.symbol)
        if tick.ask_price1 is None and tick.bid_price1 is None:
            return []

        out: list[Fill] = []
        if tick.ask_price1 is not None:
            out.extend(
                book.match(
                    tick.ask_price1,
                    10**12,
                    aggressor_buy=False,
                    commission_rate=self._config.commission_rate,
                    tick_time=tick.datetime,
                    contract_multiplier=self._config.contract_multiplier,
                )
            )
        if tick.bid_price1 is not None:
            out.extend(
                book.match(
                    tick.bid_price1,
                    10**12,
                    aggressor_buy=True,
                    commission_rate=self._config.commission_rate,
                    tick_time=tick.datetime,
                    contract_multiplier=self._config.contract_multiplier,
                )
            )
        return out

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel a resting order by id on the given symbol's book."""
        return self._book(symbol).cancel_order(order_id)
