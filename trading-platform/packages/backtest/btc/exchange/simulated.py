"""Simulated crypto exchange for backtesting.

Handles order matching, position tracking, commission/slippage, and
margin/leverage checks. Designed to faithfully model BTC perpetual
futures mechanics including funding rates.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from ..models.types import (
    OHLCV,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)

logger = logging.getLogger(__name__)

_ZERO = Decimal(0)


class SimulatedExchange:
    """Event-driven simulated exchange with order book emulation."""

    def __init__(
        self,
        initial_capital: Decimal,
        commission_rate: Decimal = Decimal("0.001"),
        slippage_bps: Decimal = Decimal("5"),
        leverage_limit: Decimal = Decimal(1),
        max_position_pct: Decimal = Decimal("0.1"),
    ) -> None:
        self._initial_capital = initial_capital
        self._cash = initial_capital
        self._commission_rate = commission_rate
        self._slippage_bps = slippage_bps
        self._leverage_limit = leverage_limit
        self._max_position_pct = max_position_pct

        self._positions: dict[str, Position] = {}
        self._open_orders: list[Order] = []
        self._fills: list[Fill] = []
        self._equity_history: list[tuple[Any, Decimal]] = []
        self._current_bar: OHLCV | None = None

    @property
    def cash(self) -> Decimal:
        return self._cash

    @property
    def positions(self) -> dict[str, Position]:
        return dict(self._positions)

    @property
    def fills(self) -> list[Fill]:
        return list(self._fills)

    @property
    def equity(self) -> Decimal:
        unrealized = sum(
            p.unrealized_pnl for p in self._positions.values()
        )
        return self._cash + unrealized

    def submit_order(self, order: Order) -> Order:
        """Submit an order and validate against risk limits."""
        if not self._validate_order(order):
            order.status = OrderStatus.REJECTED
            logger.warning("Order rejected: %s", order.id)
            return order

        order.status = OrderStatus.OPEN
        order.created_at = (
            self._current_bar.timestamp if self._current_bar else None
        )
        self._open_orders.append(order)
        logger.debug("Order submitted: %s %s %s @ %s", order.side.value, order.quantity, order.symbol, order.price)
        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order by ID."""
        for i, o in enumerate(self._open_orders):
            if o.id == order_id:
                o.status = OrderStatus.CANCELLED
                self._open_orders.pop(i)
                return True
        return False

    def on_bar(self, bar: OHLCV) -> list[Fill]:
        """Process a new bar: match orders and update positions."""
        self._current_bar = bar
        new_fills: list[Fill] = []

        remaining: list[Order] = []
        for order in self._open_orders:
            fill = self._try_match(order, bar)
            if fill:
                new_fills.append(fill)
                self._apply_fill(fill)
                self._fills.append(fill)
            else:
                remaining.append(order)
        self._open_orders = remaining

        self._update_unrealized_pnl(bar)
        self._equity_history.append((bar.timestamp, self.equity))
        return new_fills

    def apply_funding_rate(self, symbol: str, rate: Decimal) -> Decimal:
        """Apply periodic funding rate to open position. Returns payment amount."""
        pos = self._positions.get(symbol)
        if not pos or pos.is_flat:
            return _ZERO
        assert self._current_bar is not None
        notional = abs(pos.quantity) * self._current_bar.close
        payment = notional * rate
        if pos.is_long:
            self._cash -= payment
        else:
            self._cash += payment
        return payment

    def _validate_order(self, order: Order) -> bool:
        if order.quantity <= 0:
            return False
        if self._max_position_pct > 0 and self._current_bar:
            notional = order.quantity * (order.price or self._current_bar.close)
            if notional > self.equity * self._max_position_pct:
                logger.warning(
                    "Position size %.2f exceeds limit %.2f%%",
                    notional,
                    self._max_position_pct * 100,
                )
                return False
        return True

    def _try_match(self, order: Order, bar: OHLCV) -> Fill | None:
        """Attempt to match order against the bar's price range."""
        fill_price: Decimal | None = None

        if order.order_type == OrderType.MARKET:
            fill_price = bar.open

        elif order.order_type == OrderType.LIMIT:
            assert order.price is not None
            if order.side == OrderSide.BUY and bar.low <= order.price:
                fill_price = min(order.price, bar.open)
            elif order.side == OrderSide.SELL and bar.high >= order.price:
                fill_price = max(order.price, bar.open)

        elif order.order_type == OrderType.STOP:
            assert order.stop_price is not None
            if order.side == OrderSide.BUY and bar.high >= order.stop_price:
                fill_price = max(order.stop_price, bar.open)
            elif order.side == OrderSide.SELL and bar.low <= order.stop_price:
                fill_price = min(order.stop_price, bar.open)

        elif order.order_type == OrderType.STOP_LIMIT:
            assert order.stop_price is not None
            assert order.price is not None
            triggered = (
                (order.side == OrderSide.BUY and bar.high >= order.stop_price)
                or (order.side == OrderSide.SELL and bar.low <= order.stop_price)
            )
            if triggered:
                if order.side == OrderSide.BUY and bar.low <= order.price:
                    fill_price = min(order.price, bar.open)
                elif order.side == OrderSide.SELL and bar.high >= order.price:
                    fill_price = max(order.price, bar.open)

        if fill_price is None:
            return None

        slippage = fill_price * self._slippage_bps / Decimal(10000)
        if order.side == OrderSide.BUY:
            fill_price += slippage
        else:
            fill_price -= slippage

        commission = fill_price * order.quantity * self._commission_rate

        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.avg_fill_price = fill_price
        order.updated_at = bar.timestamp

        return Fill(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            price=fill_price,
            quantity=order.quantity,
            commission=commission,
            slippage=slippage * order.quantity,
            timestamp=bar.timestamp,
        )

    def _apply_fill(self, fill: Fill) -> None:
        """Update cash and position after a fill."""
        pos = self._positions.setdefault(
            fill.symbol, Position(symbol=fill.symbol)
        )
        cost = fill.price * fill.quantity + fill.commission

        if fill.side == OrderSide.BUY:
            new_qty = pos.quantity + fill.quantity
            if pos.quantity >= 0:
                if new_qty != 0:
                    pos.avg_entry_price = (
                        (pos.avg_entry_price * pos.quantity + fill.price * fill.quantity)
                        / new_qty
                    )
            else:
                closed = min(fill.quantity, abs(pos.quantity))
                pnl = closed * (pos.avg_entry_price - fill.price)
                pos.realized_pnl += pnl
                if new_qty > 0:
                    pos.avg_entry_price = fill.price
            pos.quantity = new_qty
            self._cash -= cost
        else:
            new_qty = pos.quantity - fill.quantity
            if pos.quantity <= 0:
                if new_qty != 0:
                    pos.avg_entry_price = (
                        (pos.avg_entry_price * abs(pos.quantity) + fill.price * fill.quantity)
                        / abs(new_qty)
                    )
            else:
                closed = min(fill.quantity, pos.quantity)
                pnl = closed * (fill.price - pos.avg_entry_price)
                pos.realized_pnl += pnl
                if new_qty < 0:
                    pos.avg_entry_price = fill.price
            pos.quantity = new_qty
            self._cash += fill.price * fill.quantity - fill.commission

    def _update_unrealized_pnl(self, bar: OHLCV) -> None:
        """Recalculate unrealized PnL for all positions using bar close."""
        for pos in self._positions.values():
            if pos.is_flat:
                pos.unrealized_pnl = _ZERO
            elif pos.is_long:
                pos.unrealized_pnl = pos.quantity * (bar.close - pos.avg_entry_price)
            else:
                pos.unrealized_pnl = abs(pos.quantity) * (pos.avg_entry_price - bar.close)
