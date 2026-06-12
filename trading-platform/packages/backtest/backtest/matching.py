"""撮合引擎 - 模拟交易所撮合逻辑。"""

from __future__ import annotations

import logging
from decimal import Decimal
from uuid import uuid4

from .events import Event, EventBus, EventType
from .models import (
    Bar,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Trade,
)

logger = logging.getLogger(__name__)


class MatchingEngine:
    """
    回测撮合引擎。

    接收订单并根据当前行情模拟撮合，支持：
    - 市价单：按下一根Bar的开盘价 + 滑点成交
    - 限价单：当价格触及时成交
    - 止损单 / 止损限价单
    - 手续费计算
    - 持仓更新
    """

    def __init__(
        self,
        event_bus: EventBus,
        commission_rate: Decimal = Decimal("0.0001"),
        slippage_ticks: int = 1,
        tick_size: Decimal = Decimal("1"),
        contract_multiplier: int = 1,
    ) -> None:
        self._event_bus = event_bus
        self._commission_rate = commission_rate
        self._slippage_ticks = slippage_ticks
        self._tick_size = tick_size
        self._contract_multiplier = contract_multiplier

        self._pending_orders: list[Order] = []
        self._positions: dict[str, Position] = {}
        self._trades: list[Trade] = []

        self._event_bus.subscribe(EventType.BAR, self._on_bar)

    @property
    def positions(self) -> dict[str, Position]:
        return self._positions

    @property
    def trades(self) -> list[Trade]:
        return list(self._trades)

    def submit_order(self, order: Order) -> None:
        """提交订单到撮合队列。"""
        order.status = OrderStatus.SUBMITTED
        self._pending_orders.append(order)
        self._event_bus.publish(
            Event(type=EventType.ORDER_SUBMITTED, data=order, dt=order.created_at, source="matching")
        )
        logger.info("Order submitted: %s %s %s vol=%d", order.symbol, order.side.value, order.order_type.value, order.volume)

    def cancel_order(self, order: Order) -> bool:
        """取消订单。"""
        if order in self._pending_orders and order.is_active:
            order.status = OrderStatus.CANCELLED
            self._pending_orders.remove(order)
            self._event_bus.publish(
                Event(type=EventType.ORDER_CANCELLED, data=order, source="matching")
            )
            return True
        return False

    def _on_bar(self, event: Event) -> None:
        bar: Bar = event.data
        still_active: list[Order] = []

        for order in self._pending_orders:
            if order.symbol != bar.symbol:
                still_active.append(order)
                continue

            fill_price = self._try_fill(order, bar)
            if fill_price is not None:
                self._execute_fill(order, fill_price, order.remaining_volume, bar)

            if order.is_active:
                still_active.append(order)

        self._pending_orders = still_active

    def _try_fill(self, order: Order, bar: Bar) -> Decimal | None:
        """根据订单类型和行情判断是否可以成交，返回成交价格。"""
        if order.order_type == OrderType.MARKET:
            slippage = self._tick_size * self._slippage_ticks
            if order.side == OrderSide.BUY:
                return bar.open + slippage
            return bar.open - slippage

        if order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and bar.low <= order.price:
                return min(order.price, bar.open)
            if order.side == OrderSide.SELL and bar.high >= order.price:
                return max(order.price, bar.open)

        if order.order_type == OrderType.STOP:
            if order.side == OrderSide.BUY and bar.high >= order.price:
                slippage = self._tick_size * self._slippage_ticks
                return max(order.price, bar.open) + slippage
            if order.side == OrderSide.SELL and bar.low <= order.price:
                slippage = self._tick_size * self._slippage_ticks
                return min(order.price, bar.open) - slippage

        if order.order_type == OrderType.STOP_LIMIT:
            stop_price = order.price
            limit_price = order.limit_price if order.limit_price is not None else order.price
            if order.side == OrderSide.BUY and bar.high >= stop_price:
                fill = max(stop_price, bar.open)
                if fill <= limit_price:
                    return fill
            if order.side == OrderSide.SELL and bar.low <= stop_price:
                fill = min(stop_price, bar.open)
                if fill >= limit_price:
                    return fill

        return None

    def _execute_fill(self, order: Order, price: Decimal, volume: int, bar: Bar) -> None:
        """执行成交。"""
        commission = price * volume * self._contract_multiplier * self._commission_rate
        slippage_cost = abs(price - bar.open) * volume * self._contract_multiplier

        trade = Trade(
            id=uuid4(),
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            price=price,
            volume=volume,
            commission=commission,
            slippage=slippage_cost,
            dt=bar.dt,
        )
        self._trades.append(trade)

        prev_cost = order.avg_fill_price * order.filled_volume
        order.filled_volume += volume
        order.avg_fill_price = (prev_cost + price * volume) / order.filled_volume if order.filled_volume else Decimal(0)

        if order.remaining_volume == 0:
            order.status = OrderStatus.FILLED
            event_type = EventType.ORDER_FILLED
        else:
            order.status = OrderStatus.PARTIAL_FILLED
            event_type = EventType.ORDER_PARTIAL_FILLED

        order.updated_at = bar.dt

        self._update_position(trade)

        self._event_bus.publish(Event(type=EventType.TRADE, data=trade, dt=bar.dt, source="matching"))
        self._event_bus.publish(Event(type=event_type, data=order, dt=bar.dt, source="matching"))

    def _update_position(self, trade: Trade) -> None:
        """根据成交更新持仓。"""
        pos = self._positions.get(trade.symbol)
        if pos is None:
            pos = Position(symbol=trade.symbol)
            self._positions[trade.symbol] = pos

        multiplier = self._contract_multiplier

        if trade.side == OrderSide.BUY:
            if pos.short_volume > 0:
                close_vol = min(trade.volume, pos.short_volume)
                pnl = (pos.short_avg_price - trade.price) * close_vol * multiplier
                pos.realized_pnl += pnl
                pos.short_volume -= close_vol
                remaining = trade.volume - close_vol
                if remaining > 0:
                    total_cost = pos.long_avg_price * pos.long_volume + trade.price * remaining
                    pos.long_volume += remaining
                    pos.long_avg_price = total_cost / pos.long_volume if pos.long_volume else Decimal(0)
            else:
                total_cost = pos.long_avg_price * pos.long_volume + trade.price * trade.volume
                pos.long_volume += trade.volume
                pos.long_avg_price = total_cost / pos.long_volume if pos.long_volume else Decimal(0)
        else:
            if pos.long_volume > 0:
                close_vol = min(trade.volume, pos.long_volume)
                pnl = (trade.price - pos.long_avg_price) * close_vol * multiplier
                pos.realized_pnl += pnl
                pos.long_volume -= close_vol
                remaining = trade.volume - close_vol
                if remaining > 0:
                    total_cost = pos.short_avg_price * pos.short_volume + trade.price * remaining
                    pos.short_volume += remaining
                    pos.short_avg_price = total_cost / pos.short_volume if pos.short_volume else Decimal(0)
            else:
                total_cost = pos.short_avg_price * pos.short_volume + trade.price * trade.volume
                pos.short_volume += trade.volume
                pos.short_avg_price = total_cost / pos.short_volume if pos.short_volume else Decimal(0)

        self._event_bus.publish(
            Event(type=EventType.POSITION_UPDATE, data=pos, dt=trade.dt, source="matching")
        )

    def mark_to_market(self, symbol: str, price: Decimal) -> None:
        """按当前价格更新未实现盈亏。"""
        pos = self._positions.get(symbol)
        if pos is None:
            return
        multiplier = self._contract_multiplier
        pos.unrealized_pnl = (
            (price - pos.long_avg_price) * pos.long_volume * multiplier
            - (price - pos.short_avg_price) * pos.short_volume * multiplier
        )
