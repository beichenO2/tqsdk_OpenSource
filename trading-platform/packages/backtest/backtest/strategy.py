"""策略抽象基类 - 用户策略需要继承此类。"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import BacktestEngine
    from .models import Bar, Order, Position, Trade


class Strategy(abc.ABC):
    """回测策略抽象基类。"""

    def __init__(self) -> None:
        self._engine: BacktestEngine | None = None

    def bind(self, engine: BacktestEngine) -> None:
        self._engine = engine

    @property
    def engine(self) -> BacktestEngine:
        if self._engine is None:
            raise RuntimeError("Strategy not bound to engine")
        return self._engine

    def buy(self, symbol: str, volume: int, price: float | None = None) -> Order:
        """买入开仓。"""
        from decimal import Decimal
        from .models import Order, OrderSide, OrderType

        order = Order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT if price else OrderType.MARKET,
            price=Decimal(str(price)) if price else Decimal(0),
            volume=volume,
            strategy_id=self.__class__.__name__,
        )
        self.engine.submit_order(order)
        return order

    def sell(self, symbol: str, volume: int, price: float | None = None) -> Order:
        """卖出平仓。"""
        from decimal import Decimal
        from .models import Order, OrderSide, OrderType

        order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT if price else OrderType.MARKET,
            price=Decimal(str(price)) if price else Decimal(0),
            volume=volume,
            strategy_id=self.__class__.__name__,
        )
        self.engine.submit_order(order)
        return order

    def get_position(self, symbol: str) -> Position | None:
        """获取持仓。"""
        return self.engine.get_position(symbol)

    @abc.abstractmethod
    def on_bar(self, bar: Bar) -> None:
        """K线回调 - 每根新K线到达时调用。"""

    def on_trade(self, trade: Trade) -> None:
        """成交回调（可选）。"""

    def on_init(self) -> None:
        """初始化回调（可选）。"""

    def on_stop(self) -> None:
        """停止回调（可选）。"""
