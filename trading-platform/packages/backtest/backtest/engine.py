"""回测引擎核心 - 事件驱动架构，串联数据、撮合、策略、报告。"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from decimal import Decimal

from .datafeed import DataFeed
from .events import Event, EventBus, EventType
from .matching import MatchingEngine
from .models import (
    BacktestConfig,
    BacktestResult,
    Bar,
    EquityCurvePoint,
    Order,
    Position,
)
from .report import ReportGenerator
from .strategy import Strategy

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    事件驱动回测引擎。

    流程:
    1. 配置引擎参数
    2. 注册策略
    3. 加载数据
    4. 逐Bar驱动: DataFeed -> EventBus -> MatchingEngine -> Strategy
    5. 生成报告
    """

    def __init__(self, config: BacktestConfig) -> None:
        self._config = config
        self._event_bus = EventBus()
        self._matching = MatchingEngine(
            event_bus=self._event_bus,
            commission_rate=config.commission_rate,
            slippage_ticks=config.slippage_ticks,
            tick_size=config.tick_size,
            contract_multiplier=config.contract_multiplier,
        )
        self._report_gen = ReportGenerator()
        self._datafeed: DataFeed | None = None
        self._strategy: Strategy | None = None

        self._cash = config.initial_capital
        self._equity_curve: list[EquityCurvePoint] = []
        self._current_bar: dict[str, Bar] = {}

        self._event_bus.subscribe(EventType.TRADE, self._on_trade)
        self._event_bus.subscribe(EventType.BAR, self._on_bar_equity)

    @property
    def event_bus(self) -> EventBus:
        return self._event_bus

    @property
    def config(self) -> BacktestConfig:
        return self._config

    @property
    def cash(self) -> Decimal:
        return self._cash

    def set_datafeed(self, datafeed: DataFeed) -> None:
        self._datafeed = datafeed

    def set_strategy(self, strategy: Strategy) -> None:
        self._strategy = strategy
        strategy.bind(self)
        self._event_bus.subscribe(EventType.BAR, self._on_bar_strategy)

    def submit_order(self, order: Order) -> None:
        self._matching.submit_order(order)

    def cancel_order(self, order: Order) -> bool:
        return self._matching.cancel_order(order)

    def get_position(self, symbol: str) -> Position | None:
        return self._matching.positions.get(symbol)

    def get_all_positions(self) -> dict[str, Position]:
        return dict(self._matching.positions)

    def run(self) -> BacktestResult:
        """执行回测。"""
        if self._datafeed is None:
            raise RuntimeError("DataFeed not set")
        if self._strategy is None:
            raise RuntimeError("Strategy not set")

        start_time = datetime.now()
        wall_start = time.monotonic()

        logger.info(
            "Backtest starting: strategy=%s symbols=%s period=%s~%s",
            self._config.strategy_id,
            self._config.symbols,
            self._config.start_date,
            self._config.end_date,
        )

        self._event_bus.publish(Event(type=EventType.ENGINE_START, source="engine"))
        self._strategy.on_init()

        if self._config.start_date and self._config.end_date:
            self._datafeed.load(self._config.symbols, self._config.start_date, self._config.end_date)

        bar_count = 0
        for _bar in self._datafeed:
            bar_count += 1

        self._strategy.on_stop()
        self._event_bus.publish(Event(type=EventType.ENGINE_STOP, source="engine"))

        end_time = datetime.now()
        wall_elapsed = time.monotonic() - wall_start

        logger.info("Backtest finished: %d bars in %.2fs", bar_count, wall_elapsed)

        result = self._report_gen.generate(
            config=self._config,
            trades=self._matching.trades,
            equity_curve=self._equity_curve,
            start_time=start_time,
            end_time=end_time,
        )
        return result

    def _on_bar_strategy(self, event: Event) -> None:
        """将Bar事件分发给策略。"""
        if self._strategy:
            self._strategy.on_bar(event.data)

    def _on_trade(self, event: Event) -> None:
        """成交事件处理：更新现金（买入扣款，卖出回款，双向扣手续费）。"""
        from .models import OrderSide, Trade

        trade: Trade = event.data
        notional = trade.price * trade.volume * self._config.contract_multiplier
        if trade.side == OrderSide.BUY:
            self._cash -= notional + trade.commission
        else:
            self._cash += notional - trade.commission

        if self._strategy:
            self._strategy.on_trade(trade)

    def _on_bar_equity(self, event: Event) -> None:
        """每根Bar更新权益曲线。"""
        bar: Bar = event.data
        self._current_bar[bar.symbol] = bar

        for symbol, b in self._current_bar.items():
            self._matching.mark_to_market(symbol, b.close)

        unrealized = Decimal(0)
        for pos in self._matching.positions.values():
            unrealized += pos.unrealized_pnl

        equity = self._cash + unrealized

        self._equity_curve.append(
            EquityCurvePoint(
                dt=bar.dt,
                equity=equity,
                cash=self._cash,
                position_value=unrealized,
            )
        )
