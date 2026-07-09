"""ExecutionService — high-level service for API DI injection.

This is the single entry point that API routes use to interact with
the execution and risk layer. It encapsulates ExecutionEngine, RiskEngine,
and RiskMonitor setup.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Optional

from core.enums.direction import Direction, Offset
from core.models.order import Order
from core.models.position import Position

from execution.account_history import AccountHistoryStore
from execution.broker_adapter import BrokerAdapter
from execution.engine import ExecutionEngine
from execution.order_manager import OrderRequest
from risk.engine import RiskEngine
from risk.futures_limits import DeliveryMonthLimit, LimitUpDownLimit, TradingSessionLimit
from risk.gate import RiskGate
from risk.limits import (
    DailyLossLimit,
    MarginUtilizationLimit,
    MaxOrderSizeLimit,
    MaxPositionLimit,
    OrderFrequencyLimit,
    PriceBandLimit,
)
from risk.monitor import RiskMonitor

logger = logging.getLogger(__name__)


class ExecutionService:
    """Facade consumed by FastAPI routes via dependency injection."""

    def __init__(self, broker: BrokerAdapter) -> None:
        self.risk_engine = RiskEngine()
        self.execution_engine = ExecutionEngine(broker)
        self.risk_monitor = RiskMonitor(self.risk_engine)
        self.account_history = AccountHistoryStore()
        self._snapshot_task = None

        self._setup_default_risk_limits()
        # RiskGate wraps the same engine (adds reject events); checker stays on engine.
        self.risk_gate = RiskGate(engine=self.risk_engine, enable_futures_limits=False)
        self.execution_engine.set_risk_checker(self.risk_gate.check_as_tuple)

    def _setup_default_risk_limits(self) -> None:
        self.risk_engine.add_limit(MaxOrderSizeLimit(max_volume=200))
        self.risk_engine.add_limit(MaxPositionLimit(max_position=1000))
        self.risk_engine.add_limit(PriceBandLimit(max_deviation_pct=Decimal("0.05")))
        self.risk_engine.add_limit(OrderFrequencyLimit(max_orders=30, window_seconds=60))
        self.risk_engine.add_limit(MarginUtilizationLimit(max_ratio=Decimal("0.8")))
        self.risk_engine.add_limit(DailyLossLimit(max_loss_pct=Decimal("0.05")))
        # Futures-specific pre-trade gates (涨跌停 / 交割月 / 交易时段)
        self.risk_engine.add_limit(LimitUpDownLimit(band_pct=Decimal("0.10")))
        self.risk_engine.add_limit(DeliveryMonthLimit())
        import os
        if os.getenv("RISK_SKIP_SESSION_CHECK", "").strip().lower() not in ("1", "true", "yes"):
            self.risk_engine.add_limit(TradingSessionLimit(allow_night=True))

    async def start(self) -> None:
        import asyncio
        await self.execution_engine.start()
        await self.risk_monitor.start()
        self._snapshot_task = asyncio.create_task(self._snapshot_loop())
        logger.info("ExecutionService started")

    async def stop(self) -> None:
        import asyncio
        if self._snapshot_task:
            self._snapshot_task.cancel()
            try:
                await self._snapshot_task
            except asyncio.CancelledError:
                pass
        await self.risk_monitor.stop()
        await self.execution_engine.stop()
        logger.info("ExecutionService stopped")

    async def _snapshot_loop(self) -> None:
        """开市时每分钟落一条权益快照；闭市查询失败则静默跳过。"""
        import asyncio
        while True:
            try:
                await asyncio.sleep(60.0)
                account = await asyncio.wait_for(self.get_account_info(), timeout=8.0)
                if account:
                    self.account_history.append(account)
            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001 — closed market / gateway slow
                logger.debug("account snapshot skipped: %s", e)

    def get_pnl_history(self, days: int = 30) -> list[dict]:
        """账户权益历史（上一交易区间的静止曲线来自这里）。"""
        return self.account_history.load(days=days)

    async def place_order(
        self,
        strategy_id: str,
        symbol: str,
        exchange: str,
        direction: Direction,
        offset: Offset,
        price: Decimal,
        volume: int,
    ) -> Order:
        """Place a new order — validates through risk, submits to broker."""
        request = OrderRequest(
            symbol=symbol,
            exchange=exchange,
            direction=direction,
            offset=offset,
            price=price,
            volume=volume,
            strategy_id=strategy_id,
        )
        return await self.execution_engine.place_order(request)

    async def cancel_order(self, order_id: str) -> bool:
        return await self.execution_engine.cancel_order(order_id)

    async def cancel_all(self, symbol: Optional[str] = None) -> int:
        return await self.execution_engine.cancel_all(symbol)

    def get_order(self, order_id: str) -> Optional[Order]:
        return self.execution_engine.order_manager.get_order(order_id)

    def get_active_orders(self, symbol: Optional[str] = None) -> list[Order]:
        return self.execution_engine.order_manager.get_active_orders(symbol)

    def get_all_orders(self) -> list[Order]:
        return self.execution_engine.order_manager.get_all_orders()

    def get_positions(self) -> list[Position]:
        return self.execution_engine.position_manager.get_all_positions()

    def get_position(self, symbol: str, direction: Direction) -> Optional[Position]:
        return self.execution_engine.position_manager.get_position(symbol, direction)

    async def get_account_info(self) -> dict:
        return await self.execution_engine.get_account_info()

    def get_risk_status(self) -> dict:
        return self.risk_engine.get_status()

    def update_prices(self, prices: dict[str, Decimal]) -> None:
        self.risk_engine.update_prices(prices)
        self.execution_engine.update_market_prices(prices)
