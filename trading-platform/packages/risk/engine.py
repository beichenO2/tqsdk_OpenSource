"""Risk engine — aggregates risk limits and provides pre-trade check interface."""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import TYPE_CHECKING, Optional

from core.models.position import Position
from risk.limits import DailyLossLimit, RiskContext, RiskLimit

if TYPE_CHECKING:
    from execution.order_manager import OrderRequest

logger = logging.getLogger(__name__)


class RiskEngine:
    """Central risk management engine.

    Usage:
        risk = RiskEngine()
        risk.add_limit(MaxOrderSizeLimit(100))
        risk.add_limit(MaxPositionLimit(500))

        # Wire into execution engine
        execution.set_risk_checker(risk.pre_trade_check)
    """

    def __init__(self) -> None:
        self._limits: list[RiskLimit] = []
        self._positions: list[Position] = []
        self._balance: Decimal = Decimal("0")
        self._available: Decimal = Decimal("0")
        self._margin_ratio: Decimal = Decimal("0")
        self._last_prices: dict[str, Decimal] = {}
        self._daily_loss_limit: Optional[DailyLossLimit] = None

    def add_limit(self, limit: RiskLimit) -> None:
        if isinstance(limit, DailyLossLimit):
            self._daily_loss_limit = limit
        self._limits.append(limit)
        logger.info("Risk limit added: %s", limit.name)

    def remove_limit(self, name: str) -> None:
        self._limits = [l for l in self._limits if l.name != name]

    def update_positions(self, positions: list[Position]) -> None:
        self._positions = positions

    def update_account(self, balance: Decimal, available: Decimal, margin_ratio: Decimal) -> None:
        self._balance = balance
        self._available = available
        self._margin_ratio = margin_ratio

    def update_prices(self, prices: dict[str, Decimal]) -> None:
        self._last_prices.update(prices)

    def pre_trade_check(self, request: OrderRequest) -> tuple[bool, str]:
        """Run all risk limits — first failure stops the chain."""
        ctx = RiskContext(
            positions=self._positions,
            balance=self._balance,
            available=self._available,
            margin_ratio=self._margin_ratio,
            last_prices=self._last_prices,
        )
        for limit in self._limits:
            passed, reason = limit.check(request, ctx)
            if not passed:
                logger.warning(
                    "Risk FAILED [%s]: symbol=%s reason=%s",
                    limit.name, request.symbol, reason,
                )
                return False, f"[{limit.name}] {reason}"
        return True, ""

    def check_daily_loss(self, daily_pnl: Decimal) -> bool:
        if self._daily_loss_limit is None:
            return True
        if self._balance <= 0:
            if daily_pnl < 0:
                self._daily_loss_limit.trip()
                logger.critical("DAILY LOSS LIMIT BREACHED: balance=%.2f, pnl=%.2f", float(self._balance), float(daily_pnl))
                return False
            return True
        loss_pct = -daily_pnl / self._balance
        if loss_pct >= self._daily_loss_limit.max_loss_pct:
            self._daily_loss_limit.trip()
            logger.critical("DAILY LOSS LIMIT BREACHED: %.2f%%", float(loss_pct * 100))
            return False
        return True

    def reset_daily(self) -> None:
        if self._daily_loss_limit:
            self._daily_loss_limit.reset()
        logger.info("Daily risk limits reset")

    def get_status(self) -> dict:
        return {
            "limits": [l.name for l in self._limits],
            "positions_count": len(self._positions),
            "margin_ratio": float(self._margin_ratio),
            "daily_loss_tripped": self._daily_loss_limit.is_tripped if self._daily_loss_limit else False,
        }
