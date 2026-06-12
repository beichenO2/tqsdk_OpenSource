"""Real-time risk monitor — background task watching positions and firing alerts."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Callable, Optional

from risk.engine import RiskEngine

logger = logging.getLogger(__name__)


class AlertLevel(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class RiskAlert:
    level: AlertLevel
    source: str
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    data: Optional[dict] = None


AlertCallback = Callable[[RiskAlert], None]


class RiskMonitor:
    """Async background monitor evaluating risk state on a timer."""

    def __init__(self, risk_engine: RiskEngine, interval: float = 5.0) -> None:
        self._engine = risk_engine
        self._interval = interval
        self._callbacks: list[AlertCallback] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None

        self._margin_warn = Decimal("0.6")
        self._margin_critical = Decimal("0.8")
        self._concentration_threshold = Decimal("0.3")

    def on_alert(self, callback: AlertCallback) -> None:
        self._callbacks.append(callback)

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Risk monitor started (interval=%.1fs)", self._interval)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self._interval)
                self._check_margin()
                self._check_concentration()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Risk monitor error")

    def _check_margin(self) -> None:
        status = self._engine.get_status()
        ratio = Decimal(str(status["margin_ratio"]))
        if ratio >= self._margin_critical:
            self._fire(RiskAlert(AlertLevel.CRITICAL, "Margin", f"Margin {ratio:.2%} CRITICAL"))
        elif ratio >= self._margin_warn:
            self._fire(RiskAlert(AlertLevel.WARNING, "Margin", f"Margin {ratio:.2%} warning"))

    def _check_concentration(self) -> None:
        positions = self._engine._positions
        if not positions:
            return
        total = sum(p.margin for p in positions)
        if total == 0:
            return
        for p in positions:
            conc = p.margin / total
            if conc >= self._concentration_threshold:
                self._fire(RiskAlert(
                    AlertLevel.WARNING, "Concentration",
                    f"{p.symbol} concentration {conc:.1%}",
                    data={"symbol": p.symbol, "concentration": float(conc)},
                ))

    def _fire(self, alert: RiskAlert) -> None:
        log_fn = {AlertLevel.INFO: logger.info, AlertLevel.WARNING: logger.warning, AlertLevel.CRITICAL: logger.critical}
        log_fn.get(alert.level, logger.info)("[%s] %s: %s", alert.level, alert.source, alert.message)
        for cb in self._callbacks:
            try:
                cb(alert)
            except Exception:
                logger.exception("Alert callback error")
