"""RiskGate — 下单前置统一入口。

把 RiskEngine.pre_trade_check 包装成可被 API / ExecutionService 直接调用的门面，
并附带拒绝事件回调（供 WebSocket risk 频道推送）。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Callable, Optional

from risk.engine import RiskEngine
from risk.futures_limits import DeliveryMonthLimit, LimitUpDownLimit, TradingSessionLimit
from risk.limits import (
    DailyLossLimit,
    MarginUtilizationLimit,
    MaxOrderSizeLimit,
    MaxPositionLimit,
    OrderFrequencyLimit,
    PriceBandLimit,
)

logger = logging.getLogger(__name__)

RejectCallback = Callable[[dict[str, Any]], None]


@dataclass
class GateVerdict:
    allowed: bool
    reason: str = ""
    limit_name: str = ""
    checked_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "limit_name": self.limit_name,
            "checked_at": self.checked_at,
        }


class RiskGate:
    """Pre-trade gate wrapping RiskEngine with futures-specific defaults."""

    def __init__(
        self,
        engine: Optional[RiskEngine] = None,
        *,
        enable_futures_limits: bool = True,
        on_reject: Optional[RejectCallback] = None,
    ) -> None:
        self.engine = engine or RiskEngine()
        self._on_reject = on_reject
        self._reject_count = 0
        if enable_futures_limits and not self.engine.get_status()["limits"]:
            self._install_defaults()
        elif enable_futures_limits:
            # Engine already has generic limits (e.g. from ExecutionService);
            # only append futures-specific ones if missing.
            existing = set(self.engine.get_status()["limits"])
            for limit in (
                LimitUpDownLimit(band_pct=Decimal("0.10")),
                DeliveryMonthLimit(),
            ):
                if limit.name not in existing:
                    self.engine.add_limit(limit)
            if (
                "TradingSession" not in existing
                and os.getenv("RISK_SKIP_SESSION_CHECK", "").strip().lower()
                not in ("1", "true", "yes")
            ):
                self.engine.add_limit(TradingSessionLimit(allow_night=True))

    def _install_defaults(self) -> None:
        self.engine.add_limit(MaxOrderSizeLimit(max_volume=200))
        self.engine.add_limit(MaxPositionLimit(max_position=1000))
        self.engine.add_limit(PriceBandLimit(max_deviation_pct=Decimal("0.05")))
        self.engine.add_limit(OrderFrequencyLimit(max_orders=30, window_seconds=60))
        self.engine.add_limit(MarginUtilizationLimit(max_ratio=Decimal("0.8")))
        self.engine.add_limit(DailyLossLimit(max_loss_pct=Decimal("0.05")))
        self.engine.add_limit(LimitUpDownLimit(band_pct=Decimal("0.10")))
        self.engine.add_limit(DeliveryMonthLimit())
        if os.getenv("RISK_SKIP_SESSION_CHECK", "").strip().lower() not in ("1", "true", "yes"):
            self.engine.add_limit(TradingSessionLimit(allow_night=True))

    def on_reject(self, callback: RejectCallback) -> None:
        self._on_reject = callback

    def check(self, request: Any) -> GateVerdict:
        """Run pre-trade check; emit reject event on failure."""
        ok, reason = self.engine.pre_trade_check(request)
        if ok:
            return GateVerdict(allowed=True)

        self._reject_count += 1
        # reason format: "[LimitName] detail"
        limit_name = ""
        if reason.startswith("[") and "]" in reason:
            limit_name = reason[1 : reason.index("]")]
        verdict = GateVerdict(allowed=False, reason=reason, limit_name=limit_name)
        logger.warning(
            "RiskGate REJECT symbol=%s vol=%s reason=%s",
            getattr(request, "symbol", "?"),
            getattr(request, "volume", "?"),
            reason,
        )
        if self._on_reject:
            try:
                self._on_reject(
                    {
                        "type": "risk_alert",
                        "source": "RiskGate",
                        "limit": limit_name,
                        "reason": reason,
                        "symbol": getattr(request, "symbol", None),
                        "volume": getattr(request, "volume", None),
                        "direction": str(getattr(request, "direction", "")),
                        "offset": str(getattr(request, "offset", "")),
                        "timestamp": verdict.checked_at,
                    }
                )
            except Exception:
                logger.exception("RiskGate reject callback failed")
        return verdict

    def check_as_tuple(self, request: Any) -> tuple[bool, str]:
        """Compatibility shim for ExecutionEngine.set_risk_checker((bool, str))."""
        verdict = self.check(request)
        return verdict.allowed, verdict.reason

    def get_status(self) -> dict[str, Any]:
        status = self.engine.get_status()
        status["reject_count"] = self._reject_count
        status["gate"] = "RiskGate"
        return status


def live_trading_enabled() -> bool:
    """Hard kill-switch: LIVE_TRADING_ENABLED must be explicitly true."""
    return os.getenv("LIVE_TRADING_ENABLED", "false").strip().lower() in ("1", "true", "yes")


def verify_live_confirm_token(token: Optional[str]) -> bool:
    """Validate X-Live-Confirm header against LIVE_CONFIRM_TOKEN env.

    If LIVE_CONFIRM_TOKEN is unset, any non-empty token matching the
    well-known session phrase is accepted (dev convenience). Production
    should set LIVE_CONFIRM_TOKEN explicitly.
    """
    if not token or not token.strip():
        return False
    expected = os.getenv("LIVE_CONFIRM_TOKEN", "").strip()
    if expected:
        return token.strip() == expected
    # Dev fallback: accept the fixed phrase used by the frontend confirm dialog
    return token.strip() == "I_UNDERSTAND_LIVE_RISK"
