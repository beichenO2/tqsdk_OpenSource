"""Build in-memory evidence chains for trade decisions."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class EvidenceEvent(BaseModel):
    """One step in an evidence chain (signal, risk, order, or close)."""

    timestamp: datetime
    event_type: str
    data: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("timestamp")
    @classmethod
    def ensure_tz_aware(cls, v: datetime) -> datetime:
        """Normalize naive datetimes to UTC for consistent serialization."""
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v


class EvidenceChain(BaseModel):
    """Complete evidence chain for a single trade lifecycle."""

    trade_id: str
    symbol: str
    opened_at: datetime
    events: list[EvidenceEvent] = Field(default_factory=list)
    finalized_at: datetime | None = None

    @field_validator("opened_at", "finalized_at")
    @classmethod
    def ensure_tz_aware_chain(cls, v: datetime | None) -> datetime | None:
        if v is None:
            return v
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v


class EvidenceChainBuilder:
    """Captures trading decision events and produces an :class:`EvidenceChain`."""

    def __init__(self) -> None:
        self._trade_id: str | None = None
        self._symbol: str | None = None
        self._opened_at: datetime | None = None
        self._events: list[EvidenceEvent] = []

    def create_chain(self, trade_id: str, symbol: str, timestamp: datetime) -> None:
        """Start a new chain; clears any previous events."""
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        self._trade_id = trade_id
        self._symbol = symbol
        self._opened_at = timestamp
        self._events = []

    def add_signal_event(
        self,
        timestamp: datetime,
        data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append a strategy / model signal event."""
        self._append_event("signal", timestamp, data, metadata)

    def add_risk_check_event(
        self,
        timestamp: datetime,
        data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append a risk check outcome event."""
        self._append_event("risk_check", timestamp, data, metadata)

    def add_order_event(
        self,
        timestamp: datetime,
        data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append an order submission or fill-related decision event."""
        self._append_event("order", timestamp, data, metadata)

    def add_close_event(
        self,
        timestamp: datetime,
        data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append a position close or exit event."""
        self._append_event("close", timestamp, data, metadata)

    def finalize(self) -> EvidenceChain:
        """Return a frozen :class:`EvidenceChain` with all captured events."""
        if self._trade_id is None or self._opened_at is None:
            msg = "create_chain() must be called before finalize()"
            raise RuntimeError(msg)
        symbol = self._symbol or ""
        return EvidenceChain(
            trade_id=self._trade_id,
            symbol=symbol,
            opened_at=self._opened_at,
            events=list(self._events),
            finalized_at=datetime.now(UTC),
        )

    def _append_event(
        self,
        event_type: str,
        timestamp: datetime,
        data: dict[str, Any],
        metadata: dict[str, Any] | None,
    ) -> None:
        if self._trade_id is None:
            msg = "create_chain() must be called before adding events"
            raise RuntimeError(msg)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        self._events.append(
            EvidenceEvent(
                timestamp=timestamp,
                event_type=event_type,
                data=dict(data),
                metadata=dict(metadata or {}),
            )
        )
