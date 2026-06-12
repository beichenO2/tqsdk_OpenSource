"""Pydantic models for live simulation fills, config, and execution quality."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field


class Fill(BaseModel):
    """A single simulated execution against the book or a crossing tick."""

    fill_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    order_id: str
    symbol: str
    price: Decimal
    volume: int
    commission: Decimal = Decimal("0")
    timestamp: datetime
    aggressor_buy: bool = Field(
        description="True if the aggressive side was buying (lifting offers).",
    )


class SimConfig(BaseModel):
    """Configuration for the simulation matching engine."""

    commission_rate: Decimal = Decimal("0.0001")
    slippage_ticks: int = 0
    tick_size: Decimal = Decimal("1")
    contract_multiplier: int = 1
    latency_ms: int = Field(
        default=0,
        ge=0,
        description="Order release delay in simulation time (milliseconds).",
    )
    latency_jitter_ms: int = Field(
        default=0,
        ge=0,
        description="Uniform jitter [0, latency_jitter_ms] added per order (sim time).",
    )
    use_wall_clock_latency: bool = Field(
        default=False,
        description="If True, submit_order may await asyncio.sleep for latency (live paper).",
    )


class ExecutionQuality(BaseModel):
    """Aggregated shadow vs live execution diagnostics."""

    sample_count: int = 0
    shadow_avg_slippage_vs_mid: Decimal = Decimal("0")
    live_avg_slippage_vs_mid: Decimal | None = None
    shadow_total_fills: int = 0
    live_total_fills: int = 0
    avg_price_improvement_vs_live: Decimal | None = Field(
        default=None,
        description="Mean (shadow_fill_price - live_fill_price) for paired buys; "
        "sign flipped internally for sells so positive = better for the trader.",
    )
