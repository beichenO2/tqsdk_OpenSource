"""Pluggable cost models for crypto backtesting.

Supports flat-rate, tiered (VIP), and maker/taker fee structures
found on major exchanges (Binance, OKX, etc.). Funding rate costs
for perpetual futures are also modelled.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum

_ZERO = Decimal(0)


class FeeRole(str, Enum):
    MAKER = "maker"
    TAKER = "taker"


@dataclass(frozen=True)
class FeeTier:
    """A single VIP tier with maker/taker rates."""
    tier_name: str
    min_volume_30d: Decimal
    maker_rate: Decimal
    taker_rate: Decimal


@dataclass(frozen=True)
class CostBreakdown:
    """Itemised cost for a single fill."""
    commission: Decimal
    funding_cost: Decimal = _ZERO
    rebate: Decimal = _ZERO

    @property
    def total(self) -> Decimal:
        return self.commission + self.funding_cost - self.rebate


class CostModel(abc.ABC):
    """Base class for all cost models."""

    @abc.abstractmethod
    def calculate(
        self,
        price: Decimal,
        quantity: Decimal,
        role: FeeRole = FeeRole.TAKER,
    ) -> CostBreakdown:
        """Return the cost breakdown for a single fill."""

    @abc.abstractmethod
    def funding_cost(
        self,
        notional: Decimal,
        rate: Decimal,
        is_long: bool,
    ) -> Decimal:
        """8-hour funding payment for a perpetual position."""


class FlatRateCostModel(CostModel):
    """Fixed commission rate (e.g. Binance spot default 0.1 %)."""

    def __init__(self, rate: Decimal = Decimal("0.001")) -> None:
        self._rate = rate

    def calculate(
        self,
        price: Decimal,
        quantity: Decimal,
        role: FeeRole = FeeRole.TAKER,
    ) -> CostBreakdown:
        notional = price * quantity
        return CostBreakdown(commission=notional * self._rate)

    def funding_cost(
        self, notional: Decimal, rate: Decimal, is_long: bool,
    ) -> Decimal:
        payment = notional * rate
        return payment if is_long else -payment


class MakerTakerCostModel(CostModel):
    """Separate maker/taker rates (common for derivatives)."""

    def __init__(
        self,
        maker_rate: Decimal = Decimal("0.0002"),
        taker_rate: Decimal = Decimal("0.0004"),
    ) -> None:
        self._maker = maker_rate
        self._taker = taker_rate

    def calculate(
        self,
        price: Decimal,
        quantity: Decimal,
        role: FeeRole = FeeRole.TAKER,
    ) -> CostBreakdown:
        notional = price * quantity
        rate = self._maker if role == FeeRole.MAKER else self._taker
        rebate = _ZERO
        if role == FeeRole.MAKER and self._maker < 0:
            rebate = abs(notional * self._maker)
            return CostBreakdown(commission=_ZERO, rebate=rebate)
        return CostBreakdown(commission=notional * rate)

    def funding_cost(
        self, notional: Decimal, rate: Decimal, is_long: bool,
    ) -> Decimal:
        payment = notional * rate
        return payment if is_long else -payment


@dataclass
class TieredCostModel(CostModel):
    """Volume-based VIP tier schedule.

    Pass a list of ``FeeTier`` sorted by ``min_volume_30d`` ascending.
    The model selects the highest tier whose threshold the trader meets.
    """

    tiers: list[FeeTier] = field(default_factory=list)
    trailing_volume_30d: Decimal = _ZERO

    def __post_init__(self) -> None:
        if not self.tiers:
            self.tiers = _default_tiers()

    def _active_tier(self) -> FeeTier:
        best = self.tiers[0]
        for tier in self.tiers:
            if self.trailing_volume_30d >= tier.min_volume_30d:
                best = tier
        return best

    def calculate(
        self,
        price: Decimal,
        quantity: Decimal,
        role: FeeRole = FeeRole.TAKER,
    ) -> CostBreakdown:
        notional = price * quantity
        tier = self._active_tier()
        rate = tier.maker_rate if role == FeeRole.MAKER else tier.taker_rate
        self.trailing_volume_30d += notional
        return CostBreakdown(commission=notional * rate)

    def funding_cost(
        self, notional: Decimal, rate: Decimal, is_long: bool,
    ) -> Decimal:
        payment = notional * rate
        return payment if is_long else -payment


def _default_tiers() -> list[FeeTier]:
    """Binance USDM-futures-like default tiers."""
    return [
        FeeTier("VIP0", Decimal(0), Decimal("0.0002"), Decimal("0.0004")),
        FeeTier("VIP1", Decimal("5000000"), Decimal("0.00016"), Decimal("0.0004")),
        FeeTier("VIP2", Decimal("25000000"), Decimal("0.00014"), Decimal("0.00035")),
        FeeTier("VIP3", Decimal("100000000"), Decimal("0.00012"), Decimal("0.00032")),
        FeeTier("VIP4", Decimal("250000000"), Decimal("0.0001"), Decimal("0.0003")),
    ]
