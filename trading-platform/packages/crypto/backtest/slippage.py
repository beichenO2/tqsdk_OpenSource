"""Pluggable slippage models for crypto backtesting.

Crypto markets exhibit distinct slippage characteristics compared to
traditional futures — thinner books, wider spreads at extremes, and
volume-dependent impact. Three models are provided:

- FixedBpsSlippage: constant basis-point model (simple default)
- VolumeImpactSlippage: √(volume) market-impact model
- VolatilityAdaptiveSlippage: ATR-scaled slippage
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass
from decimal import Decimal

_ZERO = Decimal(0)


@dataclass(frozen=True)
class SlippageResult:
    """Slippage applied to a single fill."""
    raw_price: Decimal
    slipped_price: Decimal
    slippage_amount: Decimal

    @property
    def bps(self) -> Decimal:
        if self.raw_price == 0:
            return _ZERO
        return self.slippage_amount / self.raw_price * Decimal(10000)


class SlippageModel(abc.ABC):
    """Base class for all slippage models."""

    @abc.abstractmethod
    def apply(
        self,
        price: Decimal,
        quantity: Decimal,
        is_buy: bool,
        *,
        bar_volume: Decimal = _ZERO,
        volatility: Decimal = _ZERO,
    ) -> SlippageResult:
        """Return price after slippage."""


class FixedBpsSlippage(SlippageModel):
    """Constant basis-point slippage (default 5 bps)."""

    def __init__(self, bps: Decimal = Decimal("5")) -> None:
        self._bps = bps

    def apply(
        self,
        price: Decimal,
        quantity: Decimal,
        is_buy: bool,
        *,
        bar_volume: Decimal = _ZERO,
        volatility: Decimal = _ZERO,
    ) -> SlippageResult:
        slip = price * self._bps / Decimal(10000)
        slipped = price + slip if is_buy else price - slip
        return SlippageResult(
            raw_price=price,
            slipped_price=slipped,
            slippage_amount=slip,
        )


class VolumeImpactSlippage(SlippageModel):
    """Square-root market-impact model.

    slippage = base_bps + impact_coeff * sqrt(qty / bar_volume) * 10000 bps

    When the order is a large fraction of bar volume, slippage grows
    non-linearly. Falls back to *base_bps* when bar_volume is zero.
    """

    def __init__(
        self,
        base_bps: Decimal = Decimal("2"),
        impact_coeff: Decimal = Decimal("0.1"),
    ) -> None:
        self._base_bps = base_bps
        self._impact_coeff = impact_coeff

    def apply(
        self,
        price: Decimal,
        quantity: Decimal,
        is_buy: bool,
        *,
        bar_volume: Decimal = _ZERO,
        volatility: Decimal = _ZERO,
    ) -> SlippageResult:
        extra_bps = _ZERO
        if bar_volume > 0 and quantity > 0:
            participation = float(quantity / bar_volume)
            extra_bps = self._impact_coeff * Decimal(str(math.sqrt(participation))) * Decimal(10000)
        total_bps = self._base_bps + extra_bps
        slip = price * total_bps / Decimal(10000)
        slipped = price + slip if is_buy else price - slip
        return SlippageResult(
            raw_price=price,
            slipped_price=slipped,
            slippage_amount=slip,
        )


class VolatilityAdaptiveSlippage(SlippageModel):
    """ATR-scaled slippage model.

    slippage = scale * volatility (typically the bar's ATR).
    During calm markets slippage shrinks; during spikes it widens,
    better modelling real crypto order books.
    """

    def __init__(
        self,
        scale: Decimal = Decimal("0.5"),
        floor_bps: Decimal = Decimal("1"),
    ) -> None:
        self._scale = scale
        self._floor_bps = floor_bps

    def apply(
        self,
        price: Decimal,
        quantity: Decimal,
        is_buy: bool,
        *,
        bar_volume: Decimal = _ZERO,
        volatility: Decimal = _ZERO,
    ) -> SlippageResult:
        vol_slip = self._scale * volatility
        floor = price * self._floor_bps / Decimal(10000)
        slip = max(vol_slip, floor)
        slipped = price + slip if is_buy else price - slip
        return SlippageResult(
            raw_price=price,
            slipped_price=slipped,
            slippage_amount=slip,
        )
