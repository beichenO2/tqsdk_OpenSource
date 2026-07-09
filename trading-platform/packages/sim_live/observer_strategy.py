"""Minimal observer strategy — records bars without generating signals.

Used as a fallback when real strategy templates are unavailable, so the
paper trading infrastructure can demonstrate feed connectivity and
scheduler operation while waiting for actual strategies from 1号位.
"""

from __future__ import annotations

from typing import Any

from strategy.base import BaseStrategy, Signal, StrategyConfig


class ObserverStrategy(BaseStrategy):
    """Passively observes market bars. Never trades."""

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.bar_count = 0

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self.bar_count += 1
        return []

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        return []
