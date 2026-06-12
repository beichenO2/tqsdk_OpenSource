"""Cross-Sectional Momentum Rotation Strategy.

Ranks multiple crypto assets by recent performance and goes long
the winners while avoiding/shorting the losers. This is a pure
relative value strategy — profits from dispersion between assets.

Research: 75% annualized returns, Feb 2026 +11.27% during -23.49% market.
Uses 30-day lookback and 7-day holding period (adapted for bar-based).

Requires multi-asset data fed via bar.extra or separate bars per symbol.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig
from strategy.registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "lookback_bars": 30,
    "hold_period": 7,
    "top_n": 2,
    "bottom_n": 1,
    "rebalance_interval": 7,
    "min_return_threshold": 0.01,
    "max_risk_per_trade": 0.02,
    "position_fraction": 0.3,
}


@auto_register("momentum_rotation")
class MomentumRotationStrategy(BaseStrategy):
    """Rank assets by momentum and rotate into top performers."""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._closes: dict[str, deque[float]] = {}
        self._bar_count: int = 0
        self._last_rebalance: int = 0
        self._held_long: set[str] = set()
        self._held_short: set[str] = set()

    def _init(self, s: str) -> None:
        if s not in self._closes:
            self._closes[s] = deque(maxlen=200)

    def _compute_momentum(self, s: str) -> float | None:
        closes = list(self._closes[s])
        lb = self.get_param("lookback_bars")
        if len(closes) < lb:
            return None
        if closes[-lb] <= 0:
            return None
        return (closes[-1] - closes[-lb]) / closes[-lb]

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._init(symbol)
        c = bar["close"]
        self._closes[symbol].append(c)
        return []

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        self._bar_count += 1

        for symbol in self.config.symbols:
            bar = market_data.get(symbol)
            if bar:
                await self.on_bar(symbol, bar)

        rebalance_interval = self.get_param("rebalance_interval")
        if self._bar_count - self._last_rebalance < rebalance_interval:
            return []

        rankings: list[tuple[str, float]] = []
        for s in self.config.symbols:
            mom = self._compute_momentum(s)
            if mom is not None:
                rankings.append((s, mom))

        if len(rankings) < 3:
            return []

        rankings.sort(key=lambda x: x[1], reverse=True)
        self._last_rebalance = self._bar_count

        top_n = self.get_param("top_n")
        bottom_n = self.get_param("bottom_n")
        min_ret = self.get_param("min_return_threshold")
        pos_frac = self.get_param("position_fraction")

        new_longs = set()
        new_shorts = set()

        for sym, mom in rankings[:top_n]:
            if mom > min_ret:
                new_longs.add(sym)

        for sym, mom in rankings[-bottom_n:]:
            if mom < -min_ret:
                new_shorts.add(sym)

        signals: list[Signal] = []

        for sym in self._held_long - new_longs:
            pos = self.get_position(sym)
            if pos is not None:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=sym,
                    signal_type=SignalType.LONG_EXIT, strength=0.7,
                    price=self._closes[sym][-1] if self._closes[sym] else 0,
                    reason=f"ROTATION exit long (no longer top {top_n})",
                ))

        for sym in self._held_short - new_shorts:
            pos = self.get_position(sym)
            if pos is not None:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=sym,
                    signal_type=SignalType.SHORT_EXIT, strength=0.7,
                    price=self._closes[sym][-1] if self._closes[sym] else 0,
                    reason=f"ROTATION exit short (no longer bottom {bottom_n})",
                ))

        for sym in new_longs - self._held_long:
            if self.get_position(sym) is None:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=sym,
                    signal_type=SignalType.LONG_ENTRY, strength=0.8,
                    price=self._closes[sym][-1] if self._closes[sym] else 0,
                    reason=f"ROTATION long (top {top_n} momentum)",
                    metadata={"momentum": self._compute_momentum(sym), "position_fraction": pos_frac},
                ))

        for sym in new_shorts - self._held_short:
            if self.get_position(sym) is None:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=sym,
                    signal_type=SignalType.SHORT_ENTRY, strength=0.6,
                    price=self._closes[sym][-1] if self._closes[sym] else 0,
                    reason=f"ROTATION short (bottom {bottom_n} momentum)",
                    metadata={"momentum": self._compute_momentum(sym), "position_fraction": pos_frac / 2},
                ))

        self._held_long = new_longs
        self._held_short = new_shorts

        return signals
