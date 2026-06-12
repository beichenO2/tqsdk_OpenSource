"""Grid Strategy v2 — regime-filtered with dynamic ATR-based spacing.

Improvements over archived grid strategy:
1. Only active in 'ranging' or 'high_volatility' regimes
2. Grid spacing = 2x ATR(20) instead of fixed percentage
3. Grid center auto-adjusts every N bars based on current price
4. Cooldown after regime changes to avoid whipsaw
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig
from strategy.indicators import calc_atr
from strategy.registry import auto_register
from .regime_detector import MarketRegimeDetector

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "grid_count": 10,
    "atr_period": 20,
    "atr_grid_mult": 2.0,
    "order_qty_per_grid": 0.001,
    "recenter_interval": 48,
    "allowed_regimes": ["ranging", "high_volatility"],
    "regime_cooldown": 5,
    "max_open_grids": 5,
}


@auto_register("grid_v2")
class GridV2Strategy(BaseStrategy):
    """ATR-dynamic grid with regime gating."""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._closes: dict[str, deque[float]] = {}
        self._highs: dict[str, deque[float]] = {}
        self._lows: dict[str, deque[float]] = {}
        self._regime: dict[str, MarketRegimeDetector] = {}
        self._grid_levels: dict[str, list[float]] = {}
        self._filled_buys: dict[str, set[int]] = {}
        self._last_regime_change: dict[str, int] = {}
        self._bar_count: dict[str, int] = {}
        self._active: dict[str, bool] = {}

    def _init(self, s: str) -> None:
        if s not in self._closes:
            self._closes[s] = deque(maxlen=200)
            self._highs[s] = deque(maxlen=200)
            self._lows[s] = deque(maxlen=200)
            self._regime[s] = MarketRegimeDetector()
            self._filled_buys[s] = set()
            self._bar_count[s] = 0
            self._active[s] = False

    def _build_grid(self, s: str, center: float, atr: float) -> None:
        count = self.get_param("grid_count")
        spacing = atr * self.get_param("atr_grid_mult")
        if spacing <= 0:
            return

        half = count // 2
        self._grid_levels[s] = [center + (i - half) * spacing for i in range(count + 1)]
        self._filled_buys[s] = set()

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._init(symbol)
        c, h, l = bar["close"], bar["high"], bar["low"]
        self._closes[symbol].append(c)
        self._highs[symbol].append(h)
        self._lows[symbol].append(l)
        self._bar_count[symbol] += 1

        self._regime[symbol].update(h, l, c)
        regime = self._regime[symbol].current_regime

        allowed = self.get_param("allowed_regimes")
        was_active = self._active.get(symbol, False)
        now_active = regime.value in allowed

        if now_active != was_active:
            self._last_regime_change[symbol] = self._bar_count[symbol]
            self._active[symbol] = now_active

        cooldown = self.get_param("regime_cooldown")
        bars_since_change = self._bar_count[symbol] - self._last_regime_change.get(symbol, 0)
        if bars_since_change < cooldown:
            return []

        if not now_active:
            if self._filled_buys.get(symbol):
                return self._close_all_grids(symbol, c)
            return []

        atr = calc_atr(self._highs[symbol], self._lows[symbol], self._closes[symbol], self.get_param("atr_period"))
        if atr is None or atr <= 0:
            return []

        recenter = self.get_param("recenter_interval")
        if symbol not in self._grid_levels or self._bar_count[symbol] % recenter == 0:
            self._build_grid(symbol, c, atr)

        levels = self._grid_levels.get(symbol, [])
        if not levels:
            return []

        prev_c = list(self._closes[symbol])[-2] if len(self._closes[symbol]) >= 2 else c
        signals: list[Signal] = []
        qty = self.get_param("order_qty_per_grid")
        max_open = self.get_param("max_open_grids")

        for i, level in enumerate(levels):
            if prev_c > level >= c and i not in self._filled_buys[symbol]:
                if len(self._filled_buys[symbol]) >= max_open:
                    continue
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY, strength=0.6,
                    price=level, suggested_qty=qty,
                    reason=f"GridV2 BUY @{level:.1f} [{regime.value}]",
                    metadata={"grid_idx": i, "regime": regime.value},
                ))
                self._filled_buys[symbol].add(i)

            elif prev_c < level <= c and i in self._filled_buys[symbol]:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_EXIT, strength=0.6,
                    price=level, suggested_qty=qty,
                    reason=f"GridV2 SELL @{level:.1f} [{regime.value}]",
                    metadata={"grid_idx": i, "regime": regime.value},
                ))
                self._filled_buys[symbol].discard(i)

        return signals

    def _close_all_grids(self, symbol: str, price: float) -> list[Signal]:
        signals = []
        qty = self.get_param("order_qty_per_grid")
        for i in list(self._filled_buys.get(symbol, set())):
            signals.append(Signal(
                strategy_id=self.strategy_id, symbol=symbol,
                signal_type=SignalType.LONG_EXIT, strength=0.5,
                price=price, suggested_qty=qty,
                reason=f"GridV2 regime exit (flat all)",
            ))
        self._filled_buys[symbol] = set()
        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        for s in self.config.symbols:
            if s in market_data:
                out.extend(await self.on_bar(s, market_data[s]))
        return out
