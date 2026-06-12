"""Cross-Asset Momentum — multi-instrument relative strength.

Ranks instruments by risk-adjusted momentum and goes long the
strongest, short the weakest.  Uses a cross-sectional z-score
to normalise signals across different volatility profiles.

Key technique (2024-2026 cross-sectional research):
- Dual-horizon momentum: fast (5-bar) + slow (20-bar)
- Vol-normalised returns for fair comparison across instruments
- Minimum holding period to avoid excessive turnover
- Signal decay: gradually reduce conviction as holding extends

This strategy operates on a SINGLE instrument but is designed
to receive cross-sectional rank info via ``bar["cs_rank"]``
(0.0 = weakest, 1.0 = strongest among the universe).
When no rank is provided, it falls back to absolute momentum.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

from strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig
from strategy.registry import auto_register

logger = logging.getLogger(__name__)


@auto_register("cross_asset_momentum")
class CrossAssetMomentumStrategy(BaseStrategy):
    """Cross-sectional momentum strategy for futures.

    Config keys:
    - ``fast_period``: fast momentum lookback (default 5)
    - ``slow_period``: slow momentum lookback (default 20)
    - ``rank_long_threshold``: CS rank above this → long (default 0.7)
    - ``rank_short_threshold``: CS rank below this → short (default 0.3)
    - ``min_hold_bars``: minimum holding period (default 5)
    - ``tp_atr_mult``: take-profit ATR multiplier (default 4.0)
    - ``sl_atr_mult``: stop-loss ATR multiplier (default 2.0)
    - ``vol_window``: volatility normalisation window (default 20)
    """

    def __init__(self, config: StrategyConfig | None = None) -> None:
        super().__init__(config or StrategyConfig(name="cross_asset_momentum"))
        self._fast = int(self.config.params.get("fast_period", 5))
        self._slow = int(self.config.params.get("slow_period", 20))
        self._rank_long = float(self.config.params.get("rank_long_threshold", 0.7))
        self._rank_short = float(self.config.params.get("rank_short_threshold", 0.3))
        self._min_hold = int(self.config.params.get("min_hold_bars", 5))
        self._tp_mult = float(self.config.params.get("tp_atr_mult", 4.0))
        self._sl_mult = float(self.config.params.get("sl_atr_mult", 2.0))
        self._vol_window = int(self.config.params.get("vol_window", 20))

        self._closes: deque[float] = deque(maxlen=max(self._slow + 5, 50))
        self._returns: deque[float] = deque(maxlen=self._vol_window + 1)
        self._atr: float = 0.0
        self._hold_count: int = 0
        self._current_side: int = 0

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        return []

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        result = self._compute_signal(symbol, bar)
        if result is not None:
            self.record_signal(result)
            return [result]
        return []

    def _compute_signal(self, symbol: str, bar: dict[str, Any]) -> Signal | None:
        c = float(bar.get("close", 0))
        h = float(bar.get("high", 0))
        l = float(bar.get("low", 0))
        cs_rank = bar.get("cs_rank")

        self._closes.append(c)
        n = len(self._closes)

        if n >= 2:
            prev = self._closes[-2]
            ret = (c - prev) / max(abs(prev), 1e-12)
            self._returns.append(ret)
            tr = max(h - l, abs(h - prev), abs(l - prev))
            alpha = 2.0 / 15.0
            self._atr = alpha * tr + (1 - alpha) * self._atr if self._atr > 0 else tr

        if n < self._slow + 2:
            return None

        closes_arr = np.array(list(self._closes))
        fast_ret = (closes_arr[-1] - closes_arr[-1 - self._fast]) / max(abs(closes_arr[-1 - self._fast]), 1e-12)
        slow_ret = (closes_arr[-1] - closes_arr[-1 - self._slow]) / max(abs(closes_arr[-1 - self._slow]), 1e-12)

        vol = np.std(list(self._returns)[-self._vol_window:]) if len(self._returns) >= self._vol_window else 0.01
        vol = max(vol, 1e-8)
        fast_z = fast_ret / vol
        slow_z = slow_ret / vol

        combined = 0.6 * fast_z + 0.4 * slow_z

        if self._current_side != 0:
            self._hold_count += 1

        signal_type = SignalType.HOLD
        confidence = 0.0

        entry_threshold = float(self.config.params.get("entry_threshold", 2.5))

        if cs_rank is not None:
            rank = float(cs_rank)
            if rank > self._rank_long and combined > entry_threshold * 0.5:
                signal_type = SignalType.LONG_ENTRY
                confidence = min(0.5 + rank * 0.3 + combined * 0.05, 0.95)
            elif rank < self._rank_short and combined < -entry_threshold * 0.5:
                signal_type = SignalType.SHORT_ENTRY
                confidence = min(0.5 + (1 - rank) * 0.3 + abs(combined) * 0.05, 0.95)
        else:
            if combined > entry_threshold:
                signal_type = SignalType.LONG_ENTRY
                confidence = min(0.5 + (combined - entry_threshold) * 0.1, 0.9)
            elif combined < -entry_threshold:
                signal_type = SignalType.SHORT_ENTRY
                confidence = min(0.5 + (abs(combined) - entry_threshold) * 0.1, 0.9)

        if self._current_side != 0 and self._hold_count < self._min_hold:
            if (self._current_side == 1 and signal_type == SignalType.SHORT_ENTRY) or \
               (self._current_side == -1 and signal_type == SignalType.LONG_ENTRY):
                signal_type = SignalType.HOLD
                confidence = 0.0

        if signal_type == SignalType.LONG_ENTRY:
            self._current_side = 1
            self._hold_count = 0
        elif signal_type == SignalType.SHORT_ENTRY:
            self._current_side = -1
            self._hold_count = 0

        if signal_type == SignalType.HOLD:
            return None

        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            strength=round(confidence, 4),
            price=c,
            metadata={
                "fast_z": round(fast_z, 3),
                "slow_z": round(slow_z, 3),
                "combined": round(combined, 3),
                "cs_rank": cs_rank,
                "vol": round(vol, 6),
                "atr": round(self._atr, 2),
            },
        )
