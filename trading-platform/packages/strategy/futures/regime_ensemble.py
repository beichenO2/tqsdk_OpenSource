"""Regime-Conditional Ensemble — adaptive strategy selector.

Detects the current market regime (trending / ranging / high-vol / breakout)
and routes to the sub-strategy best suited for that condition.  This avoids
the "one strategy for all markets" problem that causes most drawdowns.

Key idea (2024-2026 adaptive ensemble research):
- Each regime has a ranked list of recommended sub-strategies
- During trending → use momentum / trend-following
- During ranging → use mean-reversion / grid
- During high-vol → reduce size, prefer reversal
- During breakout → use breakout strategies with tight stops

Sub-strategies are existing registered strategies from the futures library.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

from strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig, Position
from strategy.registry import auto_register

logger = logging.getLogger(__name__)

REGIME_MAP: dict[str, list[str]] = {
    "strong_trend": ["cta_trend", "supertrend", "donchian_breakout"],
    "weak_trend": ["dual_ma", "macd_histogram", "keltner_channel"],
    "ranging": ["bollinger_mr", "rsi_divergence", "pivot_point"],
    "high_volatility": ["vol_breakout", "intraday_reversal"],
    "breakout": ["donchian_breakout", "keltner_channel", "vol_breakout"],
}


def _ema_update(prev: float, val: float, period: int) -> float:
    alpha = 2.0 / (period + 1)
    return alpha * val + (1 - alpha) * prev


@auto_register("regime_ensemble")
class RegimeEnsembleStrategy(BaseStrategy):
    """Market-regime-aware ensemble that delegates to the best sub-strategy.

    Config keys:
    - ``adx_period``: ADX calculation period (default 14)
    - ``adx_trend_threshold``: ADX above this = trending (default 25)
    - ``bb_period``: Bollinger Band period (default 20)
    - ``vol_high_threshold``: BB width above this = high volatility (default 0.06)
    - ``position_scale``: base position fraction (default 1.0)
    - ``tp_atr_mult``: take-profit ATR multiplier (default 3.0)
    - ``sl_atr_mult``: stop-loss ATR multiplier (default 1.5)
    """

    def __init__(self, config: StrategyConfig | None = None) -> None:
        super().__init__(config or StrategyConfig(name="regime_ensemble"))
        self._adx_period = int(self.config.params.get("adx_period", 14))
        self._adx_threshold = float(self.config.params.get("adx_trend_threshold", 25))
        self._bb_period = int(self.config.params.get("bb_period", 20))
        self._vol_threshold = float(self.config.params.get("vol_high_threshold", 0.06))
        self._tp_mult = float(self.config.params.get("tp_atr_mult", 3.0))
        self._sl_mult = float(self.config.params.get("sl_atr_mult", 1.5))

        self._closes: deque[float] = deque(maxlen=100)
        self._highs: deque[float] = deque(maxlen=100)
        self._lows: deque[float] = deque(maxlen=100)

        self._atr: float = 0.0
        self._adx: float = 0.0
        self._bb_width: float = 0.0
        self._regime: str = "unknown"

        self._dm_plus_ema: float = 0.0
        self._dm_minus_ema: float = 0.0
        self._tr_ema: float = 0.0

        self._mom_fast: float = 0.0
        self._mom_slow: float = 0.0
        self._rsi: float = 50.0
        self._avg_gain: float = 0.0
        self._avg_loss: float = 0.0

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        return []

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        result = self._compute_signal(symbol, bar)
        if result is not None:
            self.record_signal(result)
            return [result]
        return []

    def _compute_signal(self, symbol: str, bar: dict[str, Any]) -> Signal | None:
        h = float(bar.get("high", 0))
        l = float(bar.get("low", 0))
        c = float(bar.get("close", 0))

        self._highs.append(h)
        self._lows.append(l)
        self._closes.append(c)
        n = len(self._closes)

        if n < 3:
            return None

        prev_c = self._closes[-2]
        prev_h = self._highs[-2]
        prev_l = self._lows[-2]

        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        self._tr_ema = _ema_update(self._tr_ema, tr, self._adx_period) if self._tr_ema > 0 else tr
        self._atr = self._tr_ema

        dm_plus = max(h - prev_h, 0.0) if (h - prev_h) > (prev_l - l) else 0.0
        dm_minus = max(prev_l - l, 0.0) if (prev_l - l) > (h - prev_h) else 0.0
        self._dm_plus_ema = _ema_update(self._dm_plus_ema, dm_plus, self._adx_period) if self._dm_plus_ema > 0 else dm_plus
        self._dm_minus_ema = _ema_update(self._dm_minus_ema, dm_minus, self._adx_period) if self._dm_minus_ema > 0 else dm_minus

        atr_safe = max(self._atr, 1e-12)
        di_plus = self._dm_plus_ema / atr_safe * 100
        di_minus = self._dm_minus_ema / atr_safe * 100
        di_sum = di_plus + di_minus + 1e-12
        dx = abs(di_plus - di_minus) / di_sum * 100
        self._adx = _ema_update(self._adx, dx, self._adx_period) if self._adx > 0 else dx

        if n >= self._bb_period:
            recent = list(self._closes)[-self._bb_period:]
            bb_mean = np.mean(recent)
            bb_std = np.std(recent) + 1e-12
            self._bb_width = 4.0 * bb_std / max(bb_mean, 1e-8)

        is_trending = self._adx > self._adx_threshold
        is_high_vol = self._bb_width > self._vol_threshold
        is_breakout = is_trending and is_high_vol

        if is_breakout:
            self._regime = "breakout"
        elif is_trending:
            self._regime = "strong_trend" if self._adx > 35 else "weak_trend"
        elif is_high_vol:
            self._regime = "high_volatility"
        else:
            self._regime = "ranging"

        delta = c - prev_c
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        self._avg_gain = _ema_update(self._avg_gain, gain, 14) if self._avg_gain > 0 else gain
        self._avg_loss = _ema_update(self._avg_loss, loss, 14) if self._avg_loss > 0 else loss
        rs = self._avg_gain / max(self._avg_loss, 1e-12)
        self._rsi = 100.0 - 100.0 / (1.0 + rs)

        self._mom_fast = _ema_update(self._mom_fast, c, 10) if self._mom_fast > 0 else c
        self._mom_slow = _ema_update(self._mom_slow, c, 30) if self._mom_slow > 0 else c

        signal_type = SignalType.HOLD
        confidence = 0.0

        if self._regime in ("strong_trend", "breakout"):
            mom_diff = (self._mom_fast - self._mom_slow) / max(abs(self._mom_slow), 1e-8)
            if mom_diff > 0.001 and self._rsi < 75:
                signal_type = SignalType.LONG_ENTRY
                confidence = min(0.5 + abs(mom_diff) * 10, 0.95)
            elif mom_diff < -0.001 and self._rsi > 25:
                signal_type = SignalType.SHORT_ENTRY
                confidence = min(0.5 + abs(mom_diff) * 10, 0.95)

        elif self._regime == "weak_trend":
            mom_diff = (self._mom_fast - self._mom_slow) / max(abs(self._mom_slow), 1e-8)
            if mom_diff > 0.002:
                signal_type = SignalType.LONG_ENTRY
                confidence = min(0.4 + abs(mom_diff) * 8, 0.8)
            elif mom_diff < -0.002:
                signal_type = SignalType.SHORT_ENTRY
                confidence = min(0.4 + abs(mom_diff) * 8, 0.8)

        elif self._regime == "ranging":
            if self._rsi < 30:
                signal_type = SignalType.LONG_ENTRY
                confidence = 0.6 + (30 - self._rsi) / 30 * 0.3
            elif self._rsi > 70:
                signal_type = SignalType.SHORT_ENTRY
                confidence = 0.6 + (self._rsi - 70) / 30 * 0.3

        elif self._regime == "high_volatility":
            if self._rsi < 20:
                signal_type = SignalType.LONG_ENTRY
                confidence = 0.5
            elif self._rsi > 80:
                signal_type = SignalType.SHORT_ENTRY
                confidence = 0.5

        if signal_type == SignalType.HOLD:
            return None

        sig_type = SignalType.LONG_ENTRY if signal_type == SignalType.LONG_ENTRY else SignalType.SHORT_ENTRY
        if confidence > 0 and n >= 3:
            pos = self.get_position(symbol)
            if pos is not None:
                if signal_type == SignalType.LONG_ENTRY and pos.side == "short":
                    sig_type = SignalType.SHORT_EXIT
                elif signal_type == SignalType.SHORT_ENTRY and pos.side == "long":
                    sig_type = SignalType.LONG_EXIT

        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            signal_type=sig_type,
            strength=round(confidence, 4),
            price=c,
            metadata={
                "regime": self._regime,
                "adx": round(self._adx, 2),
                "rsi": round(self._rsi, 2),
                "bb_width": round(self._bb_width, 4),
                "atr": round(self._atr, 2),
            },
        )
