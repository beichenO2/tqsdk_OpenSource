"""Funding Rate Mean-Reversion Alpha (2024-2025).

Perpetual futures funding rates reflect leveraged sentiment. When funding
is extremely positive (longs pay shorts), the market is over-leveraged long
and a correction is likely. Vice versa for extreme negative funding.

This strategy:
1. Maintains an EWM of the funding rate
2. Enters contrarian positions when funding deviates > Z-score threshold
3. Uses ATR-based risk management
4. Exits when funding normalizes or barriers are hit

Alpha source: Funding rate as a contrarian sentiment indicator.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..indicators import calc_atr, ema_update, check_atr_exit
from ..registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "funding_z_entry": 1.14,
    "funding_z_exit": 0.31,
    "funding_ewm_span": 15,
    "funding_lookback": 61,
    "atr_period": 19,
    "stop_loss_atr_mult": 1.93,
    "take_profit_atr_mult": 4.2,
    "max_hold_bars": 29,
    "ema_trend_period": 44,
    "require_trend_alignment": True,
}


@auto_register("funding_rate_alpha")
class FundingRateAlphaStrategy(BaseStrategy):
    """Contrarian funding rate strategy for crypto perpetual futures."""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._close: dict[str, deque[float]] = {}
        self._high: dict[str, deque[float]] = {}
        self._low: dict[str, deque[float]] = {}
        self._funding_rates: dict[str, deque[float]] = {}
        self._funding_ewm: dict[str, float | None] = {}
        self._ema_trend: dict[str, float | None] = {}
        self._bars_in_pos: dict[str, int] = {}
        self._buf = 200

    def _ensure_buffers(self, symbol: str) -> None:
        if symbol not in self._close:
            self._close[symbol] = deque(maxlen=self._buf)
            self._high[symbol] = deque(maxlen=self._buf)
            self._low[symbol] = deque(maxlen=self._buf)
            self._funding_rates[symbol] = deque(maxlen=self._buf)

    def _funding_zscore(self, symbol: str) -> float | None:
        """Z-score of current EWM funding rate vs historical distribution."""
        rates = list(self._funding_rates[symbol])
        lookback = self.get_param("funding_lookback")
        if len(rates) < max(lookback, 20):
            return None

        window = rates[-lookback:]
        mean_f = sum(window) / len(window)
        var_f = sum((r - mean_f) ** 2 for r in window) / len(window)
        std_f = math.sqrt(var_f) if var_f > 0 else 0.0

        if std_f < 1e-8:
            return None

        current = self._funding_ewm.get(symbol, 0.0) or 0.0
        z = (current - mean_f) / std_f
        return max(-10.0, min(10.0, z))

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._ensure_buffers(symbol)
        close = bar["close"]
        high = bar["high"]
        low = bar["low"]
        funding = bar.get("funding_rate", 0.0)

        self._close[symbol].append(close)
        self._high[symbol].append(high)
        self._low[symbol].append(low)
        self._funding_rates[symbol].append(funding)

        span = self.get_param("funding_ewm_span")
        alpha = 2.0 / (span + 1)
        prev_ewm = self._funding_ewm.get(symbol)
        if prev_ewm is None:
            self._funding_ewm[symbol] = funding
        else:
            self._funding_ewm[symbol] = alpha * funding + (1 - alpha) * prev_ewm

        self._ema_trend[symbol] = ema_update(
            self._ema_trend.get(symbol), close, self.get_param("ema_trend_period")
        )

        atr = calc_atr(
            self._high[symbol], self._low[symbol], self._close[symbol],
            self.get_param("atr_period"),
        )

        signals: list[Signal] = []
        if atr is None or atr <= 0:
            return signals

        z = self._funding_zscore(symbol)
        if z is None:
            return signals

        pos = self.get_position(symbol)
        z_entry = self.get_param("funding_z_entry")
        z_exit = self.get_param("funding_z_exit")

        if pos is None:
            ema_t = self._ema_trend.get(symbol) or close
            trend_ok = True
            if self.get_param("require_trend_alignment"):
                if z > z_entry:
                    trend_ok = close < ema_t
                elif z < -z_entry:
                    trend_ok = close > ema_t

            if z > z_entry and trend_ok:
                strength = min(abs(z) / (z_entry * 2), 1.0)
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=round(strength, 4),
                    price=close,
                    reason=f"Funding过高(z={z:.2f})→做空",
                    metadata={"funding_z": z, "funding_ewm": self._funding_ewm[symbol]},
                )
                signals.append(sig)
                self.record_signal(sig)
                self._bars_in_pos[symbol] = 0

            elif z < -z_entry and trend_ok:
                strength = min(abs(z) / (z_entry * 2), 1.0)
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=round(strength, 4),
                    price=close,
                    reason=f"Funding过低(z={z:.2f})→做多",
                    metadata={"funding_z": z, "funding_ewm": self._funding_ewm[symbol]},
                )
                signals.append(sig)
                self.record_signal(sig)
                self._bars_in_pos[symbol] = 0

        else:
            self._bars_in_pos[symbol] = self._bars_in_pos.get(symbol, 0) + 1

            should_exit = False
            reason = ""

            if abs(z) < z_exit:
                should_exit = True
                reason = f"Funding正常化(z={z:.2f})"

            if not should_exit:
                should_exit, reason = check_atr_exit(
                    side=pos.side.value,
                    close=close,
                    avg_price=pos.avg_price,
                    atr=atr,
                    hold_bars=self._bars_in_pos[symbol],
                    sl_mult=self.get_param("stop_loss_atr_mult"),
                    tp_mult=self.get_param("take_profit_atr_mult"),
                    max_hold=self.get_param("max_hold_bars"),
                )

            if should_exit:
                exit_type = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_type, strength=0.8,
                    price=close, reason=f"FundingAlpha平仓: {reason}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._bars_in_pos[symbol] = 0

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        all_signals: list[Signal] = []
        for symbol in self.config.symbols:
            bar = market_data.get(symbol)
            if bar:
                sigs = await self.on_bar(symbol, bar)
                all_signals.extend(sigs)
        return all_signals
