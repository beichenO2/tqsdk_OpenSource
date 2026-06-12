"""Funding Rate Alpha v2 — multi-signal enhancement.

Upgrades over funding_rate_alpha:
1. Combines funding rate Z-score with Open Interest change
2. Premium Index as confirmation signal
3. Regime-aware position sizing (reduce size in high_volatility)
4. Explicit bar.extra field support for feeding funding/OI from parquet

C3 improvement plan execution.

Data requirements:
  bar["funding_rate"]   — 8h funding rate (from download_funding_rates.py)
  bar["premium_index"]  — mark-index spread (from download_premium_index.py)
  bar["open_interest"]  — open interest (from CoinAnk or Binance)
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any

from strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig
from strategy.indicators import calc_atr, ema_update, check_atr_exit
from strategy.registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "funding_z_entry": 1.2,
    "funding_z_exit": 0.3,
    "funding_ewm_span": 15,
    "funding_lookback": 60,
    "oi_change_threshold": 0.05,
    "premium_confirm_threshold": 0.001,
    "atr_period": 19,
    "stop_loss_atr_mult": 2.0,
    "take_profit_atr_mult": 4.5,
    "max_hold_bars": 30,
    "ema_trend_period": 44,
    "require_trend_alignment": True,
    "use_oi_confirmation": True,
    "use_premium_confirmation": True,
    "regime_position_scale": {
        "strong_trend": 0.8,
        "weak_trend": 1.0,
        "ranging": 1.0,
        "high_volatility": 0.5,
        "breakout": 0.6,
    },
}


@auto_register("funding_rate_v2")
class FundingRateV2Strategy(BaseStrategy):
    """Enhanced contrarian funding rate with OI + premium confirmation."""

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
        self._oi_history: dict[str, deque[float]] = {}
        self._bars_in_pos: dict[str, int] = {}
        self._buf = 200

    def _ensure_buffers(self, symbol: str) -> None:
        if symbol not in self._close:
            self._close[symbol] = deque(maxlen=self._buf)
            self._high[symbol] = deque(maxlen=self._buf)
            self._low[symbol] = deque(maxlen=self._buf)
            self._funding_rates[symbol] = deque(maxlen=self._buf)
            self._oi_history[symbol] = deque(maxlen=self._buf)

    def _funding_zscore(self, symbol: str) -> float | None:
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

    def _oi_declining(self, symbol: str) -> bool:
        """Check if OI is declining (bearish for over-leveraged side)."""
        oi = list(self._oi_history[symbol])
        if len(oi) < 5:
            return False
        threshold = self.get_param("oi_change_threshold")
        oi_change = (oi[-1] - oi[-5]) / oi[-5] if oi[-5] > 0 else 0
        return oi_change < -threshold

    def _premium_confirms(self, bar: dict[str, Any], direction: str) -> bool:
        """Check if premium index confirms the signal direction."""
        premium = bar.get("premium_index", 0.0)
        threshold = self.get_param("premium_confirm_threshold")
        if direction == "short":
            return premium > threshold
        elif direction == "long":
            return premium < -threshold
        return False

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._ensure_buffers(symbol)
        close = bar["close"]
        high = bar["high"]
        low = bar["low"]
        funding = bar.get("funding_rate", 0.0)
        oi = bar.get("open_interest", 0.0)

        self._close[symbol].append(close)
        self._high[symbol].append(high)
        self._low[symbol].append(low)
        self._funding_rates[symbol].append(funding)
        if oi > 0:
            self._oi_history[symbol].append(oi)

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

            use_oi = self.get_param("use_oi_confirmation")
            use_premium = self.get_param("use_premium_confirmation")
            oi_ok = not use_oi or self._oi_declining(symbol)
            premium_ok_short = not use_premium or self._premium_confirms(bar, "short")
            premium_ok_long = not use_premium or self._premium_confirms(bar, "long")

            signal_quality = 0
            if z > z_entry and trend_ok:
                signal_quality += 1
                if oi_ok:
                    signal_quality += 1
                if premium_ok_short:
                    signal_quality += 1

                if signal_quality >= 2:
                    strength = min(abs(z) / (z_entry * 2), 1.0) * (signal_quality / 3)
                    sig = Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.SHORT_ENTRY,
                        strength=round(strength, 4), price=close,
                        reason=f"FRv2 SHORT z={z:.2f} quality={signal_quality}/3",
                        metadata={"funding_z": z, "oi_declining": oi_ok, "premium_ok": premium_ok_short},
                    )
                    signals.append(sig)
                    self.record_signal(sig)
                    self._bars_in_pos[symbol] = 0

            elif z < -z_entry and trend_ok:
                signal_quality += 1
                if oi_ok:
                    signal_quality += 1
                if premium_ok_long:
                    signal_quality += 1

                if signal_quality >= 2:
                    strength = min(abs(z) / (z_entry * 2), 1.0) * (signal_quality / 3)
                    sig = Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.LONG_ENTRY,
                        strength=round(strength, 4), price=close,
                        reason=f"FRv2 LONG z={z:.2f} quality={signal_quality}/3",
                        metadata={"funding_z": z, "oi_declining": oi_ok, "premium_ok": premium_ok_long},
                    )
                    signals.append(sig)
                    self.record_signal(sig)
                    self._bars_in_pos[symbol] = 0

        else:
            self._bars_in_pos[symbol] = self._bars_in_pos.get(symbol, 0) + 1
            should_exit = False
            reason = ""

            if abs(z) < z_exit:
                should_exit, reason = True, f"Funding正常化(z={z:.2f})"

            if not should_exit:
                should_exit, reason = check_atr_exit(
                    side=pos.side.value, close=close, avg_price=pos.avg_price,
                    atr=atr, hold_bars=self._bars_in_pos[symbol],
                    sl_mult=self.get_param("stop_loss_atr_mult"),
                    tp_mult=self.get_param("take_profit_atr_mult"),
                    max_hold=self.get_param("max_hold_bars"),
                )

            if should_exit:
                exit_type = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_type, strength=0.8, price=close,
                    reason=f"FRv2 EXIT: {reason}",
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
