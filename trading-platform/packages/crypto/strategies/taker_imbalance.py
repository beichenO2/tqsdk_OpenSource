"""Taker Imbalance Alpha Strategy.

Uses taker buy/sell ratio from Binance OHLCV to detect aggressive
directional flow without needing Level2 orderbook data.

Alpha source (Delphi Alpha Report 2026):
Queue Imbalance IC=0.065, Microprice Bias IC=0.061 at 60s horizons.
Our adaptation uses 4h taker ratio as proxy for institutional flow.

Strategy:
1. Compute rolling taker buy ratio = taker_buy_vol / total_vol
2. When ratio diverges from price → hidden accumulation/distribution
3. Taker ratio rising + price flat/falling = hidden buying (bullish)
4. Taker ratio falling + price flat/rising = hidden selling (bearish)
5. Enter on divergence confirmation, exit on convergence or ATR stop

Key insight: taker_buy_volume captures market orders (aggressive side).
Persistent taker buying despite flat price = absorption by limit sellers.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any

from strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig
from strategy.indicators import calc_atr, ema_update
from strategy.registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "taker_lookback": 20,
    "taker_trend_period": 10,
    "price_flat_threshold": 0.005,
    "taker_divergence_threshold": 0.06,
    "atr_period": 14,
    "sl_atr_mult": 2.0,
    "tp_atr_mult": 4.0,
    "max_hold_bars": 48,
    "cooldown_bars": 5,
    "ema_filter_period": 50,
    "min_volume_ratio": 0.8,
}


@auto_register("taker_imbalance")
class TakerImbalanceStrategy(BaseStrategy):
    """Detect hidden accumulation/distribution via taker flow divergence."""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._c: dict[str, deque[float]] = {}
        self._h: dict[str, deque[float]] = {}
        self._l: dict[str, deque[float]] = {}
        self._v: dict[str, deque[float]] = {}
        self._tbr: dict[str, deque[float]] = {}
        self._tbr_ema: dict[str, float | None] = {}
        self._price_ema: dict[str, float | None] = {}
        self._hold: dict[str, int] = {}
        self._cd: dict[str, int] = {}
        self._entry: dict[str, float] = {}
        self._buf = 200

    def _init(self, s: str) -> None:
        if s not in self._c:
            self._c[s] = deque(maxlen=self._buf)
            self._h[s] = deque(maxlen=self._buf)
            self._l[s] = deque(maxlen=self._buf)
            self._v[s] = deque(maxlen=self._buf)
            self._tbr[s] = deque(maxlen=self._buf)

    def _compute_divergence(self, s: str) -> tuple[str, float] | None:
        """Detect taker ratio vs price divergence."""
        tbr_list = list(self._tbr[s])
        closes = list(self._c[s])
        p = self.get_param("taker_lookback")
        if len(tbr_list) < p or len(closes) < p:
            return None

        tbr_change = sum(tbr_list[-p:]) / p - sum(tbr_list[-2*p:-p]) / p if len(tbr_list) >= 2*p else 0
        price_change = (closes[-1] - closes[-p]) / closes[-p] if closes[-p] > 0 else 0

        div_threshold = self.get_param("taker_divergence_threshold")
        flat_threshold = self.get_param("price_flat_threshold")

        if tbr_change > div_threshold and abs(price_change) < flat_threshold:
            return "bullish_divergence", tbr_change
        if tbr_change < -div_threshold and abs(price_change) < flat_threshold:
            return "bearish_divergence", tbr_change
        if tbr_change > div_threshold and price_change < -flat_threshold:
            return "bullish_divergence_strong", tbr_change
        if tbr_change < -div_threshold and price_change > flat_threshold:
            return "bearish_divergence_strong", tbr_change

        return None

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._init(symbol)
        c, h, l = bar["close"], bar["high"], bar["low"]
        vol = bar.get("volume", 0.0)
        tbv = bar.get("taker_buy_volume", vol * 0.5)

        self._c[symbol].append(c)
        self._h[symbol].append(h)
        self._l[symbol].append(l)
        self._v[symbol].append(vol)

        tbr = tbv / vol if vol > 0 else 0.5
        self._tbr[symbol].append(tbr)

        tp = self.get_param("taker_trend_period")
        self._tbr_ema[symbol] = ema_update(self._tbr_ema.get(symbol), tbr, tp)
        self._price_ema[symbol] = ema_update(self._price_ema.get(symbol), c, self.get_param("ema_filter_period"))

        atr = calc_atr(self._h[symbol], self._l[symbol], self._c[symbol], self.get_param("atr_period"))
        if atr is None or atr <= 0:
            return []

        if len(self._tbr[symbol]) < self.get_param("taker_lookback") * 2:
            return []

        signals: list[Signal] = []
        pos = self.get_position(symbol)
        self._cd[symbol] = max(self._cd.get(symbol, 0) - 1, 0)

        divergence = self._compute_divergence(symbol)

        if pos is None:
            if self._cd.get(symbol, 0) > 0:
                return signals
            if divergence is None:
                return signals

            div_type, div_strength = divergence

            if "bullish" in div_type:
                strength = min(abs(div_strength) / 0.1, 1.0)
                strong = "strong" in div_type
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=0.9 if strong else 0.7, price=c,
                    reason=f"TAKER {div_type} str={div_strength:.3f}",
                    metadata={"div_type": div_type, "taker_ratio": tbr, "div_strength": div_strength},
                ))
                self._hold[symbol] = 0
                self._entry[symbol] = c

            elif "bearish" in div_type:
                strength = min(abs(div_strength) / 0.1, 1.0)
                strong = "strong" in div_type
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=0.9 if strong else 0.7, price=c,
                    reason=f"TAKER {div_type} str={div_strength:.3f}",
                    metadata={"div_type": div_type, "taker_ratio": tbr, "div_strength": div_strength},
                ))
                self._hold[symbol] = 0
                self._entry[symbol] = c

        else:
            self._hold[symbol] = self._hold.get(symbol, 0) + 1
            entry = self._entry.get(symbol, c)
            ex = False
            reason = ""

            if self._hold[symbol] >= self.get_param("max_hold_bars"):
                ex, reason = True, "timeout"
            elif pos.side.value == "buy":
                if c <= entry - atr * self.get_param("sl_atr_mult"):
                    ex, reason = True, "SL"
                elif c >= entry + atr * self.get_param("tp_atr_mult"):
                    ex, reason = True, "TP"
            elif pos.side.value == "sell":
                if c >= entry + atr * self.get_param("sl_atr_mult"):
                    ex, reason = True, "SL"
                elif c <= entry - atr * self.get_param("tp_atr_mult"):
                    ex, reason = True, "TP"

            if ex:
                et = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=et, strength=0.9, price=c,
                    reason=f"TAKER: {reason}",
                ))
                self._cd[symbol] = self.get_param("cooldown_bars")

        return signals

    async def generate_signals(self, md: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        for s in self.config.symbols:
            if s in md:
                out.extend(await self.on_bar(s, md[s]))
        return out
