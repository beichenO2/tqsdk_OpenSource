"""MACD Divergence Strategy — Histogram-based reversal detection.

Uses MACD histogram divergence (not RSI) for a different perspective:
- MACD histogram measures momentum acceleration/deceleration
- Divergence between price and histogram signals momentum exhaustion
- Histogram crossing zero confirms momentum shift

Complementary to RSI divergence — uses different math but similar concept.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig
from strategy.indicators import calc_atr
from strategy.registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "lookback": 20,
    "atr_period": 14,
    "sl_atr_mult": 1.5,
    "tp_atr_mult": 3.5,
    "max_hold_bars": 48,
    "cooldown_bars": 5,
    "max_risk_per_trade": 0.02,
    "position_fraction": 0.4,
}


def _ema(data: list[float], period: int) -> list[float]:
    result = [data[0]]
    alpha = 2.0 / (period + 1)
    for i in range(1, len(data)):
        result.append(alpha * data[i] + (1 - alpha) * result[-1])
    return result


@auto_register("macd_divergence")
class MACDDivergenceStrategy(BaseStrategy):
    """Trade MACD histogram divergences."""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._c: dict[str, deque[float]] = {}
        self._h: dict[str, deque[float]] = {}
        self._l: dict[str, deque[float]] = {}
        self._hold: dict[str, int] = {}
        self._cd: dict[str, int] = {}
        self._entry: dict[str, float] = {}
        self._buf = 200

    def _init(self, s: str) -> None:
        if s not in self._c:
            self._c[s] = deque(maxlen=self._buf)
            self._h[s] = deque(maxlen=self._buf)
            self._l[s] = deque(maxlen=self._buf)

    def _compute_macd_hist(self, s: str) -> list[float] | None:
        closes = list(self._c[s])
        slow = self.get_param("macd_slow")
        sig = self.get_param("macd_signal")
        if len(closes) < slow + sig:
            return None

        fast_ema = _ema(closes, self.get_param("macd_fast"))
        slow_ema = _ema(closes, slow)
        macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]
        sig_line = _ema(macd_line, sig)
        hist = [m - s for m, s in zip(macd_line, sig_line)]
        return hist

    def _detect_divergence(self, s: str) -> tuple[str, float] | None:
        closes = list(self._c[s])
        hist = self._compute_macd_hist(s)
        if hist is None or len(hist) < 20:
            return None

        lb = self.get_param("lookback")
        recent_c = closes[-lb:]
        recent_h = hist[-lb:]

        c_min1_idx = min(range(len(recent_c) // 2), key=lambda i: recent_c[i])
        c_min2_idx = min(range(len(recent_c) // 2, len(recent_c)), key=lambda i: recent_c[i])
        h_min1_idx = min(range(len(recent_h) // 2), key=lambda i: recent_h[i])
        h_min2_idx = min(range(len(recent_h) // 2, len(recent_h)), key=lambda i: recent_h[i])

        if recent_c[c_min2_idx] < recent_c[c_min1_idx] and recent_h[h_min2_idx] > recent_h[h_min1_idx]:
            if hist[-1] > hist[-2]:
                return "bullish_divergence", hist[-1]

        c_max1_idx = max(range(len(recent_c) // 2), key=lambda i: recent_c[i])
        c_max2_idx = max(range(len(recent_c) // 2, len(recent_c)), key=lambda i: recent_c[i])
        h_max1_idx = max(range(len(recent_h) // 2), key=lambda i: recent_h[i])
        h_max2_idx = max(range(len(recent_h) // 2, len(recent_h)), key=lambda i: recent_h[i])

        if recent_c[c_max2_idx] > recent_c[c_max1_idx] and recent_h[h_max2_idx] < recent_h[h_max1_idx]:
            if hist[-1] < hist[-2]:
                return "bearish_divergence", hist[-1]

        return None

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._init(symbol)
        c, h, l = bar["close"], bar["high"], bar["low"]
        self._c[symbol].append(c)
        self._h[symbol].append(h)
        self._l[symbol].append(l)

        atr = calc_atr(self._h[symbol], self._l[symbol], self._c[symbol], self.get_param("atr_period"))
        if atr is None or atr <= 0:
            return []

        signals: list[Signal] = []
        pos = self.get_position(symbol)
        self._cd[symbol] = max(self._cd.get(symbol, 0) - 1, 0)

        risk = self.get_param("max_risk_per_trade")
        sl_dist = atr * self.get_param("sl_atr_mult")
        pos_size = min(risk / (sl_dist / c) if sl_dist > 0 and c > 0 else 0.4, self.get_param("position_fraction"))

        if pos is None:
            if self._cd.get(symbol, 0) > 0:
                return signals

            div = self._detect_divergence(symbol)
            if div is None:
                return signals

            div_type, hist_val = div

            if "bullish" in div_type:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY, strength=0.75, price=c,
                    reason=f"MACD DIV {div_type} hist={hist_val:.2f}",
                    metadata={"div_type": div_type, "macd_hist": hist_val, "position_fraction": pos_size},
                ))
                self._hold[symbol] = 0
                self._entry[symbol] = c

            elif "bearish" in div_type:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY, strength=0.75, price=c,
                    reason=f"MACD DIV {div_type} hist={hist_val:.2f}",
                    metadata={"div_type": div_type, "macd_hist": hist_val, "position_fraction": pos_size},
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
                    reason=f"MACD DIV: {reason}",
                ))
                self._cd[symbol] = self.get_param("cooldown_bars")

        return signals

    async def generate_signals(self, md: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        for s in self.config.symbols:
            if s in md:
                out.extend(await self.on_bar(s, md[s]))
        return out
