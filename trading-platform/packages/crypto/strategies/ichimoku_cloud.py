"""Ichimoku Cloud Strategy (crypto-adjusted parameters).

Classic Japanese technical analysis providing trend, momentum,
support/resistance, and confirmation in a single framework.

Uses crypto-adjusted 10-30-60 parameters instead of traditional 9-26-52
to account for 24/7 market (no 5-day workweek assumption).

Entry conditions (all must align for high-quality signal):
1. Price above/below cloud (trend direction)
2. TK Cross (Tenkan crosses Kijun — momentum)
3. Chikou Span confirms (above/below historical price)
4. Cloud ahead is same color (future trend support)

This multi-confirmation approach filters out most false signals.
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
    "tenkan_period": 10,
    "kijun_period": 30,
    "senkou_b_period": 60,
    "displacement": 30,
    "atr_period": 14,
    "sl_atr_mult": 2.0,
    "tp_atr_mult": 5.0,
    "max_hold_bars": 96,
    "cooldown_bars": 6,
    "require_chikou_confirm": True,
    "require_cloud_color": True,
    "min_confirmations": 3,
    "max_risk_per_trade": 0.02,
    "position_fraction": 0.4,
    "trail_atr_mult": 3.0,
    "use_kijun_trail": True,
}


def _period_midpoint(highs: list[float], lows: list[float], period: int) -> float | None:
    if len(highs) < period:
        return None
    h = max(highs[-period:])
    l = min(lows[-period:])
    return (h + l) / 2


@auto_register("ichimoku_cloud")
class IchimokuCloudStrategy(BaseStrategy):
    """Multi-confirmation Ichimoku trend strategy."""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._c: dict[str, deque[float]] = {}
        self._h: dict[str, deque[float]] = {}
        self._l: dict[str, deque[float]] = {}
        self._tenkan: dict[str, deque[float]] = {}
        self._kijun: dict[str, deque[float]] = {}
        self._span_a: dict[str, deque[float]] = {}
        self._span_b: dict[str, deque[float]] = {}
        self._prev_tk_cross: dict[str, str] = {}
        self._hold: dict[str, int] = {}
        self._cd: dict[str, int] = {}
        self._entry: dict[str, float] = {}
        self._peak: dict[str, float] = {}
        self._buf = 300

    def _init(self, s: str) -> None:
        if s not in self._c:
            self._c[s] = deque(maxlen=self._buf)
            self._h[s] = deque(maxlen=self._buf)
            self._l[s] = deque(maxlen=self._buf)
            self._tenkan[s] = deque(maxlen=self._buf)
            self._kijun[s] = deque(maxlen=self._buf)
            self._span_a[s] = deque(maxlen=self._buf)
            self._span_b[s] = deque(maxlen=self._buf)

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._init(symbol)
        c, h, l = bar["close"], bar["high"], bar["low"]
        self._c[symbol].append(c)
        self._h[symbol].append(h)
        self._l[symbol].append(l)

        highs = list(self._h[symbol])
        lows = list(self._l[symbol])
        closes = list(self._c[symbol])

        tenkan = _period_midpoint(highs, lows, self.get_param("tenkan_period"))
        kijun = _period_midpoint(highs, lows, self.get_param("kijun_period"))

        if tenkan is not None:
            self._tenkan[symbol].append(tenkan)
        if kijun is not None:
            self._kijun[symbol].append(kijun)

        if tenkan is not None and kijun is not None:
            span_a = (tenkan + kijun) / 2
            self._span_a[symbol].append(span_a)

        span_b = _period_midpoint(highs, lows, self.get_param("senkou_b_period"))
        if span_b is not None:
            self._span_b[symbol].append(span_b)

        disp = self.get_param("displacement")
        if tenkan is None or kijun is None or len(self._span_a[symbol]) < disp or len(self._span_b[symbol]) < disp:
            return []

        cloud_upper = list(self._span_a[symbol])[-disp]
        cloud_lower = list(self._span_b[symbol])[-disp]
        if cloud_upper < cloud_lower:
            cloud_upper, cloud_lower = cloud_lower, cloud_upper

        future_span_a = self._span_a[symbol][-1] if self._span_a[symbol] else 0
        future_span_b = self._span_b[symbol][-1] if self._span_b[symbol] else 0

        atr = calc_atr(self._h[symbol], self._l[symbol], self._c[symbol], self.get_param("atr_period"))
        if atr is None or atr <= 0:
            return []

        above_cloud = c > cloud_upper
        below_cloud = c < cloud_lower

        prev_tenkan = list(self._tenkan[symbol])[-2] if len(self._tenkan[symbol]) >= 2 else tenkan
        prev_kijun = list(self._kijun[symbol])[-2] if len(self._kijun[symbol]) >= 2 else kijun
        tk_bullish = tenkan > kijun and prev_tenkan <= prev_kijun
        tk_bearish = tenkan < kijun and prev_tenkan >= prev_kijun

        chikou_ok_bull = True
        chikou_ok_bear = True
        if self.get_param("require_chikou_confirm") and len(closes) > disp:
            chikou_ok_bull = c > closes[-disp]
            chikou_ok_bear = c < closes[-disp]

        cloud_green = future_span_a > future_span_b
        cloud_red = future_span_a < future_span_b

        signals: list[Signal] = []
        pos = self.get_position(symbol)
        self._cd[symbol] = max(self._cd.get(symbol, 0) - 1, 0)

        if pos is None:
            if self._cd.get(symbol, 0) > 0:
                return signals

            min_conf = self.get_param("min_confirmations")

            bull_score = sum([above_cloud, tk_bullish, chikou_ok_bull, cloud_green])
            bear_score = sum([below_cloud, tk_bearish, chikou_ok_bear, cloud_red])

            risk = self.get_param("max_risk_per_trade")
            sl_dist = atr * self.get_param("sl_atr_mult")
            pos_size = min(risk / (sl_dist / c) if sl_dist > 0 and c > 0 else 0.4, self.get_param("position_fraction"))

            if bull_score >= min_conf:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=min(bull_score / 4, 1.0), price=c,
                    reason=f"ICHIMOKU LONG {bull_score}/4 confirm",
                    metadata={"tenkan": tenkan, "kijun": kijun, "cloud_upper": cloud_upper, "confirmations": bull_score, "position_fraction": pos_size},
                ))
                self._hold[symbol] = 0
                self._entry[symbol] = c
                self._peak[symbol] = c

            elif bear_score >= min_conf:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=min(bear_score / 4, 1.0), price=c,
                    reason=f"ICHIMOKU SHORT {bear_score}/4 confirm",
                    metadata={"tenkan": tenkan, "kijun": kijun, "cloud_lower": cloud_lower, "confirmations": bear_score, "position_fraction": pos_size},
                ))
                self._hold[symbol] = 0
                self._entry[symbol] = c
                self._peak[symbol] = c

        else:
            self._hold[symbol] = self._hold.get(symbol, 0) + 1
            entry = self._entry.get(symbol, c)
            ex = False
            reason = ""

            trail_dist = atr * self.get_param("trail_atr_mult")

            if self._hold[symbol] >= self.get_param("max_hold_bars"):
                ex, reason = True, "timeout"
            elif pos.side.value == "buy":
                self._peak[symbol] = max(self._peak.get(symbol, c), c)
                trail_stop = self._peak[symbol] - trail_dist
                if c < trail_stop:
                    ex, reason = True, f"trail SL @{trail_stop:.0f}"
                elif self.get_param("use_kijun_trail") and c < kijun and self._hold[symbol] > 10:
                    ex, reason = True, "below Kijun"
                elif c <= entry - atr * self.get_param("sl_atr_mult"):
                    ex, reason = True, "SL"
            elif pos.side.value == "sell":
                self._peak[symbol] = min(self._peak.get(symbol, c), c)
                trail_stop = self._peak[symbol] + trail_dist
                if c > trail_stop:
                    ex, reason = True, f"trail SL @{trail_stop:.0f}"
                elif self.get_param("use_kijun_trail") and c > kijun and self._hold[symbol] > 10:
                    ex, reason = True, "above Kijun"
                elif c >= entry + atr * self.get_param("sl_atr_mult"):
                    ex, reason = True, "SL"

            if ex:
                et = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=et, strength=0.9, price=c,
                    reason=f"ICHIMOKU: {reason}",
                ))
                self._cd[symbol] = self.get_param("cooldown_bars")

        return signals

    async def generate_signals(self, md: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        for s in self.config.symbols:
            if s in md:
                out.extend(await self.on_bar(s, md[s]))
        return out
