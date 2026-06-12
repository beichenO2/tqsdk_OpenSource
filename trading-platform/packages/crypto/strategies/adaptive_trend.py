"""Adaptive Trend-Following Strategy (inspired by arXiv:2602.11708).

Core ideas from the AdaptiveTrend paper (Bui & Nguyen, 2026):
1. Volatility-calibrated trailing stop (adapts to intra-bar volatility regime)
2. Rolling Sharpe-based asset selection (focus on trending assets)
3. Asymmetric 70/30 long-short allocation (crypto has positive drift)
4. 6-hour signal frequency (sweet spot between noise and responsiveness)

Our adaptation:
- Single-asset version (portfolio construction handled by PortfolioStrategy)
- Uses MarketRegimeDetector for volatility regime classification
- ATR-based trailing stop calibrated per regime
- Donchian channel breakout as core signal (paper uses momentum)
- Rolling Sharpe for signal quality assessment

Target: Sharpe > 1.0 on BTC 4h (paper achieves 2.41 on 6h multi-asset).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig
from strategy.indicators import calc_atr, ema_update
from strategy.registry import auto_register
from .regime_detector import MarketRegimeDetector, MarketRegime

DEFAULT_PARAMS = {
    "donchian_period": 20,
    "donchian_exit_period": 10,
    "atr_period": 14,
    "ema_filter_period": 100,
    "rolling_sharpe_window": 60,
    "min_rolling_sharpe": -0.5,
    "max_hold_bars": 168,
    "cooldown_bars": 6,
    "long_allocation": 0.7,
    "short_allocation": 0.3,
    "regime_trail_mult": {
        "strong_trend": 3.0,
        "weak_trend": 2.5,
        "ranging": 1.5,
        "high_volatility": 4.0,
        "breakout": 2.0,
        "unknown": 2.5,
    },
    "use_regime_gate": True,
    "blocked_regimes": ["ranging"],
}


@auto_register("adaptive_trend")
class AdaptiveTrendStrategy(BaseStrategy):
    """Volatility-adaptive trend following with Donchian breakout."""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._c: dict[str, deque[float]] = {}
        self._h: dict[str, deque[float]] = {}
        self._l: dict[str, deque[float]] = {}
        self._regime: dict[str, MarketRegimeDetector] = {}
        self._ema: dict[str, float | None] = {}
        self._returns: dict[str, deque[float]] = {}
        self._hold: dict[str, int] = {}
        self._cd: dict[str, int] = {}
        self._trail: dict[str, float] = {}
        self._peak: dict[str, float] = {}
        self._entry: dict[str, float] = {}
        self._buf = 300

    def _init(self, s: str) -> None:
        if s not in self._c:
            self._c[s] = deque(maxlen=self._buf)
            self._h[s] = deque(maxlen=self._buf)
            self._l[s] = deque(maxlen=self._buf)
            self._returns[s] = deque(maxlen=200)
            self._regime[s] = MarketRegimeDetector()

    def _donchian(self, s: str, period: int) -> tuple[float, float] | None:
        """Return (upper, lower) Donchian channel."""
        highs = list(self._h[s])
        lows = list(self._l[s])
        if len(highs) < period:
            return None
        return max(highs[-period:]), min(lows[-period:])

    def _rolling_sharpe(self, s: str) -> float | None:
        """Rolling Sharpe ratio for signal quality filtering."""
        rets = list(self._returns[s])
        w = self.get_param("rolling_sharpe_window")
        if len(rets) < w:
            return None
        window = rets[-w:]
        mean_r = sum(window) / len(window)
        var_r = sum((r - mean_r)**2 for r in window) / len(window)
        std_r = math.sqrt(var_r) if var_r > 0 else 0.001
        annualized = mean_r / std_r * math.sqrt(365 * 6) if std_r > 0 else 0
        return annualized

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._init(symbol)
        c, h, l = bar["close"], bar["high"], bar["low"]

        if self._c[symbol]:
            prev_c = self._c[symbol][-1]
            if prev_c > 0:
                self._returns[symbol].append((c - prev_c) / prev_c)

        self._c[symbol].append(c)
        self._h[symbol].append(h)
        self._l[symbol].append(l)
        self._regime[symbol].update(h, l, c)
        self._ema[symbol] = ema_update(self._ema.get(symbol), c, self.get_param("ema_filter_period"))

        don_entry = self._donchian(symbol, self.get_param("donchian_period"))
        don_exit = self._donchian(symbol, self.get_param("donchian_exit_period"))
        ema_val = self._ema[symbol]
        atr = calc_atr(self._h[symbol], self._l[symbol], self._c[symbol], self.get_param("atr_period"))

        if don_entry is None or don_exit is None or ema_val is None or atr is None or atr <= 0:
            return []

        regime = self._regime[symbol].current_regime
        don_upper, don_lower = don_entry
        exit_upper, exit_lower = don_exit

        signals: list[Signal] = []
        pos = self.get_position(symbol)
        self._cd[symbol] = max(self._cd.get(symbol, 0) - 1, 0)

        if pos is None:
            if self._cd.get(symbol, 0) > 0:
                return signals

            if self.get_param("use_regime_gate"):
                if regime.value in self.get_param("blocked_regimes"):
                    return signals

            rolling_s = self._rolling_sharpe(symbol)
            min_sharpe = self.get_param("min_rolling_sharpe")
            if rolling_s is not None and rolling_s < min_sharpe:
                return signals

            if c > don_upper and c > ema_val:
                alloc = self.get_param("long_allocation")
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=round(alloc, 2), price=c,
                    reason=f"AT LONG breakout [{regime.value}] don={don_upper:.0f}",
                    metadata={"regime": regime.value, "donchian_upper": don_upper, "atr": atr},
                ))
                self._hold[symbol] = 0
                self._peak[symbol] = h
                self._entry[symbol] = c

            elif c < don_lower and c < ema_val:
                alloc = self.get_param("short_allocation")
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=round(alloc, 2), price=c,
                    reason=f"AT SHORT breakdown [{regime.value}] don={don_lower:.0f}",
                    metadata={"regime": regime.value, "donchian_lower": don_lower, "atr": atr},
                ))
                self._hold[symbol] = 0
                self._peak[symbol] = l
                self._entry[symbol] = c

        else:
            self._hold[symbol] = self._hold.get(symbol, 0) + 1
            trail_map = self.get_param("regime_trail_mult")
            trail_mult = trail_map.get(regime.value, 2.5)
            trail_dist = atr * trail_mult

            ex = False
            reason = ""

            if self._hold[symbol] >= self.get_param("max_hold_bars"):
                ex, reason = True, "timeout"
            elif pos.side.value == "buy":
                self._peak[symbol] = max(self._peak.get(symbol, c), h)
                trail_stop = self._peak[symbol] - trail_dist
                if l <= trail_stop:
                    ex, reason = True, f"trail SL @{trail_stop:.0f} [{regime.value}]"
                elif c < exit_lower:
                    ex, reason = True, "Donchian exit channel"
            elif pos.side.value == "sell":
                self._peak[symbol] = min(self._peak.get(symbol, c), l)
                trail_stop = self._peak[symbol] + trail_dist
                if h >= trail_stop:
                    ex, reason = True, f"trail SL @{trail_stop:.0f} [{regime.value}]"
                elif c > exit_upper:
                    ex, reason = True, "Donchian exit channel"

            if ex:
                et = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=et, strength=0.9, price=c,
                    reason=f"AT: {reason}",
                ))
                self._cd[symbol] = self.get_param("cooldown_bars")

        return signals

    async def generate_signals(self, md: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        for s in self.config.symbols:
            if s in md:
                out.extend(await self.on_bar(s, md[s]))
        return out
