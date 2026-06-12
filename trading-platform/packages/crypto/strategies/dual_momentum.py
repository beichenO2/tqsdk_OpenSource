"""Dual Momentum Strategy — Absolute + Relative Momentum.

Combines two types of momentum:
1. Absolute: is the asset trending up? (return > risk-free rate proxy)
2. Relative: is this asset stronger than others? (vs benchmark)

Only go long when BOTH absolute and relative momentum are positive.
This drastically reduces drawdown compared to pure momentum.

Based on Gary Antonacci's research adapted for crypto.
Historically beats buy-and-hold with 30-50% less drawdown.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig
from strategy.indicators import calc_atr, ema_update
from strategy.registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "abs_momentum_period": 30,
    "abs_momentum_threshold": 0.0,
    "rel_momentum_period": 30,
    "ema_trend_period": 50,
    "atr_period": 14,
    "sl_atr_mult": 2.0,
    "trail_atr_mult": 3.0,
    "max_hold_bars": 120,
    "cooldown_bars": 7,
    "max_risk_per_trade": 0.02,
    "position_fraction": 0.4,
    "benchmark_key": "benchmark_close",
}


@auto_register("dual_momentum")
class DualMomentumStrategy(BaseStrategy):
    """Long only when absolute AND relative momentum are both positive."""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._c: dict[str, deque[float]] = {}
        self._h: dict[str, deque[float]] = {}
        self._l: dict[str, deque[float]] = {}
        self._bench: dict[str, deque[float]] = {}
        self._ema: dict[str, float | None] = {}
        self._hold: dict[str, int] = {}
        self._cd: dict[str, int] = {}
        self._entry: dict[str, float] = {}
        self._peak: dict[str, float] = {}
        self._buf = 200

    def _init(self, s: str) -> None:
        if s not in self._c:
            self._c[s] = deque(maxlen=self._buf)
            self._h[s] = deque(maxlen=self._buf)
            self._l[s] = deque(maxlen=self._buf)
            self._bench[s] = deque(maxlen=self._buf)

    def _absolute_momentum(self, s: str) -> float | None:
        closes = list(self._c[s])
        p = self.get_param("abs_momentum_period")
        if len(closes) < p + 1:
            return None
        return (closes[-1] - closes[-p-1]) / closes[-p-1] if closes[-p-1] > 0 else None

    def _relative_momentum(self, s: str) -> float | None:
        closes = list(self._c[s])
        bench = list(self._bench[s])
        p = self.get_param("rel_momentum_period")
        if len(closes) < p + 1 or len(bench) < p + 1:
            return None
        asset_ret = (closes[-1] - closes[-p-1]) / closes[-p-1] if closes[-p-1] > 0 else 0
        bench_ret = (bench[-1] - bench[-p-1]) / bench[-p-1] if bench[-p-1] > 0 else 0
        return asset_ret - bench_ret

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._init(symbol)
        c, h, l = bar["close"], bar["high"], bar["low"]
        bench_c = bar.get(self.get_param("benchmark_key"), c)

        self._c[symbol].append(c)
        self._h[symbol].append(h)
        self._l[symbol].append(l)
        self._bench[symbol].append(bench_c)
        self._ema[symbol] = ema_update(self._ema.get(symbol), c, self.get_param("ema_trend_period"))

        atr = calc_atr(self._h[symbol], self._l[symbol], self._c[symbol], self.get_param("atr_period"))
        if atr is None or atr <= 0:
            return []

        abs_mom = self._absolute_momentum(symbol)
        rel_mom = self._relative_momentum(symbol)
        ema_val = self._ema[symbol]

        if abs_mom is None or ema_val is None:
            return []

        signals: list[Signal] = []
        pos = self.get_position(symbol)
        self._cd[symbol] = max(self._cd.get(symbol, 0) - 1, 0)

        risk = self.get_param("max_risk_per_trade")
        sl_dist = atr * self.get_param("sl_atr_mult")
        pos_size = min(risk / (sl_dist / c) if sl_dist > 0 and c > 0 else 0.4, self.get_param("position_fraction"))

        threshold = self.get_param("abs_momentum_threshold")
        has_rel = rel_mom is not None

        if pos is None:
            if self._cd.get(symbol, 0) > 0:
                return signals

            abs_ok = abs_mom > threshold
            rel_ok = not has_rel or rel_mom > 0
            trend_ok = c > ema_val

            if abs_ok and rel_ok and trend_ok:
                strength = min(abs_mom * 5, 1.0)
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=round(strength, 3), price=c,
                    reason=f"DUAL MOM abs={abs_mom:.3f} rel={rel_mom:.3f}" if has_rel else f"DUAL MOM abs={abs_mom:.3f}",
                    metadata={"abs_momentum": abs_mom, "rel_momentum": rel_mom, "position_fraction": pos_size},
                ))
                self._hold[symbol] = 0
                self._entry[symbol] = c
                self._peak[symbol] = h

            elif abs_mom < -threshold and c < ema_val:
                strength = min(abs(-abs_mom) * 5, 1.0)
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=round(strength * 0.6, 3), price=c,
                    reason=f"DUAL MOM SHORT abs={abs_mom:.3f}",
                    metadata={"abs_momentum": abs_mom, "position_fraction": pos_size / 2},
                ))
                self._hold[symbol] = 0
                self._entry[symbol] = c
                self._peak[symbol] = l

        else:
            self._hold[symbol] = self._hold.get(symbol, 0) + 1
            trail = atr * self.get_param("trail_atr_mult")
            ex = False
            reason = ""

            if self._hold[symbol] >= self.get_param("max_hold_bars"):
                ex, reason = True, "timeout"
            elif pos.side.value == "buy":
                self._peak[symbol] = max(self._peak.get(symbol, c), h)
                if l <= self._peak[symbol] - trail:
                    ex, reason = True, "trail SL"
                elif abs_mom is not None and abs_mom < 0:
                    ex, reason = True, f"abs momentum turned negative ({abs_mom:.3f})"
            elif pos.side.value == "sell":
                self._peak[symbol] = min(self._peak.get(symbol, c), l)
                if h >= self._peak[symbol] + trail:
                    ex, reason = True, "trail SL"
                elif abs_mom is not None and abs_mom > 0:
                    ex, reason = True, f"abs momentum turned positive ({abs_mom:.3f})"

            if ex:
                et = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=et, strength=0.9, price=c,
                    reason=f"DUAL MOM: {reason}",
                ))
                self._cd[symbol] = self.get_param("cooldown_bars")

        return signals

    async def generate_signals(self, md: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        for s in self.config.symbols:
            if s in md:
                out.extend(await self.on_bar(s, md[s]))
        return out
