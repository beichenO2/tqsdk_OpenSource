"""Ornstein-Uhlenbeck Mean Reversion Strategy.

Uses the OU process to model price mean reversion mathematically.
The OU process has three parameters:
  kappa: mean reversion speed (higher = faster reversion)
  mu: equilibrium level (long-run mean)
  sigma: volatility

Trading logic:
1. Estimate OU parameters on rolling window of log prices
2. Compute half-life = ln(2) / kappa
3. Only trade when half-life is in tradeable range (5-50 bars)
4. Enter when Z-score of deviation from mu exceeds threshold
5. Exit when Z-score returns to zero (mean reversion)

Research shows advanced OU frameworks achieve Sharpe 1.82-3.27
with max drawdown 4.3-8.7%.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any

import numpy as np

from strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig
from strategy.indicators import calc_atr
from strategy.registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "ou_window": 100,
    "z_entry": 1.5,
    "z_exit": 0.3,
    "z_stop": 3.5,
    "min_half_life": 5,
    "max_half_life": 50,
    "atr_period": 14,
    "sl_atr_mult": 2.0,
    "max_hold_bars": 60,
    "cooldown_bars": 3,
    "use_log_prices": True,
}


def _estimate_ou_params(prices: list[float], use_log: bool = True) -> dict | None:
    """Estimate OU process parameters via OLS on AR(1)."""
    if len(prices) < 30:
        return None

    if use_log:
        series = [math.log(p) for p in prices if p > 0]
    else:
        series = list(prices)

    if len(series) < 30:
        return None

    y = np.array(series[1:])
    x = np.array(series[:-1])

    n = len(y)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    beta_num = np.sum((x - x_mean) * (y - y_mean))
    beta_den = np.sum((x - x_mean) ** 2)
    if abs(beta_den) < 1e-10:
        return None

    beta = beta_num / beta_den
    alpha = y_mean - beta * x_mean

    if beta >= 1.0 or beta <= 0.0:
        return None

    kappa = -math.log(beta)
    mu = alpha / (1 - beta)
    residuals = y - alpha - beta * x
    sigma = math.sqrt(np.mean(residuals ** 2))

    half_life = math.log(2) / kappa if kappa > 0 else float("inf")

    current = series[-1]
    deviation = current - mu
    ou_std = sigma / math.sqrt(2 * kappa) if kappa > 0 else sigma
    z_score = deviation / ou_std if ou_std > 1e-10 else 0

    return {
        "kappa": kappa,
        "mu": mu,
        "sigma": sigma,
        "half_life": half_life,
        "z_score": z_score,
        "current": current,
    }


@auto_register("ou_mean_reversion")
class OUMeanReversionStrategy(BaseStrategy):
    """Trade mean reversion using OU process parameter estimation."""

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

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._init(symbol)
        c, h, l = bar["close"], bar["high"], bar["low"]
        self._c[symbol].append(c)
        self._h[symbol].append(h)
        self._l[symbol].append(l)

        ou_window = self.get_param("ou_window")
        closes = list(self._c[symbol])
        if len(closes) < ou_window:
            return []

        ou = _estimate_ou_params(closes[-ou_window:], self.get_param("use_log_prices"))
        if ou is None:
            return []

        hl = ou["half_life"]
        min_hl = self.get_param("min_half_life")
        max_hl = self.get_param("max_half_life")
        if hl < min_hl or hl > max_hl:
            return []

        z = ou["z_score"]
        z_entry = self.get_param("z_entry")
        z_exit = self.get_param("z_exit")
        z_stop = self.get_param("z_stop")

        atr = calc_atr(self._h[symbol], self._l[symbol], self._c[symbol], self.get_param("atr_period"))
        if atr is None or atr <= 0:
            return []

        signals: list[Signal] = []
        pos = self.get_position(symbol)
        self._cd[symbol] = max(self._cd.get(symbol, 0) - 1, 0)

        if pos is None:
            if self._cd.get(symbol, 0) > 0:
                return signals

            if z < -z_entry:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=min(abs(z) / 3, 1.0), price=c,
                    reason=f"OU LONG z={z:.2f} hl={hl:.1f} kappa={ou['kappa']:.3f}",
                    metadata={"z_score": z, "half_life": hl, "kappa": ou["kappa"], "mu": ou["mu"]},
                ))
                self._hold[symbol] = 0
                self._entry[symbol] = c

            elif z > z_entry:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=min(abs(z) / 3, 1.0), price=c,
                    reason=f"OU SHORT z={z:.2f} hl={hl:.1f} kappa={ou['kappa']:.3f}",
                    metadata={"z_score": z, "half_life": hl, "kappa": ou["kappa"], "mu": ou["mu"]},
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
            elif abs(z) < z_exit:
                ex, reason = True, f"OU reverted z={z:.2f}"
            elif abs(z) > z_stop:
                ex, reason = True, f"z-stop z={z:.2f}"
            elif pos.side.value == "buy" and c <= entry - atr * self.get_param("sl_atr_mult"):
                ex, reason = True, "SL"
            elif pos.side.value == "sell" and c >= entry + atr * self.get_param("sl_atr_mult"):
                ex, reason = True, "SL"

            if ex:
                et = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=et, strength=0.9, price=c,
                    reason=f"OU: {reason}",
                ))
                self._cd[symbol] = self.get_param("cooldown_bars")

        return signals

    async def generate_signals(self, md: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        for s in self.config.symbols:
            if s in md:
                out.extend(await self.on_bar(s, md[s]))
        return out
