"""Regime-Adaptive Multi-Factor Strategy — production version.

Productized from research-regime-adaptive optimizer (Round 32, OOS pass).
Uses regime-dependent factor weights to adapt between mean-reversion
and momentum styles based on market conditions.

OOS verified results (2026-04-19):
  BTCUSDT: +14.2% / Sharpe 0.63 / MaxDD 11.3%
  ETHUSDT: +41.9% / Sharpe 0.83 / MaxDD 19.6%
  SOLUSDT: +19.6% / Sharpe 0.58 / MaxDD 18.5%
  Walk-Forward: 12 windows, 75% profitable, mean Sharpe 0.98
  Monte Carlo: 90% survival, Sharpe degradation 46.3%

Parameters frozen from optimizer best trial (score 0.399).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np

from strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig
from strategy.registry import auto_register

OPTIMIZED_PARAMS = {
    "signal_threshold": 0.454,
    "tp_atr_mult": 4.946,
    "sl_atr_mult": 2.570,
    "max_hold_bars": 74,
    "cooldown_bars": 8,
    "position_fraction": 0.432,
    "kelly_fraction": 0.129,
    "atr_period": 17,
    "mr_period": 40,
    "mom_fast": 10,
    "mom_slow": 50,
    "regime_ma_fast": 20,
    "regime_ma_slow": 50,
    "regime_vol_window": 30,
    "factor_weights": {
        0: [0.386, 0.737, 0.0, -0.112, 0.0, 0.0, 0.0],
        1: [-0.791, -0.448, 0.0, 0.605, 0.0, 0.0, 0.0],
    },
    "max_risk_per_trade_pct": 0.02,
    "max_portfolio_risk_pct": 0.06,
    "vol_scale_enabled": True,
    "vol_scale_target": 0.15,
    "vol_lookback": 20,
    "drawdown_scale_enabled": True,
    "drawdown_half_kelly_threshold": 0.15,
    "drawdown_stop_threshold": 0.25,
}


def _ema_val(prev: float | None, val: float, period: int) -> float:
    if prev is None:
        return val
    alpha = 2.0 / (period + 1)
    return alpha * val + (1 - alpha) * prev


def _atr_update(atr_prev: float, high: float, low: float, close_prev: float, period: int) -> float:
    tr = max(high - low, abs(high - close_prev), abs(low - close_prev))
    return (atr_prev * (period - 1) + tr) / period


@auto_register("regime_adaptive")
class RegimeAdaptiveStrategy(BaseStrategy):
    """Multi-factor strategy with regime-dependent weights.

    Factors computed per bar:
      f0: mean_reversion — Z-score of price vs slow MA
      f1: momentum — rate of change (fast EMA - slow EMA) / slow EMA
      f3: vol_regime — normalized BB width compression/expansion

    Active factor indices match optimizer search space: [0, 1, 3].
    Factors 2 (funding), 4 (microstructure), 5 (RSI divergence),
    6 (correlation) are zeroed out in this version.
    """

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**OPTIMIZED_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._closes: dict[str, deque[float]] = {}
        self._highs: dict[str, deque[float]] = {}
        self._lows: dict[str, deque[float]] = {}
        self._ema_fast: dict[str, float | None] = {}
        self._ema_slow: dict[str, float | None] = {}
        self._atr: dict[str, float] = {}
        self._bb_widths: dict[str, deque[float]] = {}

        self._hold_bars: dict[str, int] = {}
        self._cooldown: dict[str, int] = {}
        self._entry_price: dict[str, float] = {}
        self._returns: dict[str, deque[float]] = {}
        self._equity_peak: float = 0.0
        self._current_equity: float = 0.0
        self._buf = 200

    def _init_sym(self, s: str) -> None:
        if s not in self._closes:
            self._closes[s] = deque(maxlen=self._buf)
            self._highs[s] = deque(maxlen=self._buf)
            self._lows[s] = deque(maxlen=self._buf)
            self._bb_widths[s] = deque(maxlen=100)
            self._returns[s] = deque(maxlen=200)

    def compute_position_size(self, symbol: str, signal_strength: float, atr: float, price: float) -> float:
        """Risk-managed position sizing with volatility scaling and drawdown control.

        Components:
        1. Base size from risk budget (max_risk_per_trade_pct / ATR distance)
        2. Signal strength scaling (stronger signal → larger position)
        3. Volatility scaling (target a fixed vol contribution)
        4. Drawdown scaling (halve/stop at thresholds)
        """
        risk_pct = self.get_param("max_risk_per_trade_pct")
        sl_dist = atr * self.get_param("sl_atr_mult")
        if sl_dist <= 0 or price <= 0:
            return 0.0

        base_fraction = risk_pct / (sl_dist / price)
        base_fraction = min(base_fraction, self.get_param("position_fraction"))

        strength_scale = 0.5 + 0.5 * min(signal_strength, 1.0)
        sized = base_fraction * strength_scale

        if self.get_param("vol_scale_enabled"):
            rets = list(self._returns.get(symbol, []))
            vol_lb = self.get_param("vol_lookback")
            if len(rets) >= vol_lb:
                realized_vol = math.sqrt(sum(r*r for r in rets[-vol_lb:]) / vol_lb) * math.sqrt(365 * 6)
                target_vol = self.get_param("vol_scale_target")
                if realized_vol > 0:
                    vol_ratio = target_vol / realized_vol
                    sized *= min(max(vol_ratio, 0.3), 2.0)

        if self.get_param("drawdown_scale_enabled") and self._equity_peak > 0 and self._current_equity > 0:
            dd = (self._equity_peak - self._current_equity) / self._equity_peak
            half_t = self.get_param("drawdown_half_kelly_threshold")
            stop_t = self.get_param("drawdown_stop_threshold")
            if dd >= stop_t:
                return 0.0
            elif dd >= half_t:
                sized *= 0.5

        return min(max(sized, 0.01), self.get_param("position_fraction"))

    def _detect_regime(self, s: str) -> int:
        """0 = trending, 1 = ranging/mean-reverting."""
        closes = list(self._closes[s])
        fast_p = self.get_param("regime_ma_fast")
        slow_p = self.get_param("regime_ma_slow")
        if len(closes) < slow_p:
            return 0

        fast_ma = sum(closes[-fast_p:]) / fast_p
        slow_ma = sum(closes[-slow_p:]) / slow_p
        vol_w = min(self.get_param("regime_vol_window"), len(closes))
        if vol_w < 5:
            return 0

        returns = [(closes[i] - closes[i-1]) / closes[i-1]
                    for i in range(len(closes) - vol_w, len(closes))
                    if closes[i-1] != 0]
        if len(returns) < 2:
            return 0

        vol = math.sqrt(sum(r*r for r in returns) / len(returns))
        trend_strength = abs(fast_ma - slow_ma) / slow_ma if slow_ma > 0 else 0

        if trend_strength > 0.02 and vol < 0.03:
            return 0  # trending
        return 1  # ranging

    def _compute_factors(self, s: str) -> list[float]:
        """Compute the 7-element factor vector (only f0, f1, f3 active)."""
        closes = list(self._closes[s])
        factors = [0.0] * 7

        mr_p = self.get_param("mr_period")
        if len(closes) >= mr_p:
            ma = sum(closes[-mr_p:]) / mr_p
            std = math.sqrt(sum((c - ma)**2 for c in closes[-mr_p:]) / mr_p)
            if std > 1e-10:
                z = (closes[-1] - ma) / std
                factors[0] = max(-1, min(1, -z / 3.0))

        fast_ema = self._ema_fast.get(s)
        slow_ema = self._ema_slow.get(s)
        if fast_ema is not None and slow_ema is not None and slow_ema > 0:
            mom = (fast_ema - slow_ema) / slow_ema
            factors[1] = max(-1, min(1, mom * 50))

        bb_w = list(self._bb_widths[s])
        if len(bb_w) >= 10:
            sorted_w = sorted(bb_w)
            rank = sum(1 for w in sorted_w if w <= bb_w[-1])
            pctile = rank / len(sorted_w)
            factors[3] = (pctile - 0.5) * 2

        return factors

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._init_sym(symbol)
        c, h, l = bar["close"], bar["high"], bar["low"]

        if self._closes[symbol]:
            prev = self._closes[symbol][-1]
            if prev > 0:
                self._returns[symbol].append((c - prev) / prev)

        self._closes[symbol].append(c)
        self._highs[symbol].append(h)
        self._lows[symbol].append(l)

        fast_p = self.get_param("mom_fast")
        slow_p = self.get_param("mom_slow")
        self._ema_fast[symbol] = _ema_val(self._ema_fast.get(symbol), c, fast_p)
        self._ema_slow[symbol] = _ema_val(self._ema_slow.get(symbol), c, slow_p)

        atr_p = self.get_param("atr_period")
        if len(self._closes[symbol]) >= 2:
            prev_c = list(self._closes[symbol])[-2]
            self._atr[symbol] = _atr_update(
                self._atr.get(symbol, abs(h - l)), h, l, prev_c, atr_p
            )

        bb_p = 20
        if len(self._closes[symbol]) >= bb_p:
            window = list(self._closes[symbol])[-bb_p:]
            mid = sum(window) / len(window)
            std = math.sqrt(sum((x - mid)**2 for x in window) / len(window))
            width = (4 * std) / mid if mid > 0 else 0
            self._bb_widths[symbol].append(width)

        if len(self._closes[symbol]) < slow_p + 10:
            return []

        regime = self._detect_regime(symbol)
        factors = self._compute_factors(symbol)
        fw = self.get_param("factor_weights")
        weights = fw.get(regime, fw.get(0, [0]*7))

        combined_signal = sum(f * w for f, w in zip(factors, weights))

        signals: list[Signal] = []
        pos = self.get_position(symbol)
        atr = self._atr.get(symbol, 0)
        threshold = self.get_param("signal_threshold")

        self._cooldown[symbol] = max(self._cooldown.get(symbol, 0) - 1, 0)

        if pos is None:
            if self._cooldown.get(symbol, 0) > 0:
                return signals
            if atr <= 0:
                return signals

            if combined_signal > threshold:
                strength = min(abs(combined_signal), 1.0)
                pos_size = self.compute_position_size(symbol, strength, atr, c)
                if pos_size > 0:
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.LONG_ENTRY,
                        strength=strength,
                        price=c,
                        reason=f"RA LONG sig={combined_signal:.3f} regime={regime} size={pos_size:.1%}",
                        metadata={"regime": regime, "factors": factors[:4], "signal": combined_signal, "position_fraction": pos_size},
                    ))
                    self._hold_bars[symbol] = 0
                    self._entry_price[symbol] = c

            elif combined_signal < -threshold:
                strength = min(abs(combined_signal), 1.0)
                pos_size = self.compute_position_size(symbol, strength, atr, c)
                if pos_size > 0:
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.SHORT_ENTRY,
                        strength=strength,
                        price=c,
                        reason=f"RA SHORT sig={combined_signal:.3f} regime={regime} size={pos_size:.1%}",
                        metadata={"regime": regime, "factors": factors[:4], "signal": combined_signal, "position_fraction": pos_size},
                    ))
                    self._hold_bars[symbol] = 0
                    self._entry_price[symbol] = c

        else:
            self._hold_bars[symbol] = self._hold_bars.get(symbol, 0) + 1
            entry = self._entry_price.get(symbol, c)
            should_exit = False
            reason = ""

            if self._hold_bars[symbol] >= self.get_param("max_hold_bars"):
                should_exit, reason = True, "timeout"
            elif pos.side.value == "buy":
                if c >= entry + atr * self.get_param("tp_atr_mult"):
                    should_exit, reason = True, "TP"
                elif c <= entry - atr * self.get_param("sl_atr_mult"):
                    should_exit, reason = True, "SL"
            elif pos.side.value == "sell":
                if c <= entry - atr * self.get_param("tp_atr_mult"):
                    should_exit, reason = True, "TP"
                elif c >= entry + atr * self.get_param("sl_atr_mult"):
                    should_exit, reason = True, "SL"

            if should_exit:
                exit_type = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_type, strength=0.9, price=c,
                    reason=f"RA EXIT: {reason}",
                ))
                self._cooldown[symbol] = self.get_param("cooldown_bars")

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        all_signals: list[Signal] = []
        for symbol in self.config.symbols:
            bar = market_data.get(symbol)
            if bar:
                sigs = await self.on_bar(symbol, bar)
                all_signals.extend(sigs)
        return all_signals
