"""Funding Rate + Meta-Labeling Ensemble (2025).

Combines the two best-performing strategies:
1. Funding Rate Alpha: contrarian signals from premium index
2. Meta-Labeling: ML filter to decide EXECUTE vs SKIP

The idea: funding rate generates raw signals, meta-model learns when
those signals are more likely to succeed based on market context.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig
from strategy.indicators import calc_atr, ema_update, rsi as calc_rsi, check_atr_exit
from strategy.registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "funding_z_entry": 1.3,
    "funding_z_exit": 0.25,
    "funding_ewm_span": 12,
    "funding_lookback": 60,
    "atr_period": 14,
    "stop_loss_atr_mult": 2.0,
    "take_profit_atr_mult": 5.0,
    "max_hold_bars": 36,
    "ema_trend": 50,
    "rsi_period": 14,
    "vol_lookback": 20,
    "volume_ma_period": 20,
    "meta_threshold": 0.55,
    "lookback_window": 300,
    "refit_interval": 200,
    "min_train_samples": 35,
    "barrier_atr_mult_tp": 3.5,
    "barrier_atr_mult_sl": 1.8,
    "barrier_max_bars": 24,
    "embargo_bars": 6,
}


@dataclass
class PendingLabel:
    bar_idx: int
    signal: Signal
    entry_price: float
    atr: float
    features: list[float] = field(default_factory=list)


@auto_register("funding_meta_ensemble")
class FundingMetaEnsembleStrategy(BaseStrategy):
    """Funding Rate alpha + Meta-Labeling filter."""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._close: dict[str, deque[float]] = {}
        self._high: dict[str, deque[float]] = {}
        self._low: dict[str, deque[float]] = {}
        self._volume: dict[str, deque[float]] = {}
        self._returns: dict[str, deque[float]] = {}
        self._funding_rates: dict[str, deque[float]] = {}
        self._funding_ewm: dict[str, float | None] = {}
        self._ema_trend: dict[str, float | None] = {}
        self._bars_in_pos: dict[str, int] = {}
        self._bar_count: dict[str, int] = {}

        self._pending_label: dict[str, PendingLabel | None] = {}
        self._train_X: list[list[float]] = []
        self._train_y: list[int] = []
        self._model: Any = None
        self._last_refit: int = 0
        self._total_bars: int = 0

    def _ensure_buffers(self, symbol: str) -> None:
        buf = 300
        if symbol not in self._close:
            self._close[symbol] = deque(maxlen=buf)
            self._high[symbol] = deque(maxlen=buf)
            self._low[symbol] = deque(maxlen=buf)
            self._volume[symbol] = deque(maxlen=buf)
            self._returns[symbol] = deque(maxlen=buf)
            self._funding_rates[symbol] = deque(maxlen=buf)
            self._bar_count[symbol] = 0

    def _funding_zscore(self, symbol: str) -> float | None:
        rates = list(self._funding_rates[symbol])
        lookback = self.get_param("funding_lookback")
        if len(rates) < lookback:
            return None
        window = rates[-lookback:]
        mean_f = sum(window) / len(window)
        var_f = sum((r - mean_f) ** 2 for r in window) / len(window)
        std_f = math.sqrt(var_f) if var_f > 0 else 0.0
        if std_f < 1e-10:
            return None
        current = self._funding_ewm.get(symbol, 0.0) or 0.0
        return max(-10.0, min(10.0, (current - mean_f) / std_f))

    def _calc_features(self, symbol: str, close: float, atr: float, z: float) -> list[float]:
        closes = list(self._close[symbol])
        volumes = list(self._volume[symbol])
        returns = list(self._returns[symbol])
        funding_rates = list(self._funding_rates[symbol])
        n = len(closes)

        vol_20 = float(np.std(returns[-20:])) if len(returns) >= 20 else 0.0
        vol_5 = float(np.std(returns[-5:])) if len(returns) >= 5 else 0.0
        vol_ratio = vol_5 / vol_20 if vol_20 > 0 else 1.0

        rsi_p = self.get_param("rsi_period")
        rsi_val = calc_rsi(closes, rsi_p)
        rsi = rsi_val if rsi_val is not None else 50.0

        vol_ma_p = self.get_param("volume_ma_period")
        vol_r = 1.0
        if len(volumes) >= vol_ma_p and vol_ma_p > 0:
            vol_ma = sum(volumes[-vol_ma_p:]) / vol_ma_p
            vol_r = volumes[-1] / vol_ma if vol_ma > 0 else 1.0

        mom_5 = (closes[-1] / closes[-6] - 1) if n > 6 and closes[-6] > 0 else 0.0
        mom_20 = (closes[-1] / closes[-21] - 1) if n > 21 and closes[-21] > 0 else 0.0

        ema_t = self._ema_trend.get(symbol, close)
        ema_dist = (close - ema_t) / close if close > 0 else 0.0

        atr_pct = atr / close if close > 0 else 0.0

        fr_trend = 0.0
        if len(funding_rates) >= 10:
            fr_trend = sum(funding_rates[-5:]) / 5 - sum(funding_rates[-10:-5]) / 5

        return [z, rsi, vol_20, vol_5, vol_ratio, vol_r, mom_5, mom_20,
                ema_dist, atr_pct, fr_trend]

    def _label_outcome(self, symbol: str, pending: PendingLabel) -> int | None:
        highs = list(self._high[symbol])
        lows = list(self._low[symbol])
        closes = list(self._close[symbol])
        bars_elapsed = self._bar_count[symbol] - pending.bar_idx

        tp_dist = pending.atr * self.get_param("barrier_atr_mult_tp")
        sl_dist = pending.atr * self.get_param("barrier_atr_mult_sl")
        max_bars = self.get_param("barrier_max_bars")
        is_long = pending.signal.signal_type == SignalType.LONG_ENTRY
        entry = pending.entry_price

        if is_long:
            if highs[-1] >= entry + tp_dist:
                return 1
            if lows[-1] <= entry - sl_dist:
                return -1
        else:
            if lows[-1] <= entry - tp_dist:
                return 1
            if highs[-1] >= entry + sl_dist:
                return -1

        if bars_elapsed >= max_bars:
            return 1 if ((closes[-1] > entry) == is_long) else -1
        return None

    def _fit_model(self) -> None:
        if len(self._train_X) < self.get_param("min_train_samples"):
            return
        window = self.get_param("lookback_window")
        embargo = self.get_param("embargo_bars")
        n = len(self._train_X)
        end_idx = max(0, n - embargo)
        start_idx = max(0, end_idx - window)
        if end_idx - start_idx < self.get_param("min_train_samples"):
            return
        X = np.array(self._train_X[start_idx:end_idx])
        y = np.array(self._train_y[start_idx:end_idx])

        try:
            from sklearn.ensemble import GradientBoostingClassifier
            self._model = GradientBoostingClassifier(
                n_estimators=20, max_depth=2, learning_rate=0.05,
                min_samples_leaf=8, subsample=0.8, random_state=42,
            )
            self._model.fit(X, y)
            pos_rate = sum(1 for v in y if v == 1) / len(y)
            logger.info(
                "Ensemble meta refit: samples=%d (embargo=%d), pos_rate=%.2f",
                len(y), embargo, pos_rate,
            )
        except ImportError:
            self._model = None

    def _predict_proba(self, features: list[float]) -> float:
        if self._model is None:
            return 1.0
        try:
            X = np.array([features])
            proba = self._model.predict_proba(X)[0]
            idx = list(self._model.classes_).index(1)
            return float(proba[idx])
        except Exception:
            return 1.0

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._ensure_buffers(symbol)
        close = bar["close"]
        high = bar["high"]
        low = bar["low"]
        vol = bar.get("volume", 0.0)
        funding = bar.get("funding_rate", 0.0)

        prev = self._close[symbol][-1] if self._close[symbol] else close
        ret = (close - prev) / prev if prev > 0 else 0.0

        self._close[symbol].append(close)
        self._high[symbol].append(high)
        self._low[symbol].append(low)
        self._volume[symbol].append(vol)
        self._returns[symbol].append(ret)
        self._funding_rates[symbol].append(funding)
        self._bar_count[symbol] = self._bar_count.get(symbol, 0) + 1
        self._total_bars += 1

        span = self.get_param("funding_ewm_span")
        alpha = 2.0 / (span + 1)
        prev_ewm = self._funding_ewm.get(symbol)
        self._funding_ewm[symbol] = funding if prev_ewm is None else alpha * funding + (1 - alpha) * prev_ewm

        self._ema_trend[symbol] = ema_update(
            self._ema_trend.get(symbol), close, self.get_param("ema_trend"),
        )

        atr = calc_atr(self._high[symbol], self._low[symbol], self._close[symbol], self.get_param("atr_period"))

        pending = self._pending_label.get(symbol)
        if pending is not None:
            outcome = self._label_outcome(symbol, pending)
            if outcome is not None:
                self._train_X.append(pending.features)
                self._train_y.append(1 if outcome == 1 else 0)
                self._pending_label[symbol] = None
                if self._total_bars - self._last_refit >= self.get_param("refit_interval"):
                    self._fit_model()
                    self._last_refit = self._total_bars

        signals: list[Signal] = []
        if atr is None or atr <= 0:
            return signals

        z = self._funding_zscore(symbol)
        if z is None:
            return signals

        pos = self.get_position(symbol)
        z_entry = self.get_param("funding_z_entry")
        z_exit = self.get_param("funding_z_exit")

        if pos is None and self._pending_label.get(symbol) is None:
            direction = None
            if z > z_entry:
                direction = "short"
            elif z < -z_entry:
                direction = "long"

            if direction:
                features = self._calc_features(symbol, close, atr, z)
                proba = self._predict_proba(features)

                if proba >= self.get_param("meta_threshold"):
                    sig_type = SignalType.LONG_ENTRY if direction == "long" else SignalType.SHORT_ENTRY
                    strength = round(min(proba * abs(z) / (z_entry * 3), 1.0), 4)
                    sig = Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=sig_type, strength=strength, price=close,
                        reason=f"FundMeta {'做多' if direction == 'long' else '做空'}(z={z:.2f}, p={proba:.3f})",
                        metadata={"funding_z": z, "meta_proba": proba},
                    )
                    signals.append(sig)
                    self.record_signal(sig)
                    self._bars_in_pos[symbol] = 0
                    self._pending_label[symbol] = PendingLabel(
                        bar_idx=self._bar_count[symbol], signal=sig,
                        entry_price=close, atr=atr, features=features,
                    )
        else:
            if pos is not None:
                self._bars_in_pos[symbol] = self._bars_in_pos.get(symbol, 0) + 1

                should_exit = False
                reason = ""

                if abs(z) < z_exit:
                    should_exit, reason = True, f"Funding正常化(z={z:.2f})"

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
                        price=close, reason=f"FundMeta平仓: {reason}",
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
