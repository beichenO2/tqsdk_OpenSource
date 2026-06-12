"""Meta-Labeling signal filter (de Prado 2018, updated 2024).

A secondary ML model decides whether to EXECUTE or SKIP each signal
produced by a primary strategy. Does not generate signals itself.

Pipeline:
1. Primary strategy emits signal (LONG/SHORT entry)
2. Triple-Barrier labels each signal outcome (+1 / -1)
3. Feature vector built at signal time (vol, regime, momentum, volume, etc.)
4. RandomForest / GradientBoosting meta-model predicts P(profit)
5. Signal executed only if P(profit) > threshold

Walk-forward: model re-fits every `refit_interval` bars on a rolling window.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig
from strategy.indicators import calc_atr, ema_update, rsi as calc_rsi, check_atr_exit
from strategy.registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "primary_strategy": "trend_ema",
    "barrier_atr_mult_tp": 3.5,
    "barrier_atr_mult_sl": 1.8,
    "barrier_max_bars": 24,
    "meta_threshold": 0.65,
    "lookback_window": 400,
    "refit_interval": 400,
    "min_train_samples": 100,
    "atr_period": 14,
    "ema_fast": 12,
    "ema_slow": 26,
    "ema_trend": 50,
    "rsi_period": 14,
    "vol_lookback": 20,
    "volume_ma_period": 20,
    "model_type": "gbm",
}


@dataclass
class PendingSignal:
    bar_idx: int
    signal: Signal
    entry_price: float
    atr: float
    features: dict[str, float] = field(default_factory=dict)


@auto_register("meta_labeling")
class MetaLabelingStrategy(BaseStrategy):
    """Meta-labeling filter over a built-in primary trend signal.

    The primary signal is a simple EMA crossover (fast > slow, above trend).
    The meta-model learns WHEN that signal works vs when to skip.
    """

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._close: dict[str, deque[float]] = {}
        self._high: dict[str, deque[float]] = {}
        self._low: dict[str, deque[float]] = {}
        self._volume: dict[str, deque[float]] = {}
        self._returns: dict[str, deque[float]] = {}

        self._ema_fast: dict[str, float | None] = {}
        self._ema_slow: dict[str, float | None] = {}
        self._ema_trend: dict[str, float | None] = {}
        self._bar_count: dict[str, int] = {}

        self._pending: dict[str, PendingSignal | None] = {}
        self._train_X: list[list[float]] = []
        self._train_y: list[int] = []
        self._model: Any = None
        self._last_refit_bar: int = 0
        self._total_bars: int = 0

    def _ensure_buffers(self, symbol: str) -> None:
        buf = max(
            self.get_param("lookback_window"),
            self.get_param("vol_lookback"),
            self.get_param("ema_trend"),
        ) + 20
        if symbol not in self._close:
            self._close[symbol] = deque(maxlen=buf)
            self._high[symbol] = deque(maxlen=buf)
            self._low[symbol] = deque(maxlen=buf)
            self._volume[symbol] = deque(maxlen=buf)
            self._returns[symbol] = deque(maxlen=buf)
            self._bar_count[symbol] = 0

    def _calc_features(self, symbol: str, close: float, atr: float) -> dict[str, float]:
        """Build feature vector at signal time."""
        closes = list(self._close[symbol])
        volumes = list(self._volume[symbol])
        returns = list(self._returns[symbol])
        n = len(closes)

        vol_20 = float(np.std(returns[-20:])) if len(returns) >= 20 else 0.0
        vol_5 = float(np.std(returns[-5:])) if len(returns) >= 5 else 0.0

        rsi_p = self.get_param("rsi_period")
        rsi_val = calc_rsi(closes, rsi_p)
        rsi = rsi_val if rsi_val is not None else 50.0

        vol_ma_p = self.get_param("volume_ma_period")
        vol_ratio = 1.0
        if len(volumes) >= vol_ma_p and vol_ma_p > 0:
            vol_ma = sum(volumes[-vol_ma_p:]) / vol_ma_p
            vol_ratio = volumes[-1] / vol_ma if vol_ma > 0 else 1.0

        mom_5 = (closes[-1] / closes[-6] - 1) if n > 6 and closes[-6] > 0 else 0.0
        mom_20 = (closes[-1] / closes[-21] - 1) if n > 21 and closes[-21] > 0 else 0.0

        ema_f = self._ema_fast.get(symbol, close)
        ema_s = self._ema_slow.get(symbol, close)
        ema_spread = (ema_f - ema_s) / close if close > 0 else 0.0

        atr_pct = atr / close if close > 0 else 0.0
        vol_ratio_5_20 = vol_5 / vol_20 if vol_20 > 0 else 1.0

        skew = 0.0
        if len(returns) >= 20:
            r = returns[-20:]
            mean_r = sum(r) / len(r)
            m3 = sum((x - mean_r)**3 for x in r) / len(r)
            std3 = vol_20 ** 3
            skew = m3 / std3 if std3 > 0 else 0.0

        price_pctile = 0.5
        if n > 100:
            recent_100 = closes[-100:]
            price_pctile = sum(1 for c in recent_100 if c <= close) / len(recent_100)

        high_low_range = (closes[-1] - min(closes[-20:])) / (max(closes[-20:]) - min(closes[-20:])) if n >= 20 and max(closes[-20:]) > min(closes[-20:]) else 0.5

        bb_pos = 0.5
        if n >= 20 and vol_20 > 0:
            sma20 = sum(closes[-20:]) / 20
            upper = sma20 + 2 * vol_20 * close
            lower = sma20 - 2 * vol_20 * close
            if upper > lower:
                bb_pos = (close - lower) / (upper - lower)

        return {
            "rsi": rsi,
            "vol_20": vol_20,
            "vol_5": vol_5,
            "vol_ratio_5_20": vol_ratio_5_20,
            "volume_ratio": vol_ratio,
            "mom_5": mom_5,
            "mom_20": mom_20,
            "ema_spread": ema_spread,
            "atr_pct": atr_pct,
            "skew_20": skew,
            "price_pctile": price_pctile,
            "high_low_range": high_low_range,
            "bb_position": bb_pos,
        }

    def _label_outcome(self, symbol: str, pending: PendingSignal) -> int | None:
        """Check if triple barrier has been hit. Returns +1, -1, or None if still open."""
        closes = list(self._close[symbol])
        highs = list(self._high[symbol])
        lows = list(self._low[symbol])
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
            final_price = closes[-1]
            return 1 if ((final_price > entry) == is_long) else -1

        return None

    def _fit_model(self) -> None:
        """Train meta-model on accumulated labels."""
        if len(self._train_X) < self.get_param("min_train_samples"):
            return

        window = self.get_param("lookback_window")
        X = self._train_X[-window:]
        y = self._train_y[-window:]

        X_arr = np.array(X)
        y_arr = np.array(y)

        model_type = self.get_param("model_type")

        try:
            if model_type == "gbm":
                from sklearn.ensemble import GradientBoostingClassifier
                self._model = GradientBoostingClassifier(
                    n_estimators=30,
                    max_depth=3,
                    learning_rate=0.05,
                    min_samples_leaf=10,
                    subsample=0.8,
                    random_state=42,
                )
            else:
                from sklearn.ensemble import RandomForestClassifier
                self._model = RandomForestClassifier(
                    n_estimators=30,
                    max_depth=3,
                    min_samples_leaf=10,
                    random_state=42,
                    class_weight="balanced",
                )

            self._model.fit(X_arr, y_arr)
            acc = self._model.score(X_arr, y_arr)
            pos_rate = sum(1 for v in y if v == 1) / len(y) if y else 0
            logger.info(
                "Meta-model(%s) refit: samples=%d, train_acc=%.3f, pos_rate=%.2f",
                model_type, len(y), acc, pos_rate,
            )
        except ImportError:
            logger.warning("sklearn not available; meta-model disabled")
            self._model = None

    def _predict_proba(self, features: dict[str, float]) -> float:
        """Return P(profit) for a signal. Returns 1.0 if no model yet."""
        if self._model is None:
            return 1.0
        feat_names = sorted(features.keys())
        X = np.array([[features[k] for k in feat_names]])
        try:
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

        prev_close = self._close[symbol][-1] if self._close[symbol] else close
        ret = (close - prev_close) / prev_close if prev_close > 0 else 0.0

        self._close[symbol].append(close)
        self._high[symbol].append(high)
        self._low[symbol].append(low)
        self._volume[symbol].append(vol)
        self._returns[symbol].append(ret)
        self._bar_count[symbol] = self._bar_count.get(symbol, 0) + 1
        self._total_bars += 1

        self._ema_fast[symbol] = ema_update(self._ema_fast.get(symbol), close, self.get_param("ema_fast"))
        self._ema_slow[symbol] = ema_update(self._ema_slow.get(symbol), close, self.get_param("ema_slow"))
        self._ema_trend[symbol] = ema_update(self._ema_trend.get(symbol), close, self.get_param("ema_trend"))

        atr = calc_atr(self._high[symbol], self._low[symbol], self._close[symbol], self.get_param("atr_period"))

        pending = self._pending.get(symbol)
        if pending is not None:
            outcome = self._label_outcome(symbol, pending)
            if outcome is not None:
                feat_names = sorted(pending.features.keys())
                self._train_X.append([pending.features[k] for k in feat_names])
                self._train_y.append(1 if outcome == 1 else 0)
                self._pending[symbol] = None

                if self._total_bars - self._last_refit_bar >= self.get_param("refit_interval"):
                    self._fit_model()
                    self._last_refit_bar = self._total_bars

        signals: list[Signal] = []
        if atr is None or atr <= 0:
            return signals

        ema_f = self._ema_fast.get(symbol)
        ema_s = self._ema_slow.get(symbol)
        ema_t = self._ema_trend.get(symbol)
        if any(v is None for v in (ema_f, ema_s, ema_t)):
            return signals

        pos = self.get_position(symbol)

        if pos is None and self._pending.get(symbol) is None:
            features = self._calc_features(symbol, close, atr)

            vol_ma_p = self.get_param("volume_ma_period")
            volumes = list(self._volume[symbol])
            vol_confirmed = True
            if len(volumes) >= vol_ma_p and vol_ma_p > 0:
                vol_ma = sum(volumes[-vol_ma_p:]) / vol_ma_p
                vol_confirmed = volumes[-1] > vol_ma * 0.8

            bullish = ema_f > ema_s and close > ema_t and vol_confirmed
            bearish = ema_f < ema_s and close < ema_t and vol_confirmed

            if bullish:
                raw_sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY, strength=0.5,
                    price=close, reason="EMA bullish cross",
                )
                proba = self._predict_proba(features)
                if proba >= self.get_param("meta_threshold"):
                    raw_sig.strength = round(min(proba, 1.0), 4)
                    raw_sig.reason = f"META-PASS(p={proba:.3f}) EMA bullish"
                    raw_sig.metadata = {"meta_proba": proba, **features}
                    signals.append(raw_sig)
                    self.record_signal(raw_sig)
                    self._pending[symbol] = PendingSignal(
                        bar_idx=self._bar_count[symbol],
                        signal=raw_sig, entry_price=close, atr=atr,
                        features=features,
                    )
                else:
                    logger.debug("META-SKIP(p=%.3f) bullish signal", proba)

            elif bearish:
                raw_sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY, strength=0.5,
                    price=close, reason="EMA bearish cross",
                )
                proba = self._predict_proba(features)
                if proba >= self.get_param("meta_threshold"):
                    raw_sig.strength = round(min(proba, 1.0), 4)
                    raw_sig.reason = f"META-PASS(p={proba:.3f}) EMA bearish"
                    raw_sig.metadata = {"meta_proba": proba, **features}
                    signals.append(raw_sig)
                    self.record_signal(raw_sig)
                    self._pending[symbol] = PendingSignal(
                        bar_idx=self._bar_count[symbol],
                        signal=raw_sig, entry_price=close, atr=atr,
                        features=features,
                    )
                else:
                    logger.debug("META-SKIP(p=%.3f) bearish signal", proba)

        elif pos is not None:
            hold = self._bar_count.get(symbol, 0) - (self._pending.get(symbol) or PendingSignal(bar_idx=0, signal=Signal(strategy_id="", symbol="", signal_type=SignalType.HOLD, strength=0), entry_price=0, atr=0)).bar_idx
            should_exit, reason = check_atr_exit(
                side=pos.side.value,
                close=close,
                avg_price=pos.avg_price,
                atr=atr,
                hold_bars=hold,
                sl_mult=self.get_param("barrier_atr_mult_sl"),
                tp_mult=self.get_param("barrier_atr_mult_tp"),
                max_hold=self.get_param("barrier_max_bars"),
            )
            if should_exit:
                exit_type = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_type, strength=0.9,
                    price=close, reason=f"META: {reason}",
                )
                signals.append(sig)
                self.record_signal(sig)

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        all_signals: list[Signal] = []
        for symbol in self.config.symbols:
            bar = market_data.get(symbol)
            if bar:
                sigs = await self.on_bar(symbol, bar)
                all_signals.extend(sigs)
        return all_signals
