"""ML-driven BTC strategy — uses trained LightGBM/XGBoost for signal generation.

Loads a pre-trained crypto model and generates trading signals based on
real-time feature computation and model predictions. Integrates with
the regime detector for position sizing.
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
from .regime_detector import MarketRegimeDetector

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "model_path": "",
    "model_framework": "lightgbm",
    "prediction_threshold": 0.55,
    "signal_strength_scale": 1.0,
    "feature_lookback": 50,
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal_period": 9,
    "bb_period": 20,
    "bb_std": 2.0,
    "atr_period": 14,
    "stop_loss_atr_mult": 2.0,
    "take_profit_atr_mult": 3.5,
    "cooldown_bars": 3,
    "enable_regime_filter": True,
}


def _ema(prev: float | None, val: float, p: int) -> float:
    if prev is None:
        return val
    k = 2.0 / (p + 1)
    return val * k + prev * (1 - k)


@auto_register("btc_crypto_ml")
class BTCCryptoMLStrategy(BaseStrategy):
    """BTC strategy powered by ML model predictions."""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._model: Any = None
        self._model_loaded = False
        self._feature_columns: list[str] = []

        buf = self.get_param("feature_lookback") + 10
        self._close: dict[str, deque[float]] = {}
        self._high: dict[str, deque[float]] = {}
        self._low: dict[str, deque[float]] = {}
        self._volume: dict[str, deque[float]] = {}
        self._taker_buy: dict[str, deque[float]] = {}
        self._obv: dict[str, float] = {}
        self._macd_fast_ema: dict[str, float | None] = {}
        self._macd_slow_ema: dict[str, float | None] = {}
        self._macd_sig_ema: dict[str, float | None] = {}
        self._regime: dict[str, MarketRegimeDetector] = {}
        self._cooldown: dict[str, int] = {}
        self._buf = buf

    async def on_start(self) -> None:
        await super().on_start()
        model_path = self.get_param("model_path")
        if model_path:
            self.load_model(model_path)

    def load_model(self, path: str | None = None) -> None:
        """Load a pre-trained model from disk."""
        model_path = path or self.get_param("model_path")
        if not model_path:
            logger.warning("No model path specified; strategy will output no signals")
            return

        fw = self.get_param("model_framework")
        try:
            from ml.base import MLFramework, MLModelMeta
            meta = MLModelMeta(
                model_id="crypto_ml_live",
                name=f"Live {fw} crypto",
                framework=MLFramework(fw),
            )
            if fw == "lightgbm":
                from ml.lightgbm_model import LightGBMModel
                self._model = LightGBMModel(meta)
            else:
                from ml.xgboost_model import XGBoostModel
                self._model = XGBoostModel(meta)

            self._model.load(model_path)
            self._feature_columns = list(self._model.meta.feature_columns)
            self._model_loaded = True
            logger.info("Loaded %s model from %s", fw, model_path)
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            self._model_loaded = False

    def _ensure_buffers(self, symbol: str) -> None:
        if symbol not in self._close:
            self._close[symbol] = deque(maxlen=self._buf)
            self._high[symbol] = deque(maxlen=self._buf)
            self._low[symbol] = deque(maxlen=self._buf)
            self._volume[symbol] = deque(maxlen=self._buf)
            self._taker_buy[symbol] = deque(maxlen=self._buf)
            self._obv[symbol] = 0.0
            self._macd_fast_ema[symbol] = None
            self._macd_slow_ema[symbol] = None
            self._macd_sig_ema[symbol] = None
            self._regime[symbol] = MarketRegimeDetector()
            self._cooldown[symbol] = 0

    def _compute_features(self, symbol: str, bar: dict[str, Any]) -> np.ndarray | None:
        """Compute feature vector from buffered data for model prediction."""
        closes = list(self._close[symbol])
        if len(closes) < 30:
            return None

        features: dict[str, float] = {}

        period = self.get_param("rsi_period")
        if len(closes) > period:
            gains, losses = [], []
            for i in range(-period, 0):
                d = closes[i] - closes[i - 1]
                gains.append(max(d, 0))
                losses.append(max(-d, 0))
            ag = sum(gains) / period
            al = sum(losses) / period
            features["rsi"] = 100 - 100 / (1 + ag / al) if al > 0 else 100.0
        else:
            features["rsi"] = 50.0

        close = bar["close"]
        self._macd_fast_ema[symbol] = _ema(self._macd_fast_ema[symbol], close, self.get_param("macd_fast"))
        self._macd_slow_ema[symbol] = _ema(self._macd_slow_ema[symbol], close, self.get_param("macd_slow"))
        fe = self._macd_fast_ema[symbol] or 0
        se = self._macd_slow_ema[symbol] or 0
        macd_line = fe - se
        self._macd_sig_ema[symbol] = _ema(self._macd_sig_ema[symbol], macd_line, self.get_param("macd_signal_period"))
        sig_e = self._macd_sig_ema[symbol] or 0
        features["macd"] = macd_line
        features["macd_signal"] = sig_e
        features["macd_hist"] = macd_line - sig_e

        bb_p = self.get_param("bb_period")
        if len(closes) >= bb_p:
            w = closes[-bb_p:]
            mid = sum(w) / bb_p
            std = math.sqrt(sum((x - mid) ** 2 for x in w) / bb_p)
            mult = self.get_param("bb_std")
            features["bb_upper"] = mid + std * mult
            features["bb_middle"] = mid
            features["bb_lower"] = mid - std * mult
            features["bb_width"] = (features["bb_upper"] - features["bb_lower"]) / mid if mid > 0 else 0

        atr_p = self.get_param("atr_period")
        highs = list(self._high[symbol])
        lows = list(self._low[symbol])
        if len(highs) > atr_p:
            trs = []
            for i in range(-atr_p, 0):
                trs.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])))
            features["atr"] = sum(trs) / len(trs)
        else:
            features["atr"] = 0

        features["returns_1"] = (closes[-1] - closes[-2]) / closes[-2] if len(closes) > 1 and closes[-2] != 0 else 0
        features["returns_5"] = (closes[-1] - closes[-6]) / closes[-6] if len(closes) > 5 and closes[-6] != 0 else 0
        features["returns_10"] = (closes[-1] - closes[-11]) / closes[-11] if len(closes) > 10 and closes[-11] != 0 else 0

        if len(closes) >= 10:
            rets = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(-10, 0) if closes[i - 1] != 0]
            features["vol_10"] = np.std(rets) if rets else 0
        if len(closes) >= 20:
            rets = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(-20, 0) if closes[i - 1] != 0]
            features["vol_20"] = np.std(rets) if rets else 0

        vols = list(self._volume[symbol])
        if len(vols) >= 20:
            vol_ma = sum(vols[-20:]) / 20
            features["volume_ratio"] = vols[-1] / vol_ma if vol_ma > 0 else 1.0
        else:
            features["volume_ratio"] = 1.0

        taker = list(self._taker_buy[symbol])
        if taker and vols and vols[-1] > 0:
            features["taker_ratio"] = taker[-1] / vols[-1]
        else:
            features["taker_ratio"] = 0.5

        features["high_low_range"] = (bar["high"] - bar["low"]) / close if close > 0 else 0
        features["log_returns"] = math.log(closes[-1] / closes[-2]) if len(closes) > 1 and closes[-2] > 0 else 0

        features["obv"] = self._obv.get(symbol, 0)
        features["vwap"] = close

        if len(closes) >= 20:
            features["ma_20"] = sum(closes[-20:]) / 20
            ema_val = closes[-20]
            for c in closes[-19:]:
                ema_val = _ema(ema_val, c, 20)
            features["ema_20"] = ema_val
        features.setdefault("stoch_k", 50.0)
        features.setdefault("stoch_d", 50.0)
        features.setdefault("kc_upper", close)
        features.setdefault("kc_middle", close)
        features.setdefault("kc_lower", close)
        features.setdefault("vol_50", features.get("vol_20", 0))

        for p in [5, 10, 20]:
            if len(closes) > p:
                features[f"momentum_{p}"] = closes[-1] / closes[-p - 1] - 1
        if len(closes) >= 20:
            features["price_vs_ma20"] = closes[-1] / (sum(closes[-20:]) / 20) - 1
        if len(closes) >= 50:
            features["price_vs_ma50"] = closes[-1] / (sum(closes[-50:]) / 50) - 1

        if len(vols) >= 5:
            features["volume_ma5"] = sum(vols[-5:]) / 5
        if len(vols) >= 20:
            features["volume_ma20"] = sum(vols[-20:]) / 20
        features["avg_trade_size"] = bar.get("volume", 0) / max(bar.get("trades", 1), 1)

        ordered = []
        if self._feature_columns:
            for col in self._feature_columns:
                ordered.append(features.get(col, 0.0))
        else:
            ordered = list(features.values())

        arr = np.array(ordered, dtype=np.float64).reshape(1, -1)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    def _check_exits(self, symbol: str, bar: dict[str, Any], pos: Any) -> list[Signal]:
        """ATR stop-loss / take-profit — runs even during cooldown."""
        if pos is None:
            return []
        close = bar["close"]
        atr_val = calc_atr(self._high[symbol], self._low[symbol], self._close[symbol], self.get_param("atr_period"))
        if not atr_val or atr_val <= 0:
            return []
        regime_params = self._regime[symbol].get_params()
        sl_mult = regime_params.get("stop_loss_mult", self.get_param("stop_loss_atr_mult"))
        tp_mult = regime_params.get("take_profit_mult", self.get_param("take_profit_atr_mult"))
        signals: list[Signal] = []
        if pos.side.value == "buy":
            if close < pos.avg_price - atr_val * sl_mult:
                sig = Signal(strategy_id=self.strategy_id, symbol=symbol, signal_type=SignalType.LONG_EXIT, strength=0.9, price=close, reason="ML止损")
                signals.append(sig)
                self.record_signal(sig)
            elif close > pos.avg_price + atr_val * tp_mult:
                sig = Signal(strategy_id=self.strategy_id, symbol=symbol, signal_type=SignalType.LONG_EXIT, strength=0.7, price=close, reason="ML止盈")
                signals.append(sig)
                self.record_signal(sig)
        elif pos.side.value == "sell":
            if close > pos.avg_price + atr_val * sl_mult:
                sig = Signal(strategy_id=self.strategy_id, symbol=symbol, signal_type=SignalType.SHORT_EXIT, strength=0.9, price=close, reason="ML止损")
                signals.append(sig)
                self.record_signal(sig)
            elif close < pos.avg_price - atr_val * tp_mult:
                sig = Signal(strategy_id=self.strategy_id, symbol=symbol, signal_type=SignalType.SHORT_EXIT, strength=0.7, price=close, reason="ML止盈")
                signals.append(sig)
                self.record_signal(sig)
        return signals

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._ensure_buffers(symbol)

        close = bar["close"]
        self._close[symbol].append(close)
        self._high[symbol].append(bar["high"])
        self._low[symbol].append(bar["low"])
        vol = bar.get("volume", 0)
        self._volume[symbol].append(vol)
        self._taker_buy[symbol].append(bar.get("taker_buy_volume", vol * 0.5))

        prev = list(self._close[symbol])[-2] if len(self._close[symbol]) > 1 else close
        self._obv[symbol] = self._obv.get(symbol, 0) + (vol if close >= prev else -vol)

        self._regime[symbol].update(bar["high"], bar["low"], close)

        # Exit logic must run even during cooldown to protect open positions
        pos = self.get_position(symbol)
        exit_signals = self._check_exits(symbol, bar, pos)
        if exit_signals:
            return exit_signals

        if self._cooldown[symbol] > 0:
            self._cooldown[symbol] -= 1
            return []

        if not self._model_loaded:
            return []

        X = self._compute_features(symbol, bar)
        if X is None:
            return []

        try:
            result = self._model.predict(X)
        except Exception as e:
            logger.warning("Prediction failed: %s", e)
            return []

        signals: list[Signal] = []
        threshold = self.get_param("prediction_threshold")

        pred = result.predictions[0] if result.predictions else 0
        proba = result.probabilities[0] if result.probabilities else [0.5, 0.5]
        confidence = max(proba) if proba else 0.5

        if confidence < threshold:
            return []

        regime = self._regime[symbol].current_regime
        regime_params = self._regime[symbol].get_params()
        scale = self.get_param("signal_strength_scale")

        if self.get_param("enable_regime_filter"):
            pos_scale = regime_params.get("position_scale", 1.0)
            scale *= pos_scale

        if pred == 1 and confidence >= threshold and pos is None:
            strength = min(confidence * scale, 1.0)
            sig = Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_type=SignalType.LONG_ENTRY,
                strength=round(strength, 4),
                price=close,
                reason=f"ML做多(conf={confidence:.3f},regime={regime.value})",
                metadata={
                    "confidence": confidence,
                    "prediction": pred,
                    "regime": regime.value,
                    "signal_source": "ml",
                },
            )
            signals.append(sig)
            self.record_signal(sig)

        elif pred == 0 and confidence >= threshold and pos is None:
            strength = min(confidence * scale, 1.0)
            sig = Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_type=SignalType.SHORT_ENTRY,
                strength=round(strength, 4),
                price=close,
                reason=f"ML做空(conf={confidence:.3f},regime={regime.value})",
                metadata={
                    "confidence": confidence,
                    "prediction": pred,
                    "regime": regime.value,
                    "signal_source": "ml",
                },
            )
            signals.append(sig)
            self.record_signal(sig)

        if signals:
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
