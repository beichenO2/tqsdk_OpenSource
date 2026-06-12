"""DL 策略 — 使用 LSTM 或 Transformer 模型生成交易信号。"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS: dict[str, Any] = {
    "model_type": "lstm",  # "lstm" or "transformer"
    "sequence_length": 30,
    "long_prob_threshold": 0.55,
    "short_prob_threshold": 0.55,
    "exit_prob_threshold": 0.50,
    "volatility_window": 20,
    "volume_ma_period": 20,
    "max_hold_bars": 100,
}


@auto_register("dl_timeseries")
class DLTimeseriesStrategy(BaseStrategy):
    """Uses a pre-trained LSTM or Transformer model for direction prediction.

    Pass a trained model instance via ``config.params["dl_model"]`` or the
    constructor ``model`` kwarg.
    """

    def __init__(
        self,
        config: StrategyConfig,
        model: Any | None = None,
    ) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config = config.model_copy(update={"params": merged})
        super().__init__(config)

        self._dl_model = model or config.params.get("dl_model")
        self._seq_len = int(merged["sequence_length"])

        self._feature_buffers: dict[str, deque[list[float]]] = {}
        self._close_history: dict[str, deque[float]] = {}
        self._volume_history: dict[str, deque[float]] = {}
        self._bars_in_pos: dict[str, int] = {}

    def _ensure_buffers(self, symbol: str) -> None:
        buf_len = self._seq_len + 10
        if symbol not in self._feature_buffers:
            self._feature_buffers[symbol] = deque(maxlen=buf_len)
            self._close_history[symbol] = deque(maxlen=buf_len)
            self._volume_history[symbol] = deque(maxlen=buf_len)

    def _compute_features(self, symbol: str, bar: dict[str, Any]) -> list[float]:
        """Build a feature vector from current bar + recent history."""
        o = float(bar.get("open", bar["close"]))
        h = float(bar["high"])
        l = float(bar["low"])
        c = float(bar["close"])
        v = float(bar.get("volume", 0.0))

        closes = self._close_history[symbol]
        volumes = self._volume_history[symbol]

        ret_1 = (c - closes[-1]) / closes[-1] if len(closes) >= 1 and closes[-1] != 0 else 0.0
        ret_5 = (c - closes[-5]) / closes[-5] if len(closes) >= 5 and closes[-5] != 0 else 0.0

        vw = self.get_param("volatility_window", 20)
        if len(closes) >= vw + 1:
            arr = np.array(list(closes)[-vw - 1:])
            rets = np.diff(arr) / np.where(arr[:-1] != 0, arr[:-1], 1.0)
            vol = float(np.std(rets))
        else:
            vol = 0.0

        vma_p = self.get_param("volume_ma_period", 20)
        vlist = list(volumes)
        vma = sum(vlist[-vma_p:]) / vma_p if len(vlist) >= vma_p else v
        vol_ratio = v / vma if vma > 0 else 1.0

        hl_range = (h - l) / c if c > 0 else 0.0
        body = (c - o) / c if c > 0 else 0.0

        return [o, h, l, c, v, ret_1, ret_5, vol, vol_ratio, hl_range, body]

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._ensure_buffers(symbol)

        features = self._compute_features(symbol, bar)

        c = float(bar["close"])
        v = float(bar.get("volume", 0.0))
        self._close_history[symbol].append(c)
        self._volume_history[symbol].append(v)
        self._feature_buffers[symbol].append(features)

        if len(self._feature_buffers[symbol]) < self._seq_len:
            return []

        if self._dl_model is None or not getattr(self._dl_model, "is_trained", False):
            return []

        seq = np.array(list(self._feature_buffers[symbol])[-self._seq_len:], dtype=np.float32)
        X = seq[np.newaxis, :, :]  # (1, seq_len, features)

        try:
            result = self._dl_model.predict(X)
        except Exception as exc:
            logger.debug("DL predict error: %s", exc)
            return []

        proba = result.probabilities
        if not proba or len(proba[0]) < 2:
            return []

        p_down, p_up = float(proba[0][0]), float(proba[0][1])
        long_thr = float(self.get_param("long_prob_threshold", 0.55))
        short_thr = float(self.get_param("short_prob_threshold", 0.55))

        signals: list[Signal] = []
        price = c
        exit_thr = float(self.get_param("exit_prob_threshold", 0.50))
        max_hold = int(self.get_param("max_hold_bars", 100))

        pos = self.get_position(symbol)

        if pos is not None:
            self._bars_in_pos[symbol] = self._bars_in_pos.get(symbol, 0) + 1
            is_long = pos.side.value == "buy"

            should_exit = False
            reason = ""
            if self._bars_in_pos.get(symbol, 0) >= max_hold:
                should_exit = True
                reason = f"DL max hold ({max_hold} bars)"
            elif is_long and p_down >= exit_thr:
                should_exit = True
                reason = f"DL exit long (p_down={p_down:.3f})"
            elif not is_long and p_up >= exit_thr:
                should_exit = True
                reason = f"DL exit short (p_up={p_up:.3f})"

            if should_exit:
                exit_type = SignalType.LONG_EXIT if is_long else SignalType.SHORT_EXIT
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_type, strength=0.8, price=price,
                    reason=reason, metadata={"probabilities": proba[0]},
                )
                signals.append(sig)
                self.record_signal(sig)
                self._bars_in_pos[symbol] = 0
        else:
            self._bars_in_pos[symbol] = 0
            if p_up >= long_thr and p_up >= p_down:
                sig = Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=min(p_up, 1.0),
                    price=price,
                    reason=f"DL long (p_up={p_up:.3f})",
                    metadata={"probabilities": proba[0]},
                )
                signals.append(sig)
                self.record_signal(sig)
            elif p_down >= short_thr:
                sig = Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=min(p_down, 1.0),
                    price=price,
                    reason=f"DL short (p_down={p_down:.3f})",
                    metadata={"probabilities": proba[0]},
                )
                signals.append(sig)
                self.record_signal(sig)

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        for sym in self.config.symbols:
            bar = market_data.get(sym)
            if bar:
                out.extend(await self.on_bar(sym, bar))
        return out
