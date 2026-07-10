"""因子驱动策略 — 多因子 z-score 加权合成信号。"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..feature_mixin import FeatureMixin
from ..registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "factors": {},
    "entry_z": 1.0,
    "exit_z": 0.3,
    "zscore_window": 60,
    "allow_short": True,
}


@auto_register("factor_strategy")
class FactorStrategy(FeatureMixin, BaseStrategy):
    """纯配置驱动的多因子 z-score 策略。"""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        factors = merged.get("factors") or {}
        features = list(config.features or list(factors.keys()))
        config = config.model_copy(update={"params": merged, "features": features})
        super().__init__(config)
        self._init_features()

        self._factor_history: dict[str, dict[str, deque[float | None]]] = {}
        self._pos_side: dict[str, str | None] = {}

        self._zscore_window = int(self.get_param("zscore_window"))
        self._entry_z = float(self.get_param("entry_z"))
        self._exit_z = float(self.get_param("exit_z"))
        self._allow_short = bool(self.get_param("allow_short"))
        self._factor_weights: dict[str, float] = {
            str(k): float(v) for k, v in (factors or {}).items()
        }

    def _ensure_history(self, symbol: str) -> dict[str, deque[float | None]]:
        if symbol not in self._factor_history:
            self._factor_history[symbol] = {
                name: deque(maxlen=self._zscore_window * 2)
                for name in self._factor_weights
            }
        return self._factor_history[symbol]

    @staticmethod
    def _zscore(window: list[float], value: float) -> float | None:
        if len(window) < 2:
            return None
        mean = sum(window) / len(window)
        var = sum((x - mean) ** 2 for x in window) / len(window)
        std = math.sqrt(var)
        if std == 0:
            return 0.0
        return (value - mean) / std

    def _composite_score(self, symbol: str, values: dict[str, float | None]) -> float | None:
        histories = self._ensure_history(symbol)
        weighted_sum = 0.0
        total_w = 0.0

        for name, weight in self._factor_weights.items():
            if weight == 0:
                continue
            raw = values.get(name)
            if raw is None:
                return None
            hist = histories[name]
            hist.append(raw)
            if len(hist) < self._zscore_window:
                return None
            window = [float(x) for x in list(hist)[-self._zscore_window :]]
            z = self._zscore(window, float(raw))
            if z is None:
                return None
            signed = z if weight >= 0 else -z
            w = abs(weight)
            weighted_sum += signed * w
            total_w += w

        if total_w == 0:
            return None
        return weighted_sum / total_w

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self.record_bar(symbol, bar)
        close = float(bar["close"])

        values = self.factor_values(symbol)
        score = self._composite_score(symbol, values)
        if score is None:
            return []

        signals: list[Signal] = []
        strength = min(abs(score), 1.0)
        side = self._pos_side.get(symbol)

        if side is None:
            if score > self._entry_z:
                signals.append(Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=strength,
                    price=close,
                    reason=f"factor_score={score:.3f}",
                ))
                self._pos_side[symbol] = "long"
            elif score < -self._entry_z and self._allow_short:
                signals.append(Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=strength,
                    price=close,
                    reason=f"factor_score={score:.3f}",
                ))
                self._pos_side[symbol] = "short"
        elif side == "long":
            if abs(score) < self._exit_z:
                signals.append(Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=SignalType.LONG_EXIT,
                    strength=strength,
                    price=close,
                    reason=f"factor_exit score={score:.3f}",
                ))
                self._pos_side[symbol] = None
        elif side == "short":
            if abs(score) < self._exit_z:
                signals.append(Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=SignalType.SHORT_EXIT,
                    strength=strength,
                    price=close,
                    reason=f"factor_exit score={score:.3f}",
                ))
                self._pos_side[symbol] = None

        for s in signals:
            self.record_signal(s)
        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        all_sigs: list[Signal] = []
        for symbol in self.config.symbols:
            bar = market_data.get(symbol)
            if bar:
                all_sigs.extend(await self.on_bar(symbol, bar))
        return all_sigs
