"""Signals from perpetual funding rate extremes vs rolling distribution."""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "rolling_window": 72,
    "z_score_mult": 1.5,
    "static_threshold": 0.0003,
    "min_std": 1e-6,
    "exit_band_mult": 0.35,
}


@auto_register("funding_rate_arb")
class FundingRateArbitrage(BaseStrategy):
    """Fade crowded funding: high positive funding → short; deeply negative → long.

    Threshold = max(static_threshold, mean + z_score_mult * std) computed on a rolling
    window of past funding observations. Exits when funding re-enters a tighter band
    around the rolling mean.
    """

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)
        self._history: dict[str, deque[float]] = {}
        win = int(self.get_param("rolling_window"))
        self._win = max(8, win)

    def _buf(self, symbol: str) -> deque[float]:
        if symbol not in self._history:
            self._history[symbol] = deque(maxlen=self._win + 5)
        return self._history[symbol]

    def _stats(self, symbol: str) -> tuple[float, float] | None:
        buf = list(self._buf(symbol))
        if len(buf) < max(5, self._win // 3):
            return None
        window = buf[-self._win :] if len(buf) >= self._win else buf
        n = len(window)
        if n < 5:
            return None
        mean = sum(window) / n
        var = sum((x - mean) ** 2 for x in window) / n
        std = var**0.5
        min_std = float(self.get_param("min_std"))
        std = max(std, min_std)
        return mean, std

    def _bounds(self, mean: float, std: float) -> tuple[float, float]:
        z = float(self.get_param("z_score_mult"))
        static = float(self.get_param("static_threshold"))
        upper = max(static, mean + z * std)
        lower = min(-static, mean - z * std)
        return upper, lower

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        if "funding_rate" not in bar:
            return []

        fr = float(bar["funding_rate"])
        close = float(bar["close"])
        self._buf(symbol).append(fr)

        st = self._stats(symbol)
        if st is None:
            return []

        mean, std = st
        upper, lower = self._bounds(mean, std)
        exit_scale = float(self.get_param("exit_band_mult"))
        exit_band = max(std * exit_scale, float(self.get_param("min_std")))

        pos = self.get_position(symbol)
        signals: list[Signal] = []

        if pos is not None:
            if abs(fr - mean) <= exit_band:
                if pos.side.value == "buy":
                    sig = Signal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=SignalType.LONG_EXIT,
                        strength=0.75,
                        price=close,
                        reason=f"资金费率回归均值附近 (funding={fr:.6f}, mean={mean:.6f})",
                        metadata={"funding_rate": fr, "mean": mean, "std": std},
                    )
                else:
                    sig = Signal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=SignalType.SHORT_EXIT,
                        strength=0.75,
                        price=close,
                        reason=f"资金费率回归均值附近 (funding={fr:.6f}, mean={mean:.6f})",
                        metadata={"funding_rate": fr, "mean": mean, "std": std},
                    )
                signals.append(sig)
                self.record_signal(sig)
            return signals

        if fr > upper:
            strength = min(1.0, (fr - upper) / (upper + 1e-9) * 0.5 + 0.35)
            sig = Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_type=SignalType.SHORT_ENTRY,
                strength=round(strength, 4),
                price=close,
                reason=f"资金费率偏高(拥挤多): {fr:.6f} > 动态上阈 {upper:.6f}",
                metadata={
                    "funding_rate": fr,
                    "rolling_mean": mean,
                    "rolling_std": std,
                    "threshold": upper,
                },
            )
            signals.append(sig)
            self.record_signal(sig)
        elif fr < lower:
            strength = min(1.0, (lower - fr) / (abs(lower) + 1e-9) * 0.5 + 0.35)
            sig = Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_type=SignalType.LONG_ENTRY,
                strength=round(strength, 4),
                price=close,
                reason=f"资金费率偏低(拥挤空): {fr:.6f} < 动态下阈 {lower:.6f}",
                metadata={
                    "funding_rate": fr,
                    "rolling_mean": mean,
                    "rolling_std": std,
                    "threshold": lower,
                },
            )
            signals.append(sig)
            self.record_signal(sig)

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        for symbol in self.config.symbols:
            bar = market_data.get(symbol)
            if bar:
                out.extend(await self.on_bar(symbol, bar))
        return out
