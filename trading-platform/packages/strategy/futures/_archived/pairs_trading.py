"""配对交易 — 价差 Z 分数均值回归 + Half-Kelly 仓位缩放。"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..registry import auto_register

DEFAULT_PARAMS = {
    "lookback": 80,
    "entry_z": 2.0,
    "exit_z": 0.4,
    "min_samples": 25,
    "beta": 1.0,
    "base_position": 1.0,
    "half_kelly_mult": 0.5,
    "max_suggested_qty": 100.0,
}


def _mean_stdev(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n < 2:
        return 0.0, 0.0
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return mean, math.sqrt(max(var, 0.0))


@auto_register("pairs_trading")
class PairsTradingStrategy(BaseStrategy):
    """两品种价差 ``leg_a - beta * leg_b`` 的 Z 分数交易；信号挂在 leg_a，对冲信息写入 metadata。

    Half-Kelly 简化：在价差高斯近似下用 ``0.5 * half_kelly_mult * (|z|/entry_z) * base_position`` 缩放建议手数，
    并受 ``max_suggested_qty`` 限制。
    """

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._last_close: dict[str, float] = {}
        self._spread_hist: dict[str, deque[float]] = {}

    def _pair_key(self) -> str | None:
        syms = self.config.symbols
        if len(syms) < 2:
            return None
        return f"{syms[0]}|{syms[1]}"

    def _legs(self) -> tuple[str, str] | None:
        syms = self.config.symbols
        if len(syms) < 2:
            return None
        return syms[0], syms[1]

    def _ensure_deque(self, key: str) -> None:
        lb = int(self.get_param("lookback"))
        if key not in self._spread_hist:
            self._spread_hist[key] = deque(maxlen=max(lb + 10, 40))

    def _half_kelly_qty(self, z: float, entry_z: float, stdev: float) -> float:
        """Half-Kelly style scale from z-score distance (bounded)."""
        base = float(self.get_param("base_position") or 1.0)
        hk = float(self.get_param("half_kelly_mult") or 0.5)
        cap = float(self.get_param("max_suggested_qty") or 100.0)
        if entry_z <= 0:
            return min(base, cap)
        # variance proxy from spread stdev; edge proxy ~ |z| * stdev relative to entry
        intensity = min(abs(z) / entry_z, 2.5)
        # classic half-Kelly halves full Kelly; we fold mult into one scalar
        raw = 0.5 * hk * intensity * base
        if stdev > 0:
            raw *= min(1.0, 1.0 / max(stdev, 1e-9))
        return float(min(max(raw, 0.0), cap))

    def _hold(self, symbol: str, price: float | None, reason: str) -> Signal:
        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            signal_type=SignalType.HOLD,
            strength=0.0,
            price=price,
            reason=reason,
        )

    def generate_signal(self, symbol: str, bar: dict[str, Any]) -> Signal:
        close = bar.get("close")
        if close is None:
            return self._hold(symbol, None, "incomplete bar")

        legs = self._legs()
        if legs is None:
            return self._hold(symbol, float(close), "need two symbols in config")

        leg_a, leg_b = legs
        self._last_close[symbol] = float(close)

        ca = self._last_close.get(leg_a)
        cb = self._last_close.get(leg_b)
        if ca is None or cb is None:
            return self._hold(symbol, float(close), "waiting for both legs")

        beta = float(self.get_param("beta") or 1.0)
        spread = ca - beta * cb
        key = self._pair_key()
        assert key is not None
        self._ensure_deque(key)
        self._spread_hist[key].append(spread)

        hist = list(self._spread_hist[key])
        min_s = int(self.get_param("min_samples"))
        lb = int(self.get_param("lookback"))
        window = hist[-lb:] if len(hist) >= lb else hist
        if len(window) < min_s:
            return self._hold(symbol, float(close), "warming up")

        mean, stdev = _mean_stdev(window)
        if stdev <= 0:
            return self._hold(symbol, float(close), "zero stdev")

        z = (spread - mean) / stdev
        z_in = float(self.get_param("entry_z") or 2.0)
        z_out = float(self.get_param("exit_z") or 0.4)

        if symbol != leg_a:
            return self._hold(symbol, float(close), "signal leg is leg_a only")

        pos = self.get_position(leg_a)
        fc = float(close)
        sq = self._half_kelly_qty(z, z_in, stdev)

        if pos is None:
            if z > z_in:
                sig = Signal(
                    strategy_id=self.strategy_id,
                    symbol=leg_a,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=min((z - z_in) / max(z_in, 0.1), 1.0),
                    price=fc,
                    suggested_qty=sq,
                    reason=f"配对: 价差偏高 z={z:.3f} spread={spread:.6f}",
                    metadata={
                        "spread": spread,
                        "z": z,
                        "mean": mean,
                        "stdev": stdev,
                        "hedge_symbol": leg_b,
                        "hedge_side": "buy",
                        "beta": beta,
                    },
                )
                self.record_signal(sig)
                return sig
            if z < -z_in:
                sig = Signal(
                    strategy_id=self.strategy_id,
                    symbol=leg_a,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=min((-z - z_in) / max(z_in, 0.1), 1.0),
                    price=fc,
                    suggested_qty=sq,
                    reason=f"配对: 价差偏低 z={z:.3f} spread={spread:.6f}",
                    metadata={
                        "spread": spread,
                        "z": z,
                        "mean": mean,
                        "stdev": stdev,
                        "hedge_symbol": leg_b,
                        "hedge_side": "sell",
                        "beta": beta,
                    },
                )
                self.record_signal(sig)
                return sig
            return self._hold(leg_a, fc, "no entry")

        if pos.side.value == "buy" and z >= -z_out:
            sig = Signal(
                strategy_id=self.strategy_id,
                symbol=leg_a,
                signal_type=SignalType.LONG_EXIT,
                strength=min(0.55 + abs(z) / max(z_out, 0.05) * 0.15, 1.0),
                price=fc,
                reason=f"配对: 多头价差均值回归 z={z:.3f}",
                metadata={"spread": spread, "z": z, "mean": mean},
            )
            self.record_signal(sig)
            return sig

        if pos.side.value == "sell" and z <= z_out:
            sig = Signal(
                strategy_id=self.strategy_id,
                symbol=leg_a,
                signal_type=SignalType.SHORT_EXIT,
                strength=min(0.55 + abs(z) / max(z_out, 0.05) * 0.15, 1.0),
                price=fc,
                reason=f"配对: 空头价差均值回归 z={z:.3f}",
                metadata={"spread": spread, "z": z, "mean": mean},
            )
            self.record_signal(sig)
            return sig

        return self._hold(leg_a, fc, "in position")

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        sig = self.generate_signal(symbol, bar)
        if sig.signal_type == SignalType.HOLD:
            return []
        return [sig]

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        legs = self._legs()
        if legs is None:
            return []
        leg_a, leg_b = legs

        for sym in self.config.symbols:
            b = market_data.get(sym)
            if b and b.get("close") is not None:
                self._last_close[sym] = float(b["close"])

        bar_a = market_data.get(leg_a)
        if bar_a is None:
            return []
        sig = self.generate_signal(leg_a, bar_a)
        if sig.signal_type == SignalType.HOLD:
            return []
        return [sig]
