"""跨期套利 — 近远月价差相对滚动均值的偏离与均值回归。"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "lookback": 60,
    "entry_stdev_mult": 2.0,
    "exit_stdev_mult": 0.5,
    "min_samples": 20,
}


def _mean_stdev(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n < 2:
        return 0.0, 0.0
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return mean, math.sqrt(max(var, 0.0))


@auto_register("spread_arb")
class SpreadArbitrage(BaseStrategy):
    """监控近月与远月收盘价差；价差超出均值±N倍标准差入场，向均值回归时离场。

    需在 ``StrategyConfig.symbols`` 中至少配置两个合约（近、远），信号主体在近月合约上，
    远月对冲方向写入 ``metadata``。
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

    def _near_far(self) -> tuple[str, str] | None:
        syms = self.config.symbols
        if len(syms) < 2:
            return None
        return syms[0], syms[1]

    def _ensure_spread_deque(self, key: str) -> None:
        lb = int(self.get_param("lookback"))
        if key not in self._spread_hist:
            self._spread_hist[key] = deque(maxlen=max(lb + 5, 30))

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

        nf = self._near_far()
        if nf is None:
            return self._hold(symbol, float(close), "need two symbols in config")

        near_sym, far_sym = nf
        self._last_close[symbol] = float(close)

        near_c = self._last_close.get(near_sym)
        far_c = self._last_close.get(far_sym)
        if near_c is None or far_c is None:
            return self._hold(symbol, float(close), "waiting for both legs")

        spread = near_c - far_c
        key = self._pair_key()
        assert key is not None
        self._ensure_spread_deque(key)
        self._spread_hist[key].append(spread)

        hist = list(self._spread_hist[key])
        min_s = int(self.get_param("min_samples"))
        lb = int(self.get_param("lookback"))
        window = hist[-lb:] if len(hist) >= lb else hist
        if len(window) < min_s:
            return self._hold(symbol, float(close), "warming up spread window")

        mean, stdev = _mean_stdev(window)
        if stdev <= 0:
            return self._hold(symbol, float(close), "zero stdev")

        z = (spread - mean) / stdev
        entry_n = float(self.get_param("entry_stdev_mult") or 2.0)
        exit_n = float(self.get_param("exit_stdev_mult") or 0.5)

        pos = self.get_position(near_sym)
        fc = float(close)

        if symbol != near_sym:
            return self._hold(symbol, fc, "signal leg is near month only")

        if pos is None:
            if z > entry_n:
                sig = Signal(
                    strategy_id=self.strategy_id,
                    symbol=near_sym,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=min((z - entry_n) / max(entry_n, 0.1), 1.0),
                    price=fc,
                    reason=f"价差过高 z={z:.3f} spread={spread:.4f} mean={mean:.4f}",
                    metadata={
                        "spread": spread,
                        "z": z,
                        "mean": mean,
                        "stdev": stdev,
                        "hedge_symbol": far_sym,
                        "hedge_side": "buy",
                    },
                )
                self.record_signal(sig)
                return sig
            if z < -entry_n:
                sig = Signal(
                    strategy_id=self.strategy_id,
                    symbol=near_sym,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=min((-z - entry_n) / max(entry_n, 0.1), 1.0),
                    price=fc,
                    reason=f"价差过低 z={z:.3f} spread={spread:.4f} mean={mean:.4f}",
                    metadata={
                        "spread": spread,
                        "z": z,
                        "mean": mean,
                        "stdev": stdev,
                        "hedge_symbol": far_sym,
                        "hedge_side": "sell",
                    },
                )
                self.record_signal(sig)
                return sig
            return self._hold(near_sym, fc, "no entry")

        if pos.side.value == "buy" and z >= -exit_n:
            sig = Signal(
                strategy_id=self.strategy_id,
                symbol=near_sym,
                signal_type=SignalType.LONG_EXIT,
                strength=min(0.5 + abs(z) / max(exit_n, 0.05) * 0.1, 1.0),
                price=fc,
                reason=f"做多价差均值回归 z={z:.3f}",
                metadata={"spread": spread, "z": z, "mean": mean},
            )
            self.record_signal(sig)
            return sig

        if pos.side.value == "sell" and z <= exit_n:
            sig = Signal(
                strategy_id=self.strategy_id,
                symbol=near_sym,
                signal_type=SignalType.SHORT_EXIT,
                strength=min(0.5 + abs(z) / max(exit_n, 0.05) * 0.1, 1.0),
                price=fc,
                reason=f"做空价差均值回归 z={z:.3f}",
                metadata={"spread": spread, "z": z, "mean": mean},
            )
            self.record_signal(sig)
            return sig

        return self._hold(near_sym, fc, "in position")

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        sig = self.generate_signal(symbol, bar)
        if sig.signal_type == SignalType.HOLD:
            return []
        return [sig]

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        nf = self._near_far()
        if nf is None:
            return []
        near_sym, far_sym = nf

        for sym in self.config.symbols:
            b = market_data.get(sym)
            if b and b.get("close") is not None:
                self._last_close[sym] = float(b["close"])

        near_bar = market_data.get(near_sym)
        if near_bar is None:
            return []
        sig = self.generate_signal(near_sym, near_bar)
        if sig.signal_type == SignalType.HOLD:
            return []
        return [sig]
