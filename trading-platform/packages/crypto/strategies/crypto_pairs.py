"""Crypto Pairs Trading — Spread Mean Reversion.

Based on Frontiers 2026 research on cointegrated crypto pairs.
Uses rolling OLS hedge ratio and z-score of spread for entry/exit.

Core logic:
1. Compute rolling hedge ratio between two assets (e.g., ETH vs BTC)
2. Calculate spread = asset_A - hedge_ratio * asset_B
3. Normalize spread to z-score
4. Enter when z-score > threshold (short spread) or < -threshold (long spread)
5. Exit when z-score returns to zero (mean reversion)

Market-neutral strategy — profits from relative value changes, not direction.
Uses bar.extra for secondary asset price (fed by backtest runner).
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any

from strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig
from strategy.registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "hedge_ratio_window": 60,
    "zscore_window": 40,
    "z_entry": 2.0,
    "z_exit": 0.5,
    "z_stop": 4.0,
    "max_hold_bars": 72,
    "cooldown_bars": 5,
    "secondary_symbol": "ETHUSDT",
    "secondary_price_key": "secondary_close",
}


def _rolling_ols_hedge(primary: list[float], secondary: list[float], window: int) -> float | None:
    """Simple OLS hedge ratio: primary = alpha + beta * secondary."""
    if len(primary) < window or len(secondary) < window:
        return None

    x = secondary[-window:]
    y = primary[-window:]

    mean_x = sum(x) / window
    mean_y = sum(y) / window

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / window
    var_x = sum((xi - mean_x) ** 2 for xi in x) / window

    if var_x < 1e-10:
        return None

    return cov / var_x


@auto_register("crypto_pairs")
class CryptoPairsStrategy(BaseStrategy):
    """Spread mean-reversion between two crypto assets."""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._primary: dict[str, deque[float]] = {}
        self._secondary: dict[str, deque[float]] = {}
        self._spreads: dict[str, deque[float]] = {}
        self._hold: dict[str, int] = {}
        self._cd: dict[str, int] = {}
        self._entry_z: dict[str, float] = {}
        self._buf = 200

    def _init(self, s: str) -> None:
        if s not in self._primary:
            self._primary[s] = deque(maxlen=self._buf)
            self._secondary[s] = deque(maxlen=self._buf)
            self._spreads[s] = deque(maxlen=self._buf)

    def _compute_zscore(self, s: str) -> float | None:
        spreads = list(self._spreads[s])
        w = self.get_param("zscore_window")
        if len(spreads) < w:
            return None

        window = spreads[-w:]
        mean_s = sum(window) / len(window)
        var_s = sum((x - mean_s) ** 2 for x in window) / len(window)
        std_s = math.sqrt(var_s) if var_s > 0 else 0.001

        return (spreads[-1] - mean_s) / std_s

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._init(symbol)
        c = bar["close"]
        sec_key = self.get_param("secondary_price_key")
        sec_c = bar.get(sec_key, bar.get("extra", {}).get(sec_key, 0.0))

        if sec_c <= 0:
            return []

        self._primary[symbol].append(c)
        self._secondary[symbol].append(sec_c)

        hedge = _rolling_ols_hedge(
            list(self._primary[symbol]),
            list(self._secondary[symbol]),
            self.get_param("hedge_ratio_window"),
        )
        if hedge is None:
            return []

        spread = c - hedge * sec_c
        self._spreads[symbol].append(spread)

        z = self._compute_zscore(symbol)
        if z is None:
            return []

        signals: list[Signal] = []
        pos = self.get_position(symbol)
        self._cd[symbol] = max(self._cd.get(symbol, 0) - 1, 0)

        z_entry = self.get_param("z_entry")
        z_exit = self.get_param("z_exit")
        z_stop = self.get_param("z_stop")

        if pos is None:
            if self._cd.get(symbol, 0) > 0:
                return signals

            if z > z_entry:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY, strength=min(abs(z) / 3, 1.0), price=c,
                    reason=f"PAIRS short spread z={z:.2f} hedge={hedge:.4f}",
                    metadata={"z_score": z, "hedge_ratio": hedge, "spread": spread},
                ))
                self._hold[symbol] = 0
                self._entry_z[symbol] = z

            elif z < -z_entry:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY, strength=min(abs(z) / 3, 1.0), price=c,
                    reason=f"PAIRS long spread z={z:.2f} hedge={hedge:.4f}",
                    metadata={"z_score": z, "hedge_ratio": hedge, "spread": spread},
                ))
                self._hold[symbol] = 0
                self._entry_z[symbol] = z

        else:
            self._hold[symbol] = self._hold.get(symbol, 0) + 1
            ex = False
            reason = ""

            if self._hold[symbol] >= self.get_param("max_hold_bars"):
                ex, reason = True, "timeout"
            elif pos.side.value == "buy" and z >= -z_exit:
                ex, reason = True, f"spread reverted z={z:.2f}"
            elif pos.side.value == "sell" and z <= z_exit:
                ex, reason = True, f"spread reverted z={z:.2f}"
            elif abs(z) > z_stop:
                ex, reason = True, f"z-stop z={z:.2f}"

            if ex:
                et = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=et, strength=0.9, price=c,
                    reason=f"PAIRS: {reason}",
                ))
                self._cd[symbol] = self.get_param("cooldown_bars")

        return signals

    async def generate_signals(self, md: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        for s in self.config.symbols:
            if s in md:
                out.extend(await self.on_bar(s, md[s]))
        return out
