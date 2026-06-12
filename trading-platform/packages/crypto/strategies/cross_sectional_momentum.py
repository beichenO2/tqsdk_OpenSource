"""Time-Series Momentum with Volatility Scaling (2024-2025 style).

When only a single asset is available, uses time-series momentum instead of
cross-sectional ranking. Blends short and long lookback returns, scales by
realized volatility, and trades directionally.

When multiple assets are available via generate_signals(), performs
cross-sectional ranking (long top, short bottom).

Key innovations:
- Volatility-scaled returns for signal strength
- Momentum crash detection: skip entry when vol-of-vol spikes
- Dual lookback blending (short + long momentum)
- ATR-based risk management
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any

from strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig
from strategy.indicators import calc_atr, check_atr_exit
from strategy.registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "short_lookback": 10,
    "long_lookback": 30,
    "short_weight": 0.3,
    "long_weight": 0.7,
    "vol_lookback": 20,
    "entry_threshold": 1.2,
    "exit_threshold": 0.3,
    "rebalance_interval": 6,
    "atr_period": 14,
    "stop_loss_atr_mult": 1.8,
    "take_profit_atr_mult": 5.0,
    "max_hold_bars": 20,
    "vol_of_vol_threshold": 1.8,
    "min_bars_between_trades": 6,
}


@auto_register("time_series_momentum")
class TimeSeriesMomentumStrategy(BaseStrategy):
    """Volatility-scaled time-series momentum."""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._close: dict[str, deque[float]] = {}
        self._high: dict[str, deque[float]] = {}
        self._low: dict[str, deque[float]] = {}
        self._returns: dict[str, deque[float]] = {}
        self._vol_history: dict[str, deque[float]] = {}
        self._bar_count: dict[str, int] = {}
        self._hold_bars: dict[str, int] = {}
        self._bars_since_exit: dict[str, int] = {}

    def _ensure_buffers(self, symbol: str) -> None:
        buf = max(self.get_param("long_lookback"), self.get_param("vol_lookback")) + 30
        if symbol not in self._close:
            self._close[symbol] = deque(maxlen=buf)
            self._high[symbol] = deque(maxlen=buf)
            self._low[symbol] = deque(maxlen=buf)
            self._returns[symbol] = deque(maxlen=buf)
            self._vol_history[symbol] = deque(maxlen=buf)
            self._bar_count[symbol] = 0

    def _vol_adjusted_momentum(self, symbol: str) -> float | None:
        returns = list(self._returns[symbol])
        short_lb = self.get_param("short_lookback")
        long_lb = self.get_param("long_lookback")
        vol_lb = self.get_param("vol_lookback")

        if len(returns) < max(long_lb, vol_lb):
            return None

        vol = math.sqrt(sum(r**2 for r in returns[-vol_lb:]) / vol_lb)
        if vol <= 1e-10:
            return None

        short_mom = sum(returns[-short_lb:]) / vol
        long_mom = sum(returns[-long_lb:]) / vol

        sw = self.get_param("short_weight")
        lw = self.get_param("long_weight")
        return sw * short_mom + lw * long_mom

    def _vol_of_vol_spike(self, symbol: str) -> bool:
        """Detect if vol-of-vol is spiking (momentum crash risk)."""
        vols = list(self._vol_history[symbol])
        if len(vols) < 20:
            return False
        mean_v = sum(vols) / len(vols)
        std_v = math.sqrt(sum((v - mean_v)**2 for v in vols) / len(vols))
        if std_v <= 0:
            return False
        current_v = vols[-1]
        z = (current_v - mean_v) / std_v
        return z > self.get_param("vol_of_vol_threshold")

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._ensure_buffers(symbol)
        close = bar["close"]
        high = bar["high"]
        low = bar["low"]

        prev = self._close[symbol][-1] if self._close[symbol] else close
        ret = (close - prev) / prev if prev > 0 else 0.0

        self._close[symbol].append(close)
        self._high[symbol].append(high)
        self._low[symbol].append(low)
        self._returns[symbol].append(ret)
        self._bar_count[symbol] = self._bar_count.get(symbol, 0) + 1

        vol_lb = self.get_param("vol_lookback")
        returns = list(self._returns[symbol])
        if len(returns) >= vol_lb:
            vol = math.sqrt(sum(r**2 for r in returns[-vol_lb:]) / vol_lb)
            self._vol_history[symbol].append(vol)

        atr = calc_atr(
            self._high[symbol], self._low[symbol], self._close[symbol],
            self.get_param("atr_period"),
        )

        signals: list[Signal] = []
        if atr is None or atr <= 0:
            return signals

        score = self._vol_adjusted_momentum(symbol)
        if score is None:
            return signals

        pos = self.get_position(symbol)
        entry_thresh = self.get_param("entry_threshold")
        exit_thresh = self.get_param("exit_threshold")

        if pos is None:
            self._bars_since_exit[symbol] = self._bars_since_exit.get(symbol, 99) + 1
            if self._vol_of_vol_spike(symbol):
                return signals
            min_gap = self.get_param("min_bars_between_trades")
            if self._bars_since_exit.get(symbol, 99) < min_gap:
                return signals

            if score > entry_thresh:
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=round(min(abs(score) / (entry_thresh * 3), 1.0), 4),
                    price=close,
                    reason=f"TS-MOM 做多(score={score:.3f})",
                    metadata={"momentum_score": score},
                )
                signals.append(sig)
                self.record_signal(sig)
                self._hold_bars[symbol] = 0

            elif score < -entry_thresh:
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=round(min(abs(score) / (entry_thresh * 3), 1.0), 4),
                    price=close,
                    reason=f"TS-MOM 做空(score={score:.3f})",
                    metadata={"momentum_score": score},
                )
                signals.append(sig)
                self.record_signal(sig)
                self._hold_bars[symbol] = 0

        else:
            self._hold_bars[symbol] = self._hold_bars.get(symbol, 0) + 1

            should_exit, reason = check_atr_exit(
                side=pos.side.value,
                close=close,
                avg_price=pos.avg_price,
                atr=atr,
                hold_bars=self._hold_bars[symbol],
                sl_mult=self.get_param("stop_loss_atr_mult"),
                tp_mult=self.get_param("take_profit_atr_mult"),
                max_hold=self.get_param("max_hold_bars"),
            )

            if not should_exit:
                if pos.side.value == "buy" and score < -exit_thresh:
                    should_exit, reason = True, f"动量反转(score={score:.3f})"
                elif pos.side.value == "sell" and score > exit_thresh:
                    should_exit, reason = True, f"动量反转(score={score:.3f})"

            if should_exit:
                exit_type = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_type, strength=0.8,
                    price=close, reason=f"TS-MOM平仓: {reason}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._hold_bars[symbol] = 0
                self._bars_since_exit[symbol] = 0

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        if len(self.config.symbols) <= 1:
            all_signals: list[Signal] = []
            for symbol in self.config.symbols:
                bar = market_data.get(symbol)
                if bar:
                    sigs = await self.on_bar(symbol, bar)
                    all_signals.extend(sigs)
            return all_signals

        for symbol in self.config.symbols:
            bar = market_data.get(symbol)
            if bar:
                self._ensure_buffers(symbol)
                close = bar["close"]
                prev = self._close[symbol][-1] if self._close[symbol] else close
                ret = (close - prev) / prev if prev > 0 else 0.0
                self._close[symbol].append(close)
                self._high[symbol].append(bar["high"])
                self._low[symbol].append(bar["low"])
                self._returns[symbol].append(ret)
                self._bar_count[symbol] = self._bar_count.get(symbol, 0) + 1
                vol_lb = self.get_param("vol_lookback")
                returns = list(self._returns[symbol])
                if len(returns) >= vol_lb:
                    vol = math.sqrt(sum(r**2 for r in returns[-vol_lb:]) / vol_lb)
                    self._vol_history[symbol].append(vol)

        scored: list[tuple[str, float]] = []
        for symbol in self.config.symbols:
            score = self._vol_adjusted_momentum(symbol)
            if score is not None:
                scored.append((symbol, score))

        if len(scored) < 2:
            return []

        scored.sort(key=lambda x: x[1], reverse=True)
        signals: list[Signal] = []
        top = scored[0]
        bottom = scored[-1]
        entry_thresh = self.get_param("entry_threshold")

        if top[1] > entry_thresh and self.get_position(top[0]) is None:
            if not self._vol_of_vol_spike(top[0]):
                close = float(self._close[top[0]][-1])
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=top[0],
                    signal_type=SignalType.LONG_ENTRY,
                    strength=round(min(abs(top[1]) / (entry_thresh * 3), 1.0), 4),
                    price=close,
                    reason=f"XS-MOM Long top(score={top[1]:.3f}, rank=1/{len(scored)})",
                    metadata={"momentum_score": top[1], "rank": 1, "n_symbols": len(scored)},
                )
                signals.append(sig)
                self.record_signal(sig)

        if bottom[1] < -entry_thresh and self.get_position(bottom[0]) is None:
            if not self._vol_of_vol_spike(bottom[0]):
                close = float(self._close[bottom[0]][-1])
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=bottom[0],
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=round(min(abs(bottom[1]) / (entry_thresh * 3), 1.0), 4),
                    price=close,
                    reason=f"XS-MOM Short bottom(score={bottom[1]:.3f}, rank={len(scored)}/{len(scored)})",
                    metadata={"momentum_score": bottom[1], "rank": len(scored), "n_symbols": len(scored)},
                )
                signals.append(sig)
                self.record_signal(sig)

        for symbol in self.config.symbols:
            pos = self.get_position(symbol)
            if pos is not None:
                self._hold_bars[symbol] = self._hold_bars.get(symbol, 0) + 1
                atr = calc_atr(
                    self._high[symbol], self._low[symbol], self._close[symbol],
                    self.get_param("atr_period"),
                )
                if atr and atr > 0:
                    close = float(self._close[symbol][-1])
                    should_exit, reason = check_atr_exit(
                        side=pos.side.value, close=close, avg_price=pos.avg_price,
                        atr=atr, hold_bars=self._hold_bars[symbol],
                        sl_mult=self.get_param("stop_loss_atr_mult"),
                        tp_mult=self.get_param("take_profit_atr_mult"),
                        max_hold=self.get_param("max_hold_bars"),
                    )
                    if should_exit:
                        exit_type = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                        sig = Signal(
                            strategy_id=self.strategy_id, symbol=symbol,
                            signal_type=exit_type, strength=0.8,
                            price=close, reason=f"XS-MOM Exit: {reason}",
                        )
                        signals.append(sig)
                        self.record_signal(sig)
                        self._hold_bars[symbol] = 0
                        self._bars_since_exit[symbol] = 0

        return signals
