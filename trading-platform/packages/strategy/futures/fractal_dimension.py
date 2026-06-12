"""分形维度策略 — 用分形维度判断市场复杂度并自适应交易。

理论：
  - Mandelbrot 分形理论 (1975, 经典)
  - Higuchi 分形维度算法 (Higuchi 1988, 经典信号处理)
  - FD ≈ 1.0: 低复杂度（价格走直线 = 强趋势）
  - FD ≈ 1.5: 中等复杂度（随机游走）
  - FD ≈ 2.0: 高复杂度（价格高度不规则 = 震荡/反转）

与 Hurst 指数互补：
  - Hurst 测量记忆（持续性 vs 反持续性）
  - FD 测量复杂度（规则 vs 混沌）
  - 两者结合 → 更精确的市场状态判断
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..indicators import calc_atr
from ..registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS: dict[str, Any] = {
    "fd_window": 64,
    "fd_kmax": 8,
    "fd_trend_threshold": 1.3,
    "fd_chaos_threshold": 1.7,
    "trend_lookback": 15,
    "mr_lookback": 10,
    "atr_period": 14,
    "tp_atr_mult": 2.0,
    "sl_atr_mult": 1.0,
    "max_hold_bars": 35,
    "cooldown_bars": 3,
}


def higuchi_fd(x: np.ndarray, kmax: int = 8) -> float:
    """Higuchi fractal dimension (Higuchi 1988).

    Returns FD ∈ [1, 2]:
      ~1.0 = smooth/trending, ~1.5 = random walk, ~2.0 = space-filling/chaotic
    """
    n = len(x)
    if n < kmax * 2:
        return 1.5

    lk = []
    ks = []

    for k in range(1, kmax + 1):
        lengths = []
        for m in range(1, k + 1):
            indices = np.arange(m - 1, n, k)
            if len(indices) < 2:
                continue
            segment = x[indices]
            length = np.sum(np.abs(np.diff(segment))) * (n - 1) / (k * len(indices) * k)
            lengths.append(length)

        if lengths:
            lk.append(np.mean(lengths))
            ks.append(k)

    if len(ks) < 3:
        return 1.5

    log_k = np.log(1.0 / np.array(ks))
    log_l = np.log(np.maximum(np.array(lk), 1e-10))

    coeffs = np.polyfit(log_k, log_l, 1)
    fd = float(np.clip(coeffs[0], 1.0, 2.0))
    return fd


@auto_register("fractal_dimension")
class FractalDimensionStrategy(BaseStrategy):
    """分形维度自适应策略 — 根据价格复杂度切换交易模式。"""

    def __init__(self, config: StrategyConfig) -> None:
        config = config.model_copy(
            update={"params": {**DEFAULT_PARAMS, **config.params}}
        )
        super().__init__(config)
        self._closes: deque[float] = deque(maxlen=200)
        self._highs: deque[float] = deque(maxlen=200)
        self._lows: deque[float] = deque(maxlen=200)
        self._bar_count = 0
        self._position_side: str | None = None
        self._entry_price = 0.0
        self._hold_bars = 0
        self._cd = 0

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        h = float(bar.get("high", 0))
        l = float(bar.get("low", 0))
        c = float(bar.get("close", 0))

        self._highs.append(h)
        self._lows.append(l)
        self._closes.append(c)
        self._bar_count += 1

        fd_window = self.get_param("fd_window", 64)
        if self._bar_count < fd_window + 5:
            return []

        atr = calc_atr(list(self._highs), list(self._lows), list(self._closes),
                       self.get_param("atr_period", 14))
        if atr is None or atr < 1e-10:
            return []

        close_arr = np.array(list(self._closes)[-fd_window:])
        fd = higuchi_fd(close_arr, self.get_param("fd_kmax", 8))

        signals = []

        if self._position_side:
            self._hold_bars += 1
            max_hold = self.get_param("max_hold_bars", 35)
            tp_mult = self.get_param("tp_atr_mult", 2.0)
            sl_mult = self.get_param("sl_atr_mult", 1.0)

            pnl = (c - self._entry_price) / self._entry_price if self._position_side == "long" else (self._entry_price - c) / self._entry_price
            if pnl >= tp_mult * atr / self._entry_price or pnl <= -sl_mult * atr / self._entry_price or self._hold_bars >= max_hold:
                exit_type = SignalType.LONG_EXIT if self._position_side == "long" else SignalType.SHORT_EXIT
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_type, strength=0.8, price=c,
                    reason=f"fd_exit: FD={fd:.3f} hold={self._hold_bars}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = None
                self._hold_bars = 0
                self._cd = self.get_param("cooldown_bars", 3)
                return signals

        if self._cd > 0:
            self._cd -= 1

        if not self._position_side and self._cd <= 0:
            close_list = list(self._closes)
            trend_threshold = self.get_param("fd_trend_threshold", 1.3)
            chaos_threshold = self.get_param("fd_chaos_threshold", 1.7)

            if fd < trend_threshold:
                lb = self.get_param("trend_lookback", 15)
                if len(close_list) >= lb:
                    trend = (close_list[-1] - close_list[-lb]) / close_list[-lb]
                    if trend > 0.003:
                        sig = Signal(
                            strategy_id=self.strategy_id, symbol=symbol,
                            signal_type=SignalType.LONG_ENTRY,
                            strength=min((trend_threshold - fd) * 2, 1.0), price=c,
                            reason=f"fd_trend_buy: FD={fd:.3f} trend={trend:.4f}",
                        )
                        signals.append(sig)
                        self.record_signal(sig)
                        self._position_side = "long"
                        self._entry_price = c
                        self._hold_bars = 0
                    elif trend < -0.003:
                        sig = Signal(
                            strategy_id=self.strategy_id, symbol=symbol,
                            signal_type=SignalType.SHORT_ENTRY,
                            strength=min((trend_threshold - fd) * 2, 1.0), price=c,
                            reason=f"fd_trend_sell: FD={fd:.3f} trend={trend:.4f}",
                        )
                        signals.append(sig)
                        self.record_signal(sig)
                        self._position_side = "short"
                        self._entry_price = c
                        self._hold_bars = 0

            elif fd > chaos_threshold:
                lb = self.get_param("mr_lookback", 10)
                if len(close_list) >= lb:
                    sma = np.mean(close_list[-lb:])
                    dev = (c - sma) / atr
                    if dev < -1.5:
                        sig = Signal(
                            strategy_id=self.strategy_id, symbol=symbol,
                            signal_type=SignalType.LONG_ENTRY,
                            strength=min((fd - chaos_threshold) * 2, 1.0), price=c,
                            reason=f"fd_mr_buy: FD={fd:.3f} dev={dev:.2f}",
                        )
                        signals.append(sig)
                        self.record_signal(sig)
                        self._position_side = "long"
                        self._entry_price = c
                        self._hold_bars = 0
                    elif dev > 1.5:
                        sig = Signal(
                            strategy_id=self.strategy_id, symbol=symbol,
                            signal_type=SignalType.SHORT_ENTRY,
                            strength=min((fd - chaos_threshold) * 2, 1.0), price=c,
                            reason=f"fd_mr_sell: FD={fd:.3f} dev={dev:.2f}",
                        )
                        signals.append(sig)
                        self.record_signal(sig)
                        self._position_side = "short"
                        self._entry_price = c
                        self._hold_bars = 0

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        return []
