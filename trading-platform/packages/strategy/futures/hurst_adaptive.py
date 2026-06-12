"""Hurst 指数自适应策略 — 根据市场特征自动切换趋势/回归模式。

理论：Hurst 指数 (Hurst 1951, 经典水文学/分形分析)
  H > 0.5 → 趋势持续性（长记忆，适合趋势跟踪）
  H ≈ 0.5 → 随机游走（无可预测性）
  H < 0.5 → 均值回归（反持续性，适合逆势）

计算方法：R/S (Rescaled Range) 分析
  - 经典方法，Mandelbrot (1963) 引入金融领域
  - 滚动窗口计算 → 动态判断当前市场属性

策略逻辑：
  1. 滚动计算 Hurst 指数
  2. H > 0.6 → 激活趋势模式（跟踪当前方向）
  3. H < 0.4 → 激活回归模式（逆当前方向）
  4. 0.4 ≤ H ≤ 0.6 → 不交易（随机区域）
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
    "hurst_window": 100,
    "hurst_trend_threshold": 0.6,
    "hurst_mr_threshold": 0.4,
    "trend_lookback": 20,
    "mr_lookback": 10,
    "atr_period": 14,
    "tp_atr_mult": 2.0,
    "sl_atr_mult": 1.0,
    "max_hold_bars": 40,
    "cooldown_bars": 3,
}


def compute_hurst(prices: np.ndarray) -> float:
    """Compute Hurst exponent using R/S analysis.

    Returns H ∈ (0, 1):
      H > 0.5: trending (persistent)
      H ≈ 0.5: random walk
      H < 0.5: mean-reverting (anti-persistent)
    """
    n = len(prices)
    if n < 20:
        return 0.5

    returns = np.diff(np.log(np.maximum(prices, 1e-10)))
    if len(returns) < 10:
        return 0.5

    max_k = min(len(returns) // 2, 50)
    min_k = 10

    if max_k <= min_k:
        return 0.5

    rs_values = []
    ns = []

    for k in range(min_k, max_k + 1, max(1, (max_k - min_k) // 8)):
        n_chunks = len(returns) // k
        if n_chunks < 1:
            continue

        rs_chunk = []
        for i in range(n_chunks):
            chunk = returns[i * k:(i + 1) * k]
            mean_r = np.mean(chunk)
            deviations = np.cumsum(chunk - mean_r)
            r = np.max(deviations) - np.min(deviations)
            s = np.std(chunk, ddof=1)
            if s > 1e-10:
                rs_chunk.append(r / s)

        if rs_chunk:
            rs_values.append(np.mean(rs_chunk))
            ns.append(k)

    if len(ns) < 3:
        return 0.5

    log_n = np.log(ns)
    log_rs = np.log(np.maximum(rs_values, 1e-10))

    coeffs = np.polyfit(log_n, log_rs, 1)
    hurst = float(np.clip(coeffs[0], 0.01, 0.99))

    return hurst


@auto_register("hurst_adaptive")
class HurstAdaptiveStrategy(BaseStrategy):
    """Hurst 指数自适应策略 — 自动判断市场属性并切换交易模式。"""

    def __init__(self, config: StrategyConfig) -> None:
        config = config.model_copy(
            update={"params": {**DEFAULT_PARAMS, **config.params}}
        )
        super().__init__(config)
        self._closes: deque[float] = deque(maxlen=300)
        self._highs: deque[float] = deque(maxlen=300)
        self._lows: deque[float] = deque(maxlen=300)
        self._bar_count = 0
        self._position_side: str | None = None
        self._entry_price = 0.0
        self._hold_bars = 0
        self._cd = 0
        self._last_hurst = 0.5

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        h = float(bar.get("high", 0))
        l = float(bar.get("low", 0))
        c = float(bar.get("close", 0))

        self._highs.append(h)
        self._lows.append(l)
        self._closes.append(c)
        self._bar_count += 1

        hurst_window = self.get_param("hurst_window", 100)
        if self._bar_count < hurst_window + 5:
            return []

        atr = calc_atr(list(self._highs), list(self._lows), list(self._closes),
                       self.get_param("atr_period", 14))
        if atr is None or atr < 1e-10:
            return []

        close_arr = np.array(list(self._closes)[-hurst_window:])
        self._last_hurst = compute_hurst(close_arr)

        signals = []

        if self._position_side:
            self._hold_bars += 1
            max_hold = self.get_param("max_hold_bars", 40)
            tp_mult = self.get_param("tp_atr_mult", 2.0)
            sl_mult = self.get_param("sl_atr_mult", 1.0)

            pnl = (c - self._entry_price) / self._entry_price if self._position_side == "long" else (self._entry_price - c) / self._entry_price
            tp_hit = pnl >= tp_mult * atr / self._entry_price
            sl_hit = pnl <= -sl_mult * atr / self._entry_price

            if sl_hit or tp_hit or self._hold_bars >= max_hold:
                exit_type = SignalType.LONG_EXIT if self._position_side == "long" else SignalType.SHORT_EXIT
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_type, strength=0.8, price=c,
                    reason=f"hurst_exit: H={self._last_hurst:.3f} hold={self._hold_bars}",
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
            trend_threshold = self.get_param("hurst_trend_threshold", 0.6)
            mr_threshold = self.get_param("hurst_mr_threshold", 0.4)

            close_list = list(self._closes)

            if self._last_hurst > trend_threshold:
                trend_lb = self.get_param("trend_lookback", 20)
                if len(close_list) >= trend_lb:
                    trend = (close_list[-1] - close_list[-trend_lb]) / close_list[-trend_lb]

                    if trend > 0.003:
                        sig = Signal(
                            strategy_id=self.strategy_id, symbol=symbol,
                            signal_type=SignalType.LONG_ENTRY,
                            strength=min(self._last_hurst, 1.0), price=c,
                            reason=f"hurst_trend_buy: H={self._last_hurst:.3f} trend={trend:.4f}",
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
                            strength=min(self._last_hurst, 1.0), price=c,
                            reason=f"hurst_trend_sell: H={self._last_hurst:.3f} trend={trend:.4f}",
                        )
                        signals.append(sig)
                        self.record_signal(sig)
                        self._position_side = "short"
                        self._entry_price = c
                        self._hold_bars = 0

            elif self._last_hurst < mr_threshold:
                mr_lb = self.get_param("mr_lookback", 10)
                if len(close_list) >= mr_lb:
                    sma = np.mean(close_list[-mr_lb:])
                    dev = (c - sma) / atr

                    if dev < -1.5:
                        sig = Signal(
                            strategy_id=self.strategy_id, symbol=symbol,
                            signal_type=SignalType.LONG_ENTRY,
                            strength=min(1.0 - self._last_hurst, 1.0), price=c,
                            reason=f"hurst_mr_buy: H={self._last_hurst:.3f} dev={dev:.2f}",
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
                            strength=min(1.0 - self._last_hurst, 1.0), price=c,
                            reason=f"hurst_mr_sell: H={self._last_hurst:.3f} dev={dev:.2f}",
                        )
                        signals.append(sig)
                        self.record_signal(sig)
                        self._position_side = "short"
                        self._entry_price = c
                        self._hold_bars = 0

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        return []
