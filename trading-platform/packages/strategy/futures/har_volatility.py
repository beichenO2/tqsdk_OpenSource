"""HAR-RV 波动率预测策略 — 已实现波动率的多尺度回归预测。

理论：HAR-RV (Heterogeneous Autoregressive Realized Volatility)
      Corsi (2009, J. Financial Econometrics) — 经典，白名单

核心思想：不同时间尺度的交易者（日内/周/月）共同影响波动率。
  RV_{t+1} = α + β_d * RV_d + β_w * RV_w + β_m * RV_m + ε

应用于日内期货交易：
  1. 从 5min K 线计算日内已实现波动率（RV = Σ r_i^2）
  2. HAR 回归预测下一时段的 RV
  3. 高 RV 预测 → 趋势策略（波动放大 = 趋势延续信号）
  4. 低 RV 预测 → 均值回归策略（波动收缩 = 区间震荡）
  5. RV 跳变检测 → 风控信号（可能有重大事件）
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
    "rv_window_daily": 12,
    "rv_window_weekly": 60,
    "rv_window_monthly": 240,
    "high_vol_threshold": 1.5,
    "low_vol_threshold": 0.5,
    "trend_lookback": 20,
    "mr_lookback": 10,
    "atr_period": 14,
    "tp_atr_mult": 2.0,
    "sl_atr_mult": 1.0,
    "max_hold_bars": 30,
    "cooldown_bars": 3,
}


def _realized_volatility(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    return float(np.sqrt(np.sum(np.array(returns) ** 2)))


def _har_predict(rv_d: float, rv_w: float, rv_m: float) -> float:
    """Simplified HAR-RV prediction with default coefficients.

    Based on typical estimates from Corsi (2009):
    RV_{t+1} ≈ 0.1 + 0.4 * RV_d + 0.35 * RV_w + 0.25 * RV_m
    """
    return 0.1 * rv_d + 0.4 * rv_d + 0.35 * rv_w + 0.25 * rv_m


@auto_register("har_volatility")
class HARVolatilityStrategy(BaseStrategy):
    """HAR-RV 波动率预测策略 — 根据预测波动率切换趋势/回归模式。"""

    def __init__(self, config: StrategyConfig) -> None:
        config = config.model_copy(
            update={"params": {**DEFAULT_PARAMS, **config.params}}
        )
        super().__init__(config)
        self._closes: deque[float] = deque(maxlen=300)
        self._highs: deque[float] = deque(maxlen=300)
        self._lows: deque[float] = deque(maxlen=300)
        self._returns: deque[float] = deque(maxlen=300)
        self._rv_history: deque[float] = deque(maxlen=100)
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

        if self._closes:
            ret = (c - self._closes[-1]) / self._closes[-1] if self._closes[-1] > 0 else 0.0
            self._returns.append(ret)

        self._closes.append(c)
        self._bar_count += 1

        rv_d_window = self.get_param("rv_window_daily", 12)
        rv_w_window = self.get_param("rv_window_weekly", 60)
        rv_m_window = self.get_param("rv_window_monthly", 240)

        if self._bar_count < max(rv_m_window + 5, 50):
            return []

        atr = calc_atr(list(self._highs), list(self._lows), list(self._closes),
                       self.get_param("atr_period", 14))
        if atr is None or atr < 1e-10:
            return []

        ret_list = list(self._returns)
        rv_d = _realized_volatility(ret_list[-rv_d_window:])
        rv_w = _realized_volatility(ret_list[-rv_w_window:])
        rv_m = _realized_volatility(ret_list[-rv_m_window:])

        rv_forecast = _har_predict(rv_d, rv_w, rv_m)
        self._rv_history.append(rv_forecast)

        rv_mean = np.mean(list(self._rv_history)) if len(self._rv_history) > 10 else rv_forecast
        rv_ratio = rv_forecast / max(rv_mean, 1e-10)

        high_vol = rv_ratio > self.get_param("high_vol_threshold", 1.5)
        low_vol = rv_ratio < self.get_param("low_vol_threshold", 0.5)

        signals = []

        if self._position_side:
            self._hold_bars += 1
            max_hold = self.get_param("max_hold_bars", 30)
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
                    reason=f"har_exit: rv_ratio={rv_ratio:.2f} hold={self._hold_bars}",
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

            if high_vol:
                trend_lb = self.get_param("trend_lookback", 20)
                if len(close_list) >= trend_lb:
                    trend = (close_list[-1] - close_list[-trend_lb]) / close_list[-trend_lb]
                    if trend > 0.005:
                        sig = Signal(
                            strategy_id=self.strategy_id, symbol=symbol,
                            signal_type=SignalType.LONG_ENTRY,
                            strength=min(rv_ratio / 2.0, 1.0), price=c,
                            reason=f"har_trend_buy: rv_ratio={rv_ratio:.2f} trend={trend:.4f}",
                        )
                        signals.append(sig)
                        self.record_signal(sig)
                        self._position_side = "long"
                        self._entry_price = c
                        self._hold_bars = 0
                    elif trend < -0.005:
                        sig = Signal(
                            strategy_id=self.strategy_id, symbol=symbol,
                            signal_type=SignalType.SHORT_ENTRY,
                            strength=min(rv_ratio / 2.0, 1.0), price=c,
                            reason=f"har_trend_sell: rv_ratio={rv_ratio:.2f} trend={trend:.4f}",
                        )
                        signals.append(sig)
                        self.record_signal(sig)
                        self._position_side = "short"
                        self._entry_price = c
                        self._hold_bars = 0

            elif low_vol:
                mr_lb = self.get_param("mr_lookback", 10)
                if len(close_list) >= mr_lb:
                    sma = np.mean(close_list[-mr_lb:])
                    dev = (c - sma) / atr

                    if dev < -1.5:
                        sig = Signal(
                            strategy_id=self.strategy_id, symbol=symbol,
                            signal_type=SignalType.LONG_ENTRY,
                            strength=min(abs(dev) / 3.0, 1.0), price=c,
                            reason=f"har_mr_buy: rv_ratio={rv_ratio:.2f} dev={dev:.2f}",
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
                            strength=min(abs(dev) / 3.0, 1.0), price=c,
                            reason=f"har_mr_sell: rv_ratio={rv_ratio:.2f} dev={dev:.2f}",
                        )
                        signals.append(sig)
                        self.record_signal(sig)
                        self._position_side = "short"
                        self._entry_price = c
                        self._hold_bars = 0

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        return []
