"""卡尔曼滤波趋势策略 — 自适应趋势检测与噪声过滤。

卡尔曼滤波器（Kalman 1960，经典信号处理）应用于金融时间序列：
  - 状态向量: [价格水平, 价格斜率(趋势)]
  - 观测: 收盘价（含噪声）
  - 自适应: 滤波器自动调整对新数据的响应速度
  - 优势: 比 MA/EMA 更快响应趋势变化，同时更好地过滤噪声

信号构建：
  1. 卡尔曼估计斜率 > 0 且置信度高 → 做多
  2. 卡尔曼估计斜率 < 0 且置信度高 → 做空
  3. 协方差矩阵对角线 → 估计不确定性 → 影响仓位大小
  4. 卡尔曼残差异常（Innovation） → 结构性突变检测

Method: Kalman Filter (Kalman 1960, 经典控制论/信号处理，工业级广泛使用)
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
    "process_noise_level": 0.01,
    "measurement_noise_level": 1.0,
    "slope_threshold": 0.0002,
    "confidence_threshold": 0.6,
    "innovation_threshold": 3.0,
    "atr_period": 14,
    "tp_atr_mult": 2.5,
    "sl_atr_mult": 1.2,
    "max_hold_bars": 50,
    "cooldown_bars": 3,
}


class KalmanTrendFilter:
    """1D Kalman Filter with level + slope state.

    State: x = [level, slope]^T
    Transition: x_{t+1} = F x_t + w,  w ~ N(0, Q)
    Observation: z_t = H x_t + v,     v ~ N(0, R)
    """

    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 1.0):
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.Q = np.eye(2) * process_noise
        self.R = np.array([[measurement_noise]])

        self.x = np.array([0.0, 0.0])
        self.P = np.eye(2) * 100.0
        self._initialized = False

    def update(self, z: float) -> dict[str, float]:
        if not self._initialized:
            self.x = np.array([z, 0.0])
            self._initialized = True
            return {"level": z, "slope": 0.0, "innovation": 0.0, "confidence": 0.0}

        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        innovation = z - (self.H @ x_pred)[0]
        self.x = x_pred + (K @ np.array([[innovation]])).flatten()
        self.P = (np.eye(2) - K @ self.H) @ P_pred

        level_var = self.P[0, 0]
        slope_var = self.P[1, 1]
        slope_confidence = 1.0 / (1.0 + slope_var) if slope_var > 0 else 1.0

        return {
            "level": float(self.x[0]),
            "slope": float(self.x[1]),
            "innovation": float(innovation),
            "innovation_std": float(np.sqrt(S[0, 0])),
            "confidence": float(slope_confidence),
            "level_uncertainty": float(np.sqrt(level_var)),
        }


@auto_register("kalman_trend")
class KalmanTrendStrategy(BaseStrategy):
    """卡尔曼滤波趋势策略 — 自适应趋势检测。"""

    def __init__(self, config: StrategyConfig) -> None:
        config = config.model_copy(
            update={"params": {**DEFAULT_PARAMS, **config.params}}
        )
        super().__init__(config)
        self._kf = KalmanTrendFilter(
            process_noise=self.get_param("process_noise_level", 0.01),
            measurement_noise=self.get_param("measurement_noise_level", 1.0),
        )
        self._highs: deque[float] = deque(maxlen=200)
        self._lows: deque[float] = deque(maxlen=200)
        self._closes: deque[float] = deque(maxlen=200)
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

        kf_result = self._kf.update(c)

        if self._bar_count < 20:
            return []

        atr = calc_atr(list(self._highs), list(self._lows), list(self._closes),
                       self.get_param("atr_period", 14))
        if atr is None or atr < 1e-10:
            return []

        signals = []
        slope = kf_result["slope"]
        confidence = kf_result["confidence"]
        innovation = kf_result["innovation"]
        innov_std = kf_result.get("innovation_std", 1.0)

        slope_threshold = self.get_param("slope_threshold", 0.0002)
        conf_threshold = self.get_param("confidence_threshold", 0.6)
        innov_threshold = self.get_param("innovation_threshold", 3.0)

        structural_break = abs(innovation) > innov_threshold * innov_std if innov_std > 0 else False

        if self._position_side:
            self._hold_bars += 1
            max_hold = self.get_param("max_hold_bars", 50)
            tp_mult = self.get_param("tp_atr_mult", 2.5)
            sl_mult = self.get_param("sl_atr_mult", 1.2)

            pnl = (c - self._entry_price) / self._entry_price if self._position_side == "long" else (self._entry_price - c) / self._entry_price

            trend_reversed = (self._position_side == "long" and slope < -slope_threshold) or \
                            (self._position_side == "short" and slope > slope_threshold)

            tp_hit = pnl >= tp_mult * atr / self._entry_price
            sl_hit = pnl <= -sl_mult * atr / self._entry_price

            if sl_hit or tp_hit or self._hold_bars >= max_hold or (trend_reversed and confidence > conf_threshold):
                exit_type = SignalType.LONG_EXIT if self._position_side == "long" else SignalType.SHORT_EXIT
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_type, strength=confidence, price=c,
                    reason=f"kalman_exit: slope={slope:.6f} conf={confidence:.2f} hold={self._hold_bars}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = None
                self._hold_bars = 0
                self._cd = self.get_param("cooldown_bars", 3)
                return signals

        if self._cd > 0:
            self._cd -= 1

        if not self._position_side and self._cd <= 0 and not structural_break:
            if slope > slope_threshold and confidence > conf_threshold:
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=min(confidence, 1.0), price=c,
                    reason=f"kalman_buy: slope={slope:.6f} conf={confidence:.2f}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "long"
                self._entry_price = c
                self._hold_bars = 0

            elif slope < -slope_threshold and confidence > conf_threshold:
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=min(confidence, 1.0), price=c,
                    reason=f"kalman_sell: slope={slope:.6f} conf={confidence:.2f}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "short"
                self._entry_price = c
                self._hold_bars = 0

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        return []
