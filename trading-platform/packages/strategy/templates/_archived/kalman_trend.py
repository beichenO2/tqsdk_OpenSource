"""Kalman 滤波趋势策略 — 用卡尔曼滤波器估计隐含趋势，过滤噪声。

核心思路 (SOTA):
- 将价格视为含噪声的线性动态系统: x_t = x_{t-1} + v_{t-1} + w_t
- 卡尔曼滤波器输出平滑价格(state)和速度(velocity)
- velocity > threshold → 做多; velocity < -threshold → 做空
- 比传统 MA 延迟更低，在快速趋势启动时优势明显
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..indicators import calc_atr
from ..registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "process_noise": 0.01,          # Q: 过程噪声方差
    "measurement_noise": 0.5,       # R: 观测噪声方差
    "velocity_threshold": 0.001,    # 开仓所需的最小速度
    "atr_period": 14,
    "trailing_stop_atr_mult": 2.5,
    "max_hold_bars": 200,
    "min_warmup": 20,
}


@auto_register("kalman_trend")
class KalmanTrendStrategy(BaseStrategy):
    """基于卡尔曼滤波的自适应趋势跟踪。

    状态向量 [price, velocity] 通过预测-更新循环实时估计，
    velocity 的符号和大小决定趋势方向和信号强度。
    """

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        # Kalman 状态 per symbol
        self._state: dict[str, list[float]] = {}  # [price, velocity]
        self._P: dict[str, list[list[float]]] = {}  # 2x2 协方差矩阵
        self._bars_count: dict[str, int] = {}
        self._bars_in_pos: dict[str, int] = {}
        self._peak: dict[str, float] = {}
        self._trough: dict[str, float] = {}
        self._high_buf: dict[str, deque[float]] = {}
        self._low_buf: dict[str, deque[float]] = {}
        self._close_buf: dict[str, deque[float]] = {}

    def _init_kalman(self, symbol: str, price: float) -> None:
        self._state[symbol] = [price, 0.0]
        self._P[symbol] = [[1.0, 0.0], [0.0, 1.0]]
        atr_p = int(self.get_param("atr_period"))
        buf_len = atr_p + 10
        self._high_buf[symbol] = deque(maxlen=buf_len)
        self._low_buf[symbol] = deque(maxlen=buf_len)
        self._close_buf[symbol] = deque(maxlen=buf_len)
        self._bars_count[symbol] = 0

    def _predict(self, symbol: str) -> None:
        """预测步：状态转移。"""
        s = self._state[symbol]
        P = self._P[symbol]
        Q = float(self.get_param("process_noise"))
        # F = [[1, 1], [0, 1]]
        new_s = [s[0] + s[1], s[1]]
        new_P = [
            [P[0][0] + P[0][1] + P[1][0] + P[1][1] + Q, P[0][1] + P[1][1]],
            [P[1][0] + P[1][1], P[1][1] + Q],
        ]
        self._state[symbol] = new_s
        self._P[symbol] = new_P

    def _update(self, symbol: str, measurement: float) -> None:
        """更新步：融合观测。"""
        s = self._state[symbol]
        P = self._P[symbol]
        R = float(self.get_param("measurement_noise"))
        # H = [1, 0], innovation
        y = measurement - s[0]
        S = P[0][0] + R
        if abs(S) < 1e-15:
            return
        K = [P[0][0] / S, P[1][0] / S]
        self._state[symbol] = [s[0] + K[0] * y, s[1] + K[1] * y]
        self._P[symbol] = [
            [(1 - K[0]) * P[0][0], (1 - K[0]) * P[0][1]],
            [-K[1] * P[0][0] + P[1][0], -K[1] * P[0][1] + P[1][1]],
        ]

    def _calc_atr(self, symbol: str) -> float | None:
        return calc_atr(self._high_buf[symbol], self._low_buf[symbol], self._close_buf[symbol], int(self.get_param("atr_period")))

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        close = float(bar["close"])
        high = float(bar["high"])
        low = float(bar["low"])

        if symbol not in self._state:
            self._init_kalman(symbol, close)

        self._high_buf[symbol].append(high)
        self._low_buf[symbol].append(low)
        self._close_buf[symbol].append(close)
        self._bars_count[symbol] = self._bars_count.get(symbol, 0) + 1

        self._predict(symbol)
        self._update(symbol, close)

        warmup = int(self.get_param("min_warmup"))
        if self._bars_count[symbol] < warmup:
            return []

        velocity = self._state[symbol][1]
        threshold = float(self.get_param("velocity_threshold"))
        atr = self._calc_atr(symbol)

        signals: list[Signal] = []
        pos = self.get_position(symbol)

        if pos is None and atr is not None and atr > 0:
            if velocity > threshold:
                strength = min(abs(velocity) / (threshold * 5), 1.0)
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=round(strength, 4), price=close,
                    reason=f"Kalman velocity={velocity:.6f} > {threshold}",
                    metadata={"kalman_price": self._state[symbol][0], "velocity": velocity},
                ))
                self._peak[symbol] = close
            elif velocity < -threshold:
                strength = min(abs(velocity) / (threshold * 5), 1.0)
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=round(strength, 4), price=close,
                    reason=f"Kalman velocity={velocity:.6f} < -{threshold}",
                    metadata={"kalman_price": self._state[symbol][0], "velocity": velocity},
                ))
                self._trough[symbol] = close

        elif pos is not None and atr is not None:
            self._bars_in_pos[symbol] = self._bars_in_pos.get(symbol, 0) + 1
            stop_mult = float(self.get_param("trailing_stop_atr_mult"))
            max_hold = int(self.get_param("max_hold_bars"))

            if self._bars_in_pos.get(symbol, 0) >= max_hold:
                exit_t = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_t, strength=0.6, price=close,
                    reason=f"最大持仓{max_hold}bars",
                ))
                self._bars_in_pos[symbol] = 0
            elif pos.side.value == "buy":
                self._peak[symbol] = max(self._peak.get(symbol, close), close)
                trail = self._peak[symbol] - atr * stop_mult
                if close < trail or velocity < -threshold * 0.5:
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.LONG_EXIT, strength=0.85, price=close,
                        reason=f"Kalman平多 trail={trail:.2f} vel={velocity:.6f}",
                    ))
                    self._bars_in_pos[symbol] = 0
            elif pos.side.value == "sell":
                self._trough[symbol] = min(self._trough.get(symbol, close), close)
                trail = self._trough[symbol] + atr * stop_mult
                if close > trail or velocity > threshold * 0.5:
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.SHORT_EXIT, strength=0.85, price=close,
                        reason=f"Kalman平空 trail={trail:.2f} vel={velocity:.6f}",
                    ))
                    self._bars_in_pos[symbol] = 0
        else:
            self._bars_in_pos[symbol] = 0

        for s in signals:
            self.record_signal(s)
        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        all_signals: list[Signal] = []
        for symbol in self.config.symbols:
            bar = market_data.get(symbol)
            if bar:
                all_signals.extend(await self.on_bar(symbol, bar))
        return all_signals
