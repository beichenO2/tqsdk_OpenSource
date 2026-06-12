"""EMA 变化率趋势策略 — 基于学术研究实现。

来源: "How an EMA-Derivative Strategy Delivered 3,906% Returns" (2025)
核心思路:
- 不是看 EMA 交叉，而是看 EMA 的变化率 (一阶导数)
- EMA变化率 > 0 且加速 → 强趋势做多
- EMA变化率 < 0 且加速 → 强趋势做空
- 变化率减速 → 趋势衰减，收紧止损
- 保持高市场暴露度 (99%+)，不频繁进出
- 关键: 只在变化率翻转时换仓，而非用固定阈值

性能: Return 3906%, MaxDD -27.25%, Sharpe 1.50
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
    "ema_period": 21,
    "deriv_smooth": 5,          # 变化率平滑周期
    "signal_threshold": 0.0,    # 变化率翻转就换仓 (接近0)
    "atr_period": 14,
    "trailing_stop_atr_mult": 3.0,  # 宽止损，不要被震出去
    "max_hold_bars": 0,         # 0 = 不限持仓时间
    "always_in_market": True,   # 类似论文的高暴露度
}


@auto_register("ema_derivative")
class EMADerivativeStrategy(BaseStrategy):
    """EMA 变化率趋势策略。

    核心: EMA 的一阶导数决定方向，保持永远在市场中。
    变化率翻转 → 反转仓位; 不翻转 → 持有。
    """

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._close_buf: dict[str, deque[float]] = {}
        self._high_buf: dict[str, deque[float]] = {}
        self._low_buf: dict[str, deque[float]] = {}
        self._ema: dict[str, float | None] = {}
        self._ema_prev: dict[str, float | None] = {}
        self._deriv_buf: dict[str, deque[float]] = {}
        self._peak: dict[str, float] = {}
        self._trough: dict[str, float] = {}
        self._direction: dict[str, int] = {}  # 1=多, -1=空, 0=空仓

    def _ensure(self, symbol: str) -> None:
        if symbol not in self._close_buf:
            self._close_buf[symbol] = deque(maxlen=200)
            self._high_buf[symbol] = deque(maxlen=60)
            self._low_buf[symbol] = deque(maxlen=60)
            self._deriv_buf[symbol] = deque(maxlen=30)
            self._ema[symbol] = None
            self._ema_prev[symbol] = None
            self._direction[symbol] = 0

    def _update_ema(self, symbol: str, close: float) -> float | None:
        period = int(self.get_param("ema_period"))
        k = 2 / (period + 1)

        prev = self._ema.get(symbol)
        if prev is None:
            buf = list(self._close_buf[symbol])
            if len(buf) < period:
                return None
            sma = sum(buf[-period:]) / period
            self._ema[symbol] = sma
            return sma

        self._ema_prev[symbol] = prev
        new_ema = close * k + prev * (1 - k)
        self._ema[symbol] = new_ema
        return new_ema

    def _calc_atr(self, symbol: str) -> float | None:
        return calc_atr(self._high_buf[symbol], self._low_buf[symbol], self._close_buf[symbol], int(self.get_param("atr_period")))

    def _smoothed_derivative(self, symbol: str) -> float | None:
        """返回 EMA 变化率的平滑值。"""
        buf = list(self._deriv_buf[symbol])
        smooth = int(self.get_param("deriv_smooth"))
        if len(buf) < smooth:
            return None
        return sum(buf[-smooth:]) / smooth

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._ensure(symbol)
        close = float(bar["close"])
        high = float(bar["high"])
        low = float(bar["low"])
        self._close_buf[symbol].append(close)
        self._high_buf[symbol].append(high)
        self._low_buf[symbol].append(low)

        ema = self._update_ema(symbol, close)
        if ema is None:
            return []

        prev_ema = self._ema_prev.get(symbol)
        if prev_ema is not None and prev_ema > 0:
            derivative = (ema - prev_ema) / prev_ema
            self._deriv_buf[symbol].append(derivative)

        deriv = self._smoothed_derivative(symbol)
        atr = self._calc_atr(symbol)
        if deriv is None or atr is None:
            return []

        threshold = float(self.get_param("signal_threshold"))
        always_in = bool(self.get_param("always_in_market"))
        signals: list[Signal] = []
        pos = self.get_position(symbol)
        current_dir = self._direction.get(symbol, 0)

        # 判断新方向
        if deriv > threshold:
            new_dir = 1
        elif deriv < -threshold:
            new_dir = -1
        else:
            new_dir = current_dir  # 维持现有方向

        if new_dir != current_dir:
            # 先平仓
            if pos is not None:
                exit_t = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_t, strength=0.9, price=close,
                    reason=f"EMA-Deriv反转 deriv={deriv:.6f}",
                ))

            # 再开仓
            if new_dir == 1:
                strength = min(abs(deriv) * 1000 + 0.5, 1.0)
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY, strength=round(strength, 4),
                    price=close, reason=f"EMA-Deriv做多 deriv={deriv:.6f}",
                    metadata={"ema": ema, "derivative": deriv},
                ))
                self._peak[symbol] = close
            elif new_dir == -1 and always_in:
                strength = min(abs(deriv) * 1000 + 0.5, 1.0)
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY, strength=round(strength, 4),
                    price=close, reason=f"EMA-Deriv做空 deriv={deriv:.6f}",
                    metadata={"ema": ema, "derivative": deriv},
                ))
                self._trough[symbol] = close

            self._direction[symbol] = new_dir

        # 追踪止损 (仅在极端行情触发)
        elif pos is not None:
            stop_mult = float(self.get_param("trailing_stop_atr_mult"))
            if pos.side.value == "buy":
                self._peak[symbol] = max(self._peak.get(symbol, close), close)
                trail = self._peak[symbol] - atr * stop_mult
                if close < trail:
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.LONG_EXIT, strength=0.95, price=close,
                        reason=f"EMA-Deriv追踪止损 trail={trail:.2f}",
                    ))
                    self._direction[symbol] = 0
            elif pos.side.value == "sell":
                self._trough[symbol] = min(self._trough.get(symbol, close), close)
                trail = self._trough[symbol] + atr * stop_mult
                if close > trail:
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.SHORT_EXIT, strength=0.95, price=close,
                        reason=f"EMA-Deriv追踪止损 trail={trail:.2f}",
                    ))
                    self._direction[symbol] = 0

        for s in signals:
            self.record_signal(s)
        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        all_sigs: list[Signal] = []
        for symbol in self.config.symbols:
            bar = market_data.get(symbol)
            if bar:
                all_sigs.extend(await self.on_bar(symbol, bar))
        return all_sigs
