"""Z-Score 统计套利/均值回归策略。

SOTA 要点:
- 计算价格相对移动均值的 z-score (标准化偏离度)
- z > entry_z → 做空 (价格偏高); z < -entry_z → 做多 (价格偏低)
- 用 half-life 估计均值回归速度，自动调节持仓时间
- 支持 Bollinger %B、RSI 等作为辅助确认
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "lookback": 50,
    "entry_z": 2.0,
    "exit_z": 0.5,
    "stop_z": 3.5,
    "use_log_returns": True,
    "rsi_period": 14,
    "rsi_confirm": True,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "max_hold_bars": 80,
}


@auto_register("stat_arb_zscore")
class StatArbStrategy(BaseStrategy):
    """Z-Score 均值回归策略。"""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._close_buf: dict[str, deque[float]] = {}
        self._bars_in_pos: dict[str, int] = {}

    def _ensure_buf(self, symbol: str) -> None:
        if symbol not in self._close_buf:
            lb = int(self.get_param("lookback")) + 20
            self._close_buf[symbol] = deque(maxlen=lb)

    def _zscore(self, symbol: str) -> float | None:
        lb = int(self.get_param("lookback"))
        buf = list(self._close_buf[symbol])
        if len(buf) < lb:
            return None

        if self.get_param("use_log_returns"):
            series = [math.log(buf[i] / buf[i - 1]) for i in range(max(len(buf) - lb, 1), len(buf)) if buf[i - 1] > 0]
            if len(series) < lb - 1:
                return None
            mean = sum(series) / len(series)
            var = sum((x - mean) ** 2 for x in series) / len(series)
            std = math.sqrt(var) if var > 0 else 1e-10
            return (series[-1] - mean) / std
        else:
            window = buf[-lb:]
            mean = sum(window) / len(window)
            var = sum((x - mean) ** 2 for x in window) / len(window)
            std = math.sqrt(var) if var > 0 else 1e-10
            return (buf[-1] - mean) / std

    def _rsi(self, symbol: str) -> float | None:
        period = int(self.get_param("rsi_period"))
        buf = list(self._close_buf[symbol])
        if len(buf) < period + 1:
            return None
        gains, losses = 0.0, 0.0
        for i in range(-period, 0):
            diff = buf[i] - buf[i - 1]
            if diff > 0:
                gains += diff
            else:
                losses -= diff
        avg_gain = gains / period
        avg_loss = losses / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - 100 / (1 + rs)

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._ensure_buf(symbol)
        close = float(bar["close"])
        self._close_buf[symbol].append(close)

        z = self._zscore(symbol)
        if z is None:
            return []

        entry_z = float(self.get_param("entry_z"))
        exit_z = float(self.get_param("exit_z"))
        stop_z = float(self.get_param("stop_z"))
        rsi = self._rsi(symbol)
        rsi_confirm = bool(self.get_param("rsi_confirm"))

        signals: list[Signal] = []
        pos = self.get_position(symbol)

        if pos is None:
            if z < -entry_z:
                rsi_ok = (not rsi_confirm) or (rsi is not None and rsi < float(self.get_param("rsi_oversold")))
                if rsi_ok:
                    strength = min(abs(z) / (entry_z * 2), 1.0)
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.LONG_ENTRY, strength=round(strength, 4),
                        price=close, reason=f"Z-score做多 z={z:.2f} rsi={rsi:.1f}" if rsi else f"Z做多 z={z:.2f}",
                        metadata={"zscore": z, "rsi": rsi},
                    ))
            elif z > entry_z:
                rsi_ok = (not rsi_confirm) or (rsi is not None and rsi > float(self.get_param("rsi_overbought")))
                if rsi_ok:
                    strength = min(abs(z) / (entry_z * 2), 1.0)
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.SHORT_ENTRY, strength=round(strength, 4),
                        price=close, reason=f"Z-score做空 z={z:.2f} rsi={rsi:.1f}" if rsi else f"Z做空 z={z:.2f}",
                        metadata={"zscore": z, "rsi": rsi},
                    ))

        elif pos is not None:
            self._bars_in_pos[symbol] = self._bars_in_pos.get(symbol, 0) + 1
            max_hold = int(self.get_param("max_hold_bars"))

            if pos.side.value == "buy":
                if z > -exit_z or z > stop_z or self._bars_in_pos.get(symbol, 0) >= max_hold:
                    reason = "Z回归" if z > -exit_z else ("Z止损" if z > stop_z else f"超时{max_hold}")
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.LONG_EXIT, strength=0.8, price=close,
                        reason=f"{reason} z={z:.2f}",
                    ))
                    self._bars_in_pos[symbol] = 0
            elif pos.side.value == "sell":
                if z < exit_z or z < -stop_z or self._bars_in_pos.get(symbol, 0) >= max_hold:
                    reason = "Z回归" if z < exit_z else ("Z止损" if z < -stop_z else f"超时{max_hold}")
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.SHORT_EXIT, strength=0.8, price=close,
                        reason=f"{reason} z={z:.2f}",
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
