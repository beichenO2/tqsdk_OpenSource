"""BTC 均值回归策略 - 基于布林带和 RSI 的反转策略。"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any

from strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig
from strategy.registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "bb_period": 20,
    "bb_std_dev": 2.0,
    "rsi_period": 14,
    "rsi_oversold": 25,
    "rsi_overbought": 75,
    "min_signal_strength": 0.3,
    "max_position_pct": 0.1,
    "trend_ma_period": 50,
    "trend_filter_enabled": True,
    "stop_loss_bb_mult": 0.8,
    "long_only": True,
}


@auto_register("btc_mean_reversion")
class BTCMeanReversionStrategy(BaseStrategy):
    """BTC 均值回归策略。

    核心逻辑：
    - 布林带判断价格偏离程度
    - RSI 确认超买超卖状态
    - 两者同时满足时发出反转信号
    """

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)
        self._close_history: dict[str, deque[float]] = {}
        self._entry_bb_width: dict[str, float] = {}

    def _ensure_buffers(self, symbol: str) -> None:
        max_len = max(
            self.get_param("bb_period"),
            self.get_param("rsi_period"),
            self.get_param("trend_ma_period"),
        ) + 10
        if symbol not in self._close_history:
            self._close_history[symbol] = deque(maxlen=max_len)

    def _calc_bollinger(self, symbol: str) -> tuple[float, float, float] | None:
        period = self.get_param("bb_period")
        data = list(self._close_history[symbol])
        if len(data) < period:
            return None
        window = data[-period:]
        mid = sum(window) / period
        variance = sum((x - mid) ** 2 for x in window) / period
        std = math.sqrt(variance)
        mult = self.get_param("bb_std_dev")
        return mid, mid + std * mult, mid - std * mult

    def _calc_rsi(self, symbol: str) -> float | None:
        period = self.get_param("rsi_period")
        data = list(self._close_history[symbol])
        if len(data) < period + 1:
            return None
        gains: list[float] = []
        losses: list[float] = []
        for i in range(-period, 0):
            delta = data[i] - data[i - 1]
            if delta > 0:
                gains.append(delta)
                losses.append(0.0)
            else:
                gains.append(0.0)
                losses.append(abs(delta))
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _is_trending(self, symbol: str) -> bool:
        """Trend filter: skip MR signals when price is far from long MA."""
        if not self.get_param("trend_filter_enabled"):
            return False
        ma_period = self.get_param("trend_ma_period")
        data = list(self._close_history[symbol])
        if len(data) < ma_period:
            return False
        ma = sum(data[-ma_period:]) / ma_period
        deviation = abs(data[-1] - ma) / ma
        return deviation > 0.05

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._ensure_buffers(symbol)
        self._close_history[symbol].append(bar["close"])

        bb = self._calc_bollinger(symbol)
        rsi = self._calc_rsi(symbol)
        if bb is None or rsi is None:
            return []

        mid, upper, lower = bb
        current_price = bar["close"]
        signals: list[Signal] = []

        bb_width = upper - lower
        if bb_width <= 0:
            return []

        in_trend = self._is_trending(symbol)
        pos = self.get_position(symbol)

        if pos is None and not in_trend and current_price <= lower and rsi <= self.get_param("rsi_oversold"):
            distance_ratio = (lower - current_price) / bb_width
            rsi_extreme = (self.get_param("rsi_oversold") - rsi) / self.get_param("rsi_oversold")
            strength = min((distance_ratio + rsi_extreme) / 2 + 0.3, 1.0)
            if strength >= self.get_param("min_signal_strength"):
                sig = Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=round(strength, 4),
                    price=current_price,
                    reason=f"BB下轨突破(price={current_price:.2f}<lower={lower:.2f})+RSI超卖({rsi:.1f})",
                    metadata={"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "rsi": rsi},
                )
                signals.append(sig)
                self.record_signal(sig)
                self._entry_bb_width[symbol] = bb_width

        elif pos is None and not in_trend and not self.get_param("long_only") and current_price >= upper and rsi >= self.get_param("rsi_overbought"):
            distance_ratio = (current_price - upper) / bb_width
            rsi_extreme = (rsi - self.get_param("rsi_overbought")) / (100 - self.get_param("rsi_overbought"))
            strength = min((distance_ratio + rsi_extreme) / 2 + 0.3, 1.0)
            if strength >= self.get_param("min_signal_strength"):
                sig = Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=round(strength, 4),
                    price=current_price,
                    reason=f"BB上轨突破(price={current_price:.2f}>upper={upper:.2f})+RSI超买({rsi:.1f})",
                    metadata={"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "rsi": rsi},
                )
                signals.append(sig)
                self.record_signal(sig)
                self._entry_bb_width[symbol] = bb_width

        if pos is not None:
            stop_mult = self.get_param("stop_loss_bb_mult")
            entry_bw = self._entry_bb_width.get(symbol, bb_width)

            if pos.side.value == "buy":
                stop_price = pos.avg_price - entry_bw * stop_mult
                if current_price >= mid:
                    sig = Signal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=SignalType.LONG_EXIT,
                        strength=0.6,
                        price=current_price,
                        reason=f"MR止盈(price={current_price:.2f}>=mid={mid:.2f})",
                    )
                    signals.append(sig)
                    self.record_signal(sig)
                elif current_price < stop_price:
                    sig = Signal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=SignalType.LONG_EXIT,
                        strength=0.9,
                        price=current_price,
                        reason=f"MR止损(price={current_price:.2f}<stop={stop_price:.2f})",
                    )
                    signals.append(sig)
                    self.record_signal(sig)

            elif pos.side.value == "sell":
                stop_price = pos.avg_price + entry_bw * stop_mult
                if current_price <= mid:
                    sig = Signal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=SignalType.SHORT_EXIT,
                        strength=0.6,
                        price=current_price,
                        reason=f"MR止盈(price={current_price:.2f}<=mid={mid:.2f})",
                    )
                    signals.append(sig)
                    self.record_signal(sig)
                elif current_price > stop_price:
                    sig = Signal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=SignalType.SHORT_EXIT,
                        strength=0.9,
                        price=current_price,
                        reason=f"MR止损(price={current_price:.2f}>stop={stop_price:.2f})",
                    )
                    signals.append(sig)
                    self.record_signal(sig)

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        all_signals: list[Signal] = []
        for symbol in self.config.symbols:
            bar = market_data.get(symbol)
            if bar:
                sigs = await self.on_bar(symbol, bar)
                all_signals.extend(sigs)
        return all_signals
