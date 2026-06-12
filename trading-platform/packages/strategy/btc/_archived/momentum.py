"""BTC 动量策略 - 基于价格动量和成交量确认的趋势跟踪策略。"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..indicators import calc_atr
from ..registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "fast_period": 4,
    "slow_period": 23,
    "volume_ma_period": 20,
    "momentum_threshold": 0.046,
    "volume_surge_ratio": 1.5,
    "atr_period": 14,
    "trailing_stop_atr_mult": 3.44,
    "take_profit_atr_mult": 3.57,
    "max_hold_bars": 73,
}


@auto_register("btc_momentum")
class BTCMomentumStrategy(BaseStrategy):
    """BTC 动量趋势跟踪策略。

    核心逻辑：
    - 快慢均线交叉产生方向信号
    - 成交量放大确认信号有效性
    - ATR 动态止损跟踪
    """

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._close_history: dict[str, deque[float]] = {}
        self._volume_history: dict[str, deque[float]] = {}
        self._high_history: dict[str, deque[float]] = {}
        self._low_history: dict[str, deque[float]] = {}
        self._bars_in_pos: dict[str, int] = {}
        self._trailing_high: dict[str, float] = {}
        self._trailing_low: dict[str, float] = {}
        self._prev_momentum: dict[str, float] = {}

    def _ensure_buffers(self, symbol: str) -> None:
        max_len = max(
            self.get_param("slow_period"),
            self.get_param("volume_ma_period"),
            self.get_param("atr_period"),
        ) + 10
        if symbol not in self._close_history:
            self._close_history[symbol] = deque(maxlen=max_len)
            self._volume_history[symbol] = deque(maxlen=max_len)
            self._high_history[symbol] = deque(maxlen=max_len)
            self._low_history[symbol] = deque(maxlen=max_len)

    @staticmethod
    def _sma(data: deque[float], period: int) -> float | None:
        if len(data) < period:
            return None
        return sum(list(data)[-period:]) / period

    def _calc_atr(self, symbol: str) -> float | None:
        return calc_atr(
            self._high_history[symbol],
            self._low_history[symbol],
            self._close_history[symbol],
            self.get_param("atr_period"),
        )

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._ensure_buffers(symbol)
        self._close_history[symbol].append(bar["close"])
        self._volume_history[symbol].append(bar.get("volume", 0))
        self._high_history[symbol].append(bar["high"])
        self._low_history[symbol].append(bar["low"])

        fast_ma = self._sma(self._close_history[symbol], self.get_param("fast_period"))
        slow_ma = self._sma(self._close_history[symbol], self.get_param("slow_period"))
        vol_ma = self._sma(self._volume_history[symbol], self.get_param("volume_ma_period"))

        if fast_ma is None or slow_ma is None or vol_ma is None:
            return []

        current_price = bar["close"]
        current_vol = bar.get("volume", 0)
        momentum = (fast_ma - slow_ma) / slow_ma if slow_ma != 0 else 0
        threshold = self.get_param("momentum_threshold")
        vol_surge = self.get_param("volume_surge_ratio")
        volume_confirmed = current_vol > vol_ma * vol_surge if vol_ma > 0 else False

        prev_mom = self._prev_momentum.get(symbol, 0.0)
        self._prev_momentum[symbol] = momentum

        signals: list[Signal] = []
        pos = self.get_position(symbol)

        crossed_up = prev_mom <= threshold and momentum > threshold
        crossed_down = prev_mom >= -threshold and momentum < -threshold

        if pos is None and crossed_up and volume_confirmed:
            strength = min(abs(momentum) / (threshold * 3), 1.0)
            sig = Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_type=SignalType.LONG_ENTRY,
                strength=round(strength, 4),
                price=current_price,
                reason=f"动量突破(mom={momentum:.4f})+量能确认(vol_ratio={current_vol/vol_ma:.2f})",
                metadata={"fast_ma": fast_ma, "slow_ma": slow_ma, "momentum": momentum},
            )
            signals.append(sig)
            self.record_signal(sig)

        elif pos is None and crossed_down and volume_confirmed:
            strength = min(abs(momentum) / (threshold * 3), 1.0)
            sig = Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_type=SignalType.SHORT_ENTRY,
                strength=round(strength, 4),
                price=current_price,
                reason=f"动量下破(mom={momentum:.4f})+量能确认(vol_ratio={current_vol/vol_ma:.2f})",
                metadata={"fast_ma": fast_ma, "slow_ma": slow_ma, "momentum": momentum},
            )
            signals.append(sig)
            self.record_signal(sig)

        if pos is not None:
            self._bars_in_pos[symbol] = self._bars_in_pos.get(symbol, 0) + 1
            atr = self._calc_atr(symbol)

            if pos.side.value == "buy":
                self._trailing_high[symbol] = max(
                    self._trailing_high.get(symbol, current_price), current_price
                )
            elif pos.side.value == "sell":
                self._trailing_low[symbol] = min(
                    self._trailing_low.get(symbol, current_price), current_price
                )

            max_hold = self.get_param("max_hold_bars")
            if self._bars_in_pos.get(symbol, 0) >= max_hold:
                exit_type = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_type, strength=0.7, price=current_price,
                    reason=f"最大持仓时间到期({max_hold}bars)",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._bars_in_pos[symbol] = 0

            elif atr is not None and atr > 0:
                stop_mult = self.get_param("trailing_stop_atr_mult")
                tp_mult = self.get_param("take_profit_atr_mult")

                if pos.side.value == "buy":
                    trail_stop = self._trailing_high.get(symbol, pos.avg_price) - atr * stop_mult
                    take_profit = pos.avg_price + atr * tp_mult
                    if current_price < trail_stop:
                        sig = Signal(
                            strategy_id=self.strategy_id, symbol=symbol,
                            signal_type=SignalType.LONG_EXIT, strength=0.9,
                            price=current_price,
                            reason=f"追踪止损(trail={trail_stop:.2f})",
                        )
                        signals.append(sig)
                        self.record_signal(sig)
                        self._bars_in_pos[symbol] = 0
                    elif current_price >= take_profit:
                        sig = Signal(
                            strategy_id=self.strategy_id, symbol=symbol,
                            signal_type=SignalType.LONG_EXIT, strength=0.8,
                            price=current_price,
                            reason=f"止盈(tp={take_profit:.2f})",
                        )
                        signals.append(sig)
                        self.record_signal(sig)
                        self._bars_in_pos[symbol] = 0

                elif pos.side.value == "sell":
                    trail_stop = self._trailing_low.get(symbol, pos.avg_price) + atr * stop_mult
                    take_profit = pos.avg_price - atr * tp_mult
                    if current_price > trail_stop:
                        sig = Signal(
                            strategy_id=self.strategy_id, symbol=symbol,
                            signal_type=SignalType.SHORT_EXIT, strength=0.9,
                            price=current_price,
                            reason=f"追踪止损(trail={trail_stop:.2f})",
                        )
                        signals.append(sig)
                        self.record_signal(sig)
                        self._bars_in_pos[symbol] = 0
                    elif current_price <= take_profit:
                        sig = Signal(
                            strategy_id=self.strategy_id, symbol=symbol,
                            signal_type=SignalType.SHORT_EXIT, strength=0.8,
                            price=current_price,
                            reason=f"止盈(tp={take_profit:.2f})",
                        )
                        signals.append(sig)
                        self.record_signal(sig)
                        self._bars_in_pos[symbol] = 0
        else:
            self._bars_in_pos[symbol] = 0
            self._trailing_high.pop(symbol, None)
            self._trailing_low.pop(symbol, None)

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        all_signals: list[Signal] = []
        for symbol in self.config.symbols:
            bar = market_data.get(symbol)
            if bar:
                sigs = await self.on_bar(symbol, bar)
                all_signals.extend(sigs)
        return all_signals
