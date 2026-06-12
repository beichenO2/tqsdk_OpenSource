"""国内期货双均线动量策略 — 快慢均线交叉 + 量能确认 + ATR 追踪止损。"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..indicators import calc_atr
from ..registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "fast_period": 5,
    "slow_period": 20,
    "volume_ma_period": 20,
    "momentum_threshold": 0.0,
    "volume_surge_ratio": 1.2,
    "atr_period": 14,
    "trailing_stop_atr_mult": 2.0,
}


@auto_register("futures_dual_ma")
class FuturesDualMAStrategy(BaseStrategy):
    """双均线交叉策略（适合 1/5 分钟 K 线）。

    - 金叉 / 死叉 产生方向；可选动量阈值过滤（默认 0 表示仅看交叉）
    - 成交量需大于均量 × surge 倍率
    - 持仓期间以 ATR 峰值/谷值追踪止损
    """

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._close_history: dict[str, deque[float]] = {}
        self._volume_history: dict[str, deque[float]] = {}
        self._high_history: dict[str, deque[float]] = {}
        self._low_history: dict[str, deque[float]] = {}
        self._prev_fast_ma: dict[str, float | None] = {}
        self._prev_slow_ma: dict[str, float | None] = {}
        self._peak_price: dict[str, float] = {}
        self._trough_price: dict[str, float] = {}

    def _ensure_buffers(self, symbol: str) -> None:
        max_len = max(
            int(self.get_param("slow_period")),
            int(self.get_param("volume_ma_period")),
            int(self.get_param("atr_period")),
        ) + 15
        if symbol not in self._close_history:
            self._close_history[symbol] = deque(maxlen=max_len)
            self._volume_history[symbol] = deque(maxlen=max_len)
            self._high_history[symbol] = deque(maxlen=max_len)
            self._low_history[symbol] = deque(maxlen=max_len)
            self._prev_fast_ma[symbol] = None
            self._prev_slow_ma[symbol] = None

    @staticmethod
    def _sma(data: deque[float], period: int) -> float | None:
        if period <= 0 or len(data) < period:
            return None
        return sum(list(data)[-period:]) / period

    def _calc_atr(self, symbol: str) -> float | None:
        return calc_atr(
            self._high_history[symbol],
            self._low_history[symbol],
            self._close_history[symbol],
            int(self.get_param("atr_period")),
        )

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        close = bar.get("close")
        high = bar.get("high")
        low = bar.get("low")
        if close is None or high is None or low is None:
            logger.debug("跳过不完整 K 线: %s keys=%s", symbol, list(bar.keys()))
            return []

        self._ensure_buffers(symbol)
        self._close_history[symbol].append(float(close))
        self._volume_history[symbol].append(float(bar.get("volume") or 0.0))
        self._high_history[symbol].append(float(high))
        self._low_history[symbol].append(float(low))

        fast_p = int(self.get_param("fast_period"))
        slow_p = int(self.get_param("slow_period"))
        vol_ma_p = int(self.get_param("volume_ma_period"))

        fast_ma = self._sma(self._close_history[symbol], fast_p)
        slow_ma = self._sma(self._close_history[symbol], slow_p)
        vol_ma = self._sma(self._volume_history[symbol], vol_ma_p)

        if fast_ma is None or slow_ma is None or vol_ma is None:
            return []

        prev_f = self._prev_fast_ma[symbol]
        prev_s = self._prev_slow_ma[symbol]

        current_price = float(close)
        current_vol = float(bar.get("volume") or 0.0)
        mom_thr = float(self.get_param("momentum_threshold") or 0.0)
        vol_surge = float(self.get_param("volume_surge_ratio") or 1.0)
        if vol_surge <= 0:
            vol_surge = 1.0

        volume_ok = (vol_ma > 0 and current_vol > vol_ma * vol_surge) if vol_ma > 0 else False

        momentum = (fast_ma - slow_ma) / slow_ma if slow_ma != 0 else 0.0
        mom_ok_long = momentum > mom_thr
        mom_ok_short = momentum < -mom_thr

        golden = (
            prev_f is not None
            and prev_s is not None
            and prev_f <= prev_s
            and fast_ma > slow_ma
        )
        death = (
            prev_f is not None
            and prev_s is not None
            and prev_f >= prev_s
            and fast_ma < slow_ma
        )

        self._prev_fast_ma[symbol] = fast_ma
        self._prev_slow_ma[symbol] = slow_ma

        signals: list[Signal] = []
        pos = self.get_position(symbol)

        if pos is None:
            if golden and volume_ok and mom_ok_long:
                strength = min(abs(momentum) / max(mom_thr or 0.001, 0.001) * 0.3 + 0.5, 1.0)
                sig = Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=round(strength, 4),
                    price=current_price,
                    reason=(
                        f"金叉+量能(fast={fast_ma:.4f},slow={slow_ma:.4f},"
                        f"vol_ratio={current_vol/vol_ma:.2f})"
                    ),
                    metadata={"fast_ma": fast_ma, "slow_ma": slow_ma, "momentum": momentum},
                )
                signals.append(sig)
                self.record_signal(sig)
                self._peak_price[symbol] = current_price
                logger.info("[%s] 做多信号 %s", symbol, sig.reason)

            elif death and volume_ok and mom_ok_short:
                strength = min(abs(momentum) / max(mom_thr or 0.001, 0.001) * 0.3 + 0.5, 1.0)
                sig = Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=round(strength, 4),
                    price=current_price,
                    reason=(
                        f"死叉+量能(fast={fast_ma:.4f},slow={slow_ma:.4f},"
                        f"vol_ratio={current_vol/vol_ma:.2f})"
                    ),
                    metadata={"fast_ma": fast_ma, "slow_ma": slow_ma, "momentum": momentum},
                )
                signals.append(sig)
                self.record_signal(sig)
                self._trough_price[symbol] = current_price
                logger.info("[%s] 做空信号 %s", symbol, sig.reason)

        atr = self._calc_atr(symbol)
        if pos is not None and atr is not None and atr > 0:
            stop_mult = float(self.get_param("trailing_stop_atr_mult") or 2.0)

            if pos.side.value == "buy":
                self._peak_price[symbol] = max(self._peak_price.get(symbol, current_price), current_price)
                trail = self._peak_price[symbol] - atr * stop_mult
                if current_price < trail:
                    sig = Signal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=SignalType.LONG_EXIT,
                        strength=0.9,
                        price=current_price,
                        reason=f"ATR 追踪止损(止损线={trail:.4f}, peak={self._peak_price[symbol]:.4f})",
                        metadata={"trailing_stop": trail, "atr": atr},
                    )
                    signals.append(sig)
                    self.record_signal(sig)
                    self._peak_price.pop(symbol, None)
                    logger.info("[%s] 平多 %s", symbol, sig.reason)

            elif pos.side.value == "sell":
                self._trough_price[symbol] = min(
                    self._trough_price.get(symbol, current_price), current_price
                )
                trail = self._trough_price[symbol] + atr * stop_mult
                if current_price > trail:
                    sig = Signal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=SignalType.SHORT_EXIT,
                        strength=0.9,
                        price=current_price,
                        reason=f"ATR 追踪止损(止损线={trail:.4f}, trough={self._trough_price[symbol]:.4f})",
                        metadata={"trailing_stop": trail, "atr": atr},
                    )
                    signals.append(sig)
                    self.record_signal(sig)
                    self._trough_price.pop(symbol, None)
                    logger.info("[%s] 平空 %s", symbol, sig.reason)

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        all_signals: list[Signal] = []
        for symbol in self.config.symbols:
            bar = market_data.get(symbol)
            if bar:
                sigs = await self.on_bar(symbol, bar)
                all_signals.extend(sigs)
        return all_signals
