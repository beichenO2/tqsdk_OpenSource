"""动量轮动策略 — 横截面/时间序列动量的多品种轮动。

SOTA 要点 (Jegadeesh & Titman / Asness):
- 横截面动量 (Cross-Sectional Momentum): 做多过去表现最好的品种，做空最差的
- 时间序列动量 (TSMOM): 每个品种独立判断，过去涨则做多
- 动量因子叠加波动率调整 (Vol-Adjusted Momentum)
- Dual Momentum (绝对+相对动量) 降低回撤
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..indicators import calc_atr
from ..registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "lookback": 20,
    "holding_period": 5,
    "momentum_type": "tsmom",   # "tsmom" / "dual"
    "vol_adjust": True,
    "vol_window": 20,
    "absolute_threshold": 0.0,  # dual momentum: 绝对动量阈值
    "atr_period": 14,
    "trailing_stop_atr_mult": 2.5,
    "max_hold_bars": 100,
}


@auto_register("momentum_rotation")
class MomentumRotationStrategy(BaseStrategy):
    """动量轮动策略。"""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._close_buf: dict[str, deque[float]] = {}
        self._high_buf: dict[str, deque[float]] = {}
        self._low_buf: dict[str, deque[float]] = {}
        self._bar_count: dict[str, int] = {}
        self._bars_in_pos: dict[str, int] = {}
        self._peak: dict[str, float] = {}
        self._trough: dict[str, float] = {}

    def _ensure(self, symbol: str) -> None:
        if symbol not in self._close_buf:
            ml = max(int(self.get_param("lookback")), int(self.get_param("vol_window"))) + 20
            self._close_buf[symbol] = deque(maxlen=ml)
            self._high_buf[symbol] = deque(maxlen=ml)
            self._low_buf[symbol] = deque(maxlen=ml)
            self._bar_count[symbol] = 0

    def _momentum(self, symbol: str) -> float | None:
        lookback = int(self.get_param("lookback"))
        buf = list(self._close_buf[symbol])
        if len(buf) < lookback + 1:
            return None
        if buf[-lookback - 1] == 0:
            return None
        raw = buf[-1] / buf[-lookback - 1] - 1

        if self.get_param("vol_adjust"):
            vol_w = int(self.get_param("vol_window"))
            if len(buf) < vol_w + 1:
                return raw
            rets = [math.log(buf[i] / buf[i - 1]) for i in range(-vol_w, 0) if buf[i - 1] > 0]
            if len(rets) < vol_w:
                return raw
            std = math.sqrt(sum(r**2 for r in rets) / len(rets))
            return raw / std if std > 0 else raw

        return raw

    def _calc_atr(self, symbol: str) -> float | None:
        return calc_atr(self._high_buf[symbol], self._low_buf[symbol], self._close_buf[symbol], int(self.get_param("atr_period")))

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._ensure(symbol)
        close = float(bar["close"])
        self._close_buf[symbol].append(close)
        self._high_buf[symbol].append(float(bar["high"]))
        self._low_buf[symbol].append(float(bar["low"]))
        self._bar_count[symbol] = self._bar_count.get(symbol, 0) + 1

        mom = self._momentum(symbol)
        atr = self._calc_atr(symbol)
        if mom is None or atr is None:
            return []

        signals: list[Signal] = []
        pos = self.get_position(symbol)
        mom_type = self.get_param("momentum_type")
        abs_thresh = float(self.get_param("absolute_threshold"))

        if pos is None:
            if mom_type == "dual":
                if mom > abs_thresh:
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.LONG_ENTRY, strength=min(abs(mom) * 2, 1.0),
                        price=close, reason=f"DualMom做多 mom={mom:.4f}",
                    ))
                    self._peak[symbol] = close
                elif mom < -abs_thresh:
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.SHORT_ENTRY, strength=min(abs(mom) * 2, 1.0),
                        price=close, reason=f"DualMom做空 mom={mom:.4f}",
                    ))
                    self._trough[symbol] = close
            else:  # tsmom
                if mom > 0:
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.LONG_ENTRY, strength=min(abs(mom) * 2, 1.0),
                        price=close, reason=f"TSMOM做多 mom={mom:.4f}",
                    ))
                    self._peak[symbol] = close
                elif mom < 0:
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.SHORT_ENTRY, strength=min(abs(mom) * 2, 1.0),
                        price=close, reason=f"TSMOM做空 mom={mom:.4f}",
                    ))
                    self._trough[symbol] = close

        elif pos is not None:
            self._bars_in_pos[symbol] = self._bars_in_pos.get(symbol, 0) + 1
            stop_mult = float(self.get_param("trailing_stop_atr_mult"))
            max_hold = int(self.get_param("max_hold_bars"))
            holding = int(self.get_param("holding_period"))

            rebal = self._bars_in_pos.get(symbol, 0) >= holding
            timeout = self._bars_in_pos.get(symbol, 0) >= max_hold

            if timeout or rebal:
                exit_t = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                reason = f"超时{max_hold}" if timeout else f"轮动周期{holding}"
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_t, strength=0.7, price=close, reason=reason,
                ))
                self._bars_in_pos[symbol] = 0
            elif pos.side.value == "buy":
                self._peak[symbol] = max(self._peak.get(symbol, close), close)
                trail = self._peak[symbol] - atr * stop_mult
                if close < trail:
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.LONG_EXIT, strength=0.85, price=close,
                        reason=f"MomRot平多 trail={trail:.2f}",
                    ))
                    self._bars_in_pos[symbol] = 0
            elif pos.side.value == "sell":
                self._trough[symbol] = min(self._trough.get(symbol, close), close)
                trail = self._trough[symbol] + atr * stop_mult
                if close > trail:
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.SHORT_EXIT, strength=0.85, price=close,
                        reason=f"MomRot平空 trail={trail:.2f}",
                    ))
                    self._bars_in_pos[symbol] = 0
        else:
            self._bars_in_pos[symbol] = 0

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
