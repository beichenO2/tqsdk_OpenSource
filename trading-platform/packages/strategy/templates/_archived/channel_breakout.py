"""通道突破策略 — Donchian / Keltner / 自适应通道。

SOTA 要点:
- Donchian: N 周期最高/最低价通道 (海龟交易系统)
- Keltner: EMA ± ATR*mult 的自适应通道
- 支持通过参数切换通道类型、突破确认、过滤条件
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
    "channel_type": "donchian",  # "donchian" / "keltner"
    "entry_period": 20,
    "exit_period": 10,
    "keltner_mult": 2.0,
    "atr_period": 14,
    "volume_confirm": True,
    "volume_ma_period": 20,
    "volume_ratio": 1.2,
    "trailing_stop_atr_mult": 2.5,
    "max_hold_bars": 200,
}


@auto_register("channel_breakout")
class ChannelBreakoutStrategy(BaseStrategy):
    """通用通道突破策略。"""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._close_buf: dict[str, deque[float]] = {}
        self._high_buf: dict[str, deque[float]] = {}
        self._low_buf: dict[str, deque[float]] = {}
        self._vol_buf: dict[str, deque[float]] = {}
        self._peak: dict[str, float] = {}
        self._trough: dict[str, float] = {}
        self._bars_in_pos: dict[str, int] = {}

    def _ensure(self, symbol: str) -> None:
        if symbol not in self._close_buf:
            max_len = max(
                int(self.get_param("entry_period")),
                int(self.get_param("atr_period")),
                int(self.get_param("volume_ma_period")),
            ) * 3 + 20
            self._close_buf[symbol] = deque(maxlen=max_len)
            self._high_buf[symbol] = deque(maxlen=max_len)
            self._low_buf[symbol] = deque(maxlen=max_len)
            self._vol_buf[symbol] = deque(maxlen=max_len)

    def _donchian_upper(self, symbol: str, n: int) -> float | None:
        h = list(self._high_buf[symbol])
        if len(h) <= n:
            return None
        return max(h[-(n + 1):-1])

    def _donchian_lower(self, symbol: str, n: int) -> float | None:
        lo = list(self._low_buf[symbol])
        if len(lo) <= n:
            return None
        return min(lo[-(n + 1):-1])

    def _calc_atr(self, symbol: str) -> float | None:
        return calc_atr(self._high_buf[symbol], self._low_buf[symbol], self._close_buf[symbol], int(self.get_param("atr_period")))

    def _ema(self, data: list[float], period: int) -> float | None:
        if len(data) < period:
            return None
        k = 2 / (period + 1)
        val = sum(data[:period]) / period
        for x in data[period:]:
            val = x * k + val * (1 - k)
        return val

    def _keltner(self, symbol: str) -> tuple[float, float] | None:
        period = int(self.get_param("entry_period"))
        mult = float(self.get_param("keltner_mult"))
        atr = self._calc_atr(symbol)
        ema = self._ema(list(self._close_buf[symbol]), period)
        if atr is None or ema is None:
            return None
        return (ema + mult * atr, ema - mult * atr)

    def _vol_confirmed(self, symbol: str) -> bool:
        if not self.get_param("volume_confirm"):
            return True
        vol_ma_p = int(self.get_param("volume_ma_period"))
        buf = list(self._vol_buf[symbol])
        if len(buf) < vol_ma_p:
            return False
        vol_ma = sum(buf[-vol_ma_p:]) / vol_ma_p
        ratio = float(self.get_param("volume_ratio"))
        return buf[-1] > vol_ma * ratio if vol_ma > 0 else False

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._ensure(symbol)
        close = float(bar["close"])
        high = float(bar["high"])
        low = float(bar["low"])
        volume = float(bar.get("volume", 0))
        self._close_buf[symbol].append(close)
        self._high_buf[symbol].append(high)
        self._low_buf[symbol].append(low)
        self._vol_buf[symbol].append(volume)

        ch_type = self.get_param("channel_type")
        entry_n = int(self.get_param("entry_period"))
        exit_n = int(self.get_param("exit_period"))
        atr = self._calc_atr(symbol)

        if ch_type == "keltner":
            bands = self._keltner(symbol)
            if bands is None or atr is None:
                return []
            upper, lower = bands
            exit_upper, exit_lower = upper, lower
        else:
            upper = self._donchian_upper(symbol, entry_n)
            lower = self._donchian_lower(symbol, entry_n)
            exit_upper = self._donchian_upper(symbol, exit_n)
            exit_lower = self._donchian_lower(symbol, exit_n)
            if upper is None or lower is None or atr is None:
                return []

        vol_ok = self._vol_confirmed(symbol)
        signals: list[Signal] = []
        pos = self.get_position(symbol)

        if pos is None and vol_ok:
            if close > upper:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY, strength=0.75, price=close,
                    reason=f"{ch_type}上破 upper={upper:.2f}",
                ))
                self._peak[symbol] = close
            elif close < lower:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY, strength=0.75, price=close,
                    reason=f"{ch_type}下破 lower={lower:.2f}",
                ))
                self._trough[symbol] = close

        elif pos is not None:
            self._bars_in_pos[symbol] = self._bars_in_pos.get(symbol, 0) + 1
            stop_mult = float(self.get_param("trailing_stop_atr_mult"))
            max_hold = int(self.get_param("max_hold_bars"))

            if self._bars_in_pos.get(symbol, 0) >= max_hold:
                exit_t = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_t, strength=0.6, price=close, reason=f"超时{max_hold}",
                ))
                self._bars_in_pos[symbol] = 0
            elif pos.side.value == "buy":
                self._peak[symbol] = max(self._peak.get(symbol, close), close)
                trail = self._peak[symbol] - atr * stop_mult
                channel_exit = exit_lower is not None and close < exit_lower
                if close < trail or channel_exit:
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.LONG_EXIT, strength=0.85, price=close,
                        reason=f"Channel平多 trail={trail:.2f}",
                    ))
                    self._bars_in_pos[symbol] = 0
            elif pos.side.value == "sell":
                self._trough[symbol] = min(self._trough.get(symbol, close), close)
                trail = self._trough[symbol] + atr * stop_mult
                channel_exit = exit_upper is not None and close > exit_upper
                if close > trail or channel_exit:
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.SHORT_EXIT, strength=0.85, price=close,
                        reason=f"Channel平空 trail={trail:.2f}",
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
