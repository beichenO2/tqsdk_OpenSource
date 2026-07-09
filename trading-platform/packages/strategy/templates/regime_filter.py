"""体制过滤策略 — 用市场体制检测 (Regime) 动态切换子策略。

SOTA 要点:
- 识别 3 种市场体制: 趋势(Trending) / 震荡(Ranging) / 高波动(Volatile)
- 趋势体制 → 使用动量/趋势跟踪信号
- 震荡体制 → 使用均值回归信号
- 高波动 → 缩减仓位或观望
- 判断方法: ADX + Bollinger BandWidth + 价格分形维度
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
    "regime_lookback": 30,
    "adx_period": 14,
    "adx_trend_threshold": 25,
    "bb_period": 20,
    "bb_std": 2.0,
    "bb_squeeze_pct": 0.03,     # BandWidth < 3% = 震荡
    "bb_expand_pct": 0.08,      # BandWidth > 8% = 高波动

    # 趋势子策略参数
    "trend_ma_fast": 10,
    "trend_ma_slow": 30,

    # 均值回归子策略参数
    "mr_rsi_period": 14,
    "mr_rsi_oversold": 25,
    "mr_rsi_overbought": 75,

    "atr_period": 14,
    "trailing_stop_atr_mult": 2.0,
    "max_hold_bars": 150,
}


@auto_register("regime_filter")
class RegimeFilterStrategy(BaseStrategy):
    """体制自适应策略 — 根据市场状态切换趋势/均值回归子信号。"""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._close_buf: dict[str, deque[float]] = {}
        self._high_buf: dict[str, deque[float]] = {}
        self._low_buf: dict[str, deque[float]] = {}
        self._bars_in_pos: dict[str, int] = {}
        self._peak: dict[str, float] = {}
        self._trough: dict[str, float] = {}
        self._current_regime: dict[str, str] = {}

    def _ensure(self, symbol: str) -> None:
        if symbol not in self._close_buf:
            max_len = 100
            self._close_buf[symbol] = deque(maxlen=max_len)
            self._high_buf[symbol] = deque(maxlen=max_len)
            self._low_buf[symbol] = deque(maxlen=max_len)

    def _sma(self, data: list[float], period: int) -> float | None:
        if len(data) < period:
            return None
        return sum(data[-period:]) / period

    def _detect_regime(self, symbol: str) -> str:
        """判断市场体制: 'trending' / 'ranging' / 'volatile'。"""
        bb_period = int(self.get_param("bb_period"))
        bb_std = float(self.get_param("bb_std"))
        buf = list(self._close_buf[symbol])
        if len(buf) < bb_period:
            return "unknown"

        window = buf[-bb_period:]
        sma = sum(window) / len(window)
        std = math.sqrt(sum((x - sma) ** 2 for x in window) / len(window))
        if sma == 0:
            return "unknown"
        bb_width = (2 * bb_std * std) / sma

        squeeze_pct = float(self.get_param("bb_squeeze_pct"))
        expand_pct = float(self.get_param("bb_expand_pct"))

        if bb_width > expand_pct:
            return "volatile"
        elif bb_width < squeeze_pct:
            return "ranging"
        else:
            # ADX 判断是否趋势
            adx = self._adx(symbol)
            if adx is not None and adx > float(self.get_param("adx_trend_threshold")):
                return "trending"
            return "ranging"

    def _adx(self, symbol: str) -> float | None:
        period = int(self.get_param("adx_period"))
        h = list(self._high_buf[symbol])
        lo = list(self._low_buf[symbol])
        c = list(self._close_buf[symbol])
        if len(h) < period + 2:
            return None
        plus_dm_sum, minus_dm_sum, tr_sum = 0.0, 0.0, 0.0
        for i in range(-period, 0):
            up = h[i] - h[i - 1]
            down = lo[i - 1] - lo[i]
            plus_dm_sum += max(up, 0) if up > down else 0
            minus_dm_sum += max(down, 0) if down > up else 0
            tr = max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1]))
            tr_sum += tr
        if tr_sum == 0:
            return 0.0
        di_p = plus_dm_sum / tr_sum * 100
        di_m = minus_dm_sum / tr_sum * 100
        di_sum = di_p + di_m
        return abs(di_p - di_m) / di_sum * 100 if di_sum > 0 else 0.0

    def _trend_signal(self, symbol: str, close: float) -> int:
        """趋势子信号: 快慢均线交叉方向。"""
        buf = list(self._close_buf[symbol])
        fast = self._sma(buf, int(self.get_param("trend_ma_fast")))
        slow = self._sma(buf, int(self.get_param("trend_ma_slow")))
        if fast is None or slow is None:
            return 0
        if fast > slow:
            return 1
        elif fast < slow:
            return -1
        return 0

    def _mr_signal(self, symbol: str, close: float) -> int:
        """均值回归子信号: RSI 超买超卖。"""
        period = int(self.get_param("mr_rsi_period"))
        buf = list(self._close_buf[symbol])
        if len(buf) < period + 1:
            return 0
        gains, losses = 0.0, 0.0
        for i in range(-period, 0):
            d = buf[i] - buf[i - 1]
            if d > 0:
                gains += d
            else:
                losses -= d
        avg_g = gains / period
        avg_l = losses / period
        rsi = 100 - 100 / (1 + avg_g / avg_l) if avg_l > 0 else 100.0

        if rsi < float(self.get_param("mr_rsi_oversold")):
            return 1
        elif rsi > float(self.get_param("mr_rsi_overbought")):
            return -1
        return 0

    def _calc_atr(self, symbol: str) -> float | None:
        return calc_atr(self._high_buf[symbol], self._low_buf[symbol], self._close_buf[symbol], int(self.get_param("atr_period")))

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._ensure(symbol)
        close = float(bar["close"])
        self._close_buf[symbol].append(close)
        self._high_buf[symbol].append(float(bar["high"]))
        self._low_buf[symbol].append(float(bar["low"]))

        regime = self._detect_regime(symbol)
        self._current_regime[symbol] = regime
        atr = self._calc_atr(symbol)
        if atr is None:
            return []

        signals: list[Signal] = []
        pos = self.get_position(symbol)

        if regime == "volatile":
            # 高波动：如有持仓则平仓，不开新仓
            if pos is not None:
                exit_t = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_t, strength=0.7, price=close,
                    reason=f"高波动体制平仓 regime={regime}",
                ))
                self._bars_in_pos[symbol] = 0
        elif pos is None:
            if regime == "trending":
                direction = self._trend_signal(symbol, close)
            else:
                direction = self._mr_signal(symbol, close)

            if direction > 0:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY, strength=0.7, price=close,
                    reason=f"Regime({regime})做多",
                ))
                self._peak[symbol] = close
            elif direction < 0:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY, strength=0.7, price=close,
                    reason=f"Regime({regime})做空",
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
                if close < trail:
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.LONG_EXIT, strength=0.85, price=close,
                        reason=f"Regime平多 trail={trail:.2f}",
                    ))
                    self._bars_in_pos[symbol] = 0
            elif pos.side.value == "sell":
                self._trough[symbol] = min(self._trough.get(symbol, close), close)
                trail = self._trough[symbol] + atr * stop_mult
                if close > trail:
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.SHORT_EXIT, strength=0.85, price=close,
                        reason=f"Regime平空 trail={trail:.2f}",
                    ))
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
