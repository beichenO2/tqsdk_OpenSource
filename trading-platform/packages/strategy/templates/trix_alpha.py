"""Alpha TRIX 策略 — 三层验证趋势跟踪。

来源: TradingView Alpha TRIX (BVL-Crypto), 学术三重平滑 EMA
核心思路:
- TRIX = 三重指数移动平均的变化率 (极低噪声)
- 第1层: TRIX 零线交叉决定方向
- 第2层: Choppiness Index 过滤震荡市 (CHOP > 阈值 = 震荡, 不开仓)
- 第3层: ADX 确认趋势强度 (ADX > 阈值 = 有趋势)
- ATR 自适应止损/止盈
- 高暴露度: 一旦确认趋势就全仓持有
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
    "trix_period": 18,
    "trix_signal": 9,           # 信号线周期
    "chop_period": 14,
    "chop_threshold": 55,       # CHOP > 55 = 震荡市，不开仓
    "adx_period": 14,
    "adx_threshold": 20,        # ADX > 20 = 有趋势
    "atr_period": 14,
    "trailing_stop_atr_mult": 3.5,
    "take_profit_atr_mult": 0,  # 0 = 不止盈，让利润奔跑
}


@auto_register("trix_alpha")
class TrixAlphaStrategy(BaseStrategy):
    """Alpha TRIX 三层验证趋势策略。"""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._close_buf: dict[str, deque[float]] = {}
        self._high_buf: dict[str, deque[float]] = {}
        self._low_buf: dict[str, deque[float]] = {}
        # 三层 EMA 用于 TRIX 计算
        self._ema1: dict[str, float | None] = {}
        self._ema2: dict[str, float | None] = {}
        self._ema3: dict[str, float | None] = {}
        self._prev_trix: dict[str, float | None] = {}
        self._trix_signal_ema: dict[str, float | None] = {}
        self._peak: dict[str, float] = {}
        self._trough: dict[str, float] = {}

    def _ensure(self, symbol: str) -> None:
        if symbol not in self._close_buf:
            self._close_buf[symbol] = deque(maxlen=200)
            self._high_buf[symbol] = deque(maxlen=60)
            self._low_buf[symbol] = deque(maxlen=60)
            self._ema1[symbol] = None
            self._ema2[symbol] = None
            self._ema3[symbol] = None
            self._prev_trix[symbol] = None
            self._trix_signal_ema[symbol] = None

    def _update_ema(self, prev: float | None, value: float, period: int) -> float:
        k = 2 / (period + 1)
        if prev is None:
            return value
        return value * k + prev * (1 - k)

    def _calc_trix(self, symbol: str, close: float) -> float | None:
        """计算 TRIX 值: 三重 EMA 的百分比变化率。"""
        period = int(self.get_param("trix_period"))

        self._ema1[symbol] = self._update_ema(self._ema1[symbol], close, period)
        self._ema2[symbol] = self._update_ema(self._ema2[symbol], self._ema1[symbol], period)
        self._ema3[symbol] = self._update_ema(self._ema3[symbol], self._ema2[symbol], period)

        buf = list(self._close_buf[symbol])
        if len(buf) < period * 3:
            return None

        prev_ema3 = self._prev_trix.get(symbol)
        self._prev_trix[symbol] = self._ema3[symbol]

        if prev_ema3 is None or prev_ema3 == 0:
            return None

        trix = (self._ema3[symbol] - prev_ema3) / prev_ema3 * 100
        return trix

    def _calc_choppiness(self, symbol: str) -> float | None:
        """Choppiness Index: 衡量市场是否在震荡。"""
        period = int(self.get_param("chop_period"))
        h = list(self._high_buf[symbol])
        lo = list(self._low_buf[symbol])
        c = list(self._close_buf[symbol])
        if len(h) < period + 1:
            return None

        atr_sum = 0.0
        for i in range(-period, 0):
            tr = max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1]))
            atr_sum += tr

        highest = max(h[-period:])
        lowest = min(lo[-period:])
        price_range = highest - lowest
        if price_range <= 0:
            return 100.0

        log_period = math.log10(period) if period > 1 else 1.0
        chop = 100 * math.log10(atr_sum / price_range) / log_period
        return min(max(chop, 0), 100)

    def _calc_adx(self, symbol: str) -> float | None:
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

    def _calc_atr(self, symbol: str) -> float | None:
        return calc_atr(self._high_buf[symbol], self._low_buf[symbol], self._close_buf[symbol], int(self.get_param("atr_period")))

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._ensure(symbol)
        close = float(bar["close"])
        high = float(bar["high"])
        low = float(bar["low"])
        self._close_buf[symbol].append(close)
        self._high_buf[symbol].append(high)
        self._low_buf[symbol].append(low)

        trix = self._calc_trix(symbol, close)
        if trix is None:
            return []

        chop = self._calc_choppiness(symbol)
        adx = self._calc_adx(symbol)
        atr = self._calc_atr(symbol)
        if chop is None or adx is None or atr is None:
            return []

        chop_threshold = float(self.get_param("chop_threshold"))
        adx_threshold = float(self.get_param("adx_threshold"))

        # 三层验证
        is_trending = chop < chop_threshold and adx > adx_threshold
        trix_bullish = trix > 0
        trix_bearish = trix < 0

        signals: list[Signal] = []
        pos = self.get_position(symbol)

        if pos is None and is_trending:
            if trix_bullish:
                strength = min(abs(trix) * 10 + 0.5, 1.0)
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=round(strength, 4), price=close,
                    reason=f"TRIX做多 trix={trix:.4f} chop={chop:.1f} adx={adx:.1f}",
                    metadata={"trix": trix, "chop": chop, "adx": adx},
                ))
                self._peak[symbol] = close
            elif trix_bearish:
                strength = min(abs(trix) * 10 + 0.5, 1.0)
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=round(strength, 4), price=close,
                    reason=f"TRIX做空 trix={trix:.4f} chop={chop:.1f} adx={adx:.1f}",
                    metadata={"trix": trix, "chop": chop, "adx": adx},
                ))
                self._trough[symbol] = close

        elif pos is not None:
            stop_mult = float(self.get_param("trailing_stop_atr_mult"))

            # TRIX 方向反转 → 平仓
            if pos.side.value == "buy" and trix_bearish:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_EXIT, strength=0.8, price=close,
                    reason=f"TRIX翻空 trix={trix:.4f}",
                ))
            elif pos.side.value == "sell" and trix_bullish:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_EXIT, strength=0.8, price=close,
                    reason=f"TRIX翻多 trix={trix:.4f}",
                ))
            # 追踪止损
            elif pos.side.value == "buy":
                self._peak[symbol] = max(self._peak.get(symbol, close), close)
                trail = self._peak[symbol] - atr * stop_mult
                if close < trail:
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.LONG_EXIT, strength=0.9, price=close,
                        reason=f"TRIX追踪止损 trail={trail:.2f}",
                    ))
            elif pos.side.value == "sell":
                self._trough[symbol] = min(self._trough.get(symbol, close), close)
                trail = self._trough[symbol] + atr * stop_mult
                if close > trail:
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.SHORT_EXIT, strength=0.9, price=close,
                        reason=f"TRIX追踪止损 trail={trail:.2f}",
                    ))

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
