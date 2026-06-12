"""多指标组合策略 — 通过参数化组合不同技术指标产生信号。

SOTA 要点:
- 将多个指标 (RSI/MACD/Stoch/Williams%R/CCI/MFI) 的信号加权组合
- 用 voting 或 weighted scoring 决定最终方向
- 单一指标噪声高，多指标共振大幅提升胜率
- 支持通过参数选择激活哪些指标
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
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "rsi_weight": 1.0,

    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "macd_weight": 1.0,

    "stoch_period": 14,
    "stoch_smooth": 3,
    "stoch_oversold": 20,
    "stoch_overbought": 80,
    "stoch_weight": 1.0,

    "cci_period": 20,
    "cci_threshold": 100,
    "cci_weight": 1.0,

    "min_score": 2.0,  # 最低加权分数才触发信号
    "atr_period": 14,
    "trailing_stop_atr_mult": 2.0,
    "max_hold_bars": 120,
}


@auto_register("multi_indicator")
class MultiIndicatorStrategy(BaseStrategy):
    """多指标投票策略。"""

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

    def _ensure(self, symbol: str) -> None:
        if symbol not in self._close_buf:
            max_len = 80
            self._close_buf[symbol] = deque(maxlen=max_len)
            self._high_buf[symbol] = deque(maxlen=max_len)
            self._low_buf[symbol] = deque(maxlen=max_len)

    def _rsi(self, symbol: str) -> float | None:
        period = int(self.get_param("rsi_period"))
        buf = list(self._close_buf[symbol])
        if len(buf) < period + 1:
            return None
        gains, losses = 0.0, 0.0
        for i in range(-period, 0):
            d = buf[i] - buf[i - 1]
            if d > 0:
                gains += d
            else:
                losses -= d
        avg_g = gains / period
        avg_l = losses / period
        if avg_l == 0:
            return 100.0
        return 100 - 100 / (1 + avg_g / avg_l)

    def _ema_val(self, data: list[float], period: int) -> float | None:
        if len(data) < period:
            return None
        k = 2 / (period + 1)
        v = sum(data[:period]) / period
        for x in data[period:]:
            v = x * k + v * (1 - k)
        return v

    def _macd(self, symbol: str) -> tuple[float, float] | None:
        buf = list(self._close_buf[symbol])
        fast_p = int(self.get_param("macd_fast"))
        slow_p = int(self.get_param("macd_slow"))
        _sig_p = int(self.get_param("macd_signal"))  # noqa: F841
        fast_ema = self._ema_val(buf, fast_p)
        slow_ema = self._ema_val(buf, slow_p)
        if fast_ema is None or slow_ema is None:
            return None
        macd_line = fast_ema - slow_ema
        return (macd_line, 0.0)  # 简化：无 signal line 精确计算

    def _stochastic(self, symbol: str) -> float | None:
        period = int(self.get_param("stoch_period"))
        h = list(self._high_buf[symbol])
        lo = list(self._low_buf[symbol])
        c = list(self._close_buf[symbol])
        if len(h) < period:
            return None
        hh = max(h[-period:])
        ll = min(lo[-period:])
        if hh == ll:
            return 50.0
        return (c[-1] - ll) / (hh - ll) * 100

    def _cci(self, symbol: str) -> float | None:
        period = int(self.get_param("cci_period"))
        h = list(self._high_buf[symbol])
        lo = list(self._low_buf[symbol])
        c = list(self._close_buf[symbol])
        if len(h) < period:
            return None
        tp_list = [(h[i] + lo[i] + c[i]) / 3 for i in range(-period, 0)]
        tp_mean = sum(tp_list) / period
        mean_dev = sum(abs(t - tp_mean) for t in tp_list) / period
        if mean_dev == 0:
            return 0.0
        return (tp_list[-1] - tp_mean) / (0.015 * mean_dev)

    def _calc_atr(self, symbol: str) -> float | None:
        return calc_atr(self._high_buf[symbol], self._low_buf[symbol], self._close_buf[symbol], int(self.get_param("atr_period")))

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._ensure(symbol)
        close = float(bar["close"])
        self._close_buf[symbol].append(close)
        self._high_buf[symbol].append(float(bar["high"]))
        self._low_buf[symbol].append(float(bar["low"]))

        score = 0.0

        rsi = self._rsi(symbol)
        if rsi is not None:
            w = float(self.get_param("rsi_weight"))
            if rsi < float(self.get_param("rsi_oversold")):
                score += w
            elif rsi > float(self.get_param("rsi_overbought")):
                score -= w

        macd_val = self._macd(symbol)
        if macd_val is not None:
            w = float(self.get_param("macd_weight"))
            if macd_val[0] > 0:
                score += w
            elif macd_val[0] < 0:
                score -= w

        stoch = self._stochastic(symbol)
        if stoch is not None:
            w = float(self.get_param("stoch_weight"))
            if stoch < float(self.get_param("stoch_oversold")):
                score += w
            elif stoch > float(self.get_param("stoch_overbought")):
                score -= w

        cci = self._cci(symbol)
        if cci is not None:
            w = float(self.get_param("cci_weight"))
            threshold = float(self.get_param("cci_threshold"))
            if cci < -threshold:
                score += w
            elif cci > threshold:
                score -= w

        min_score = float(self.get_param("min_score"))
        atr = self._calc_atr(symbol)
        signals: list[Signal] = []
        pos = self.get_position(symbol)

        if pos is None and atr is not None:
            if score >= min_score:
                strength = min(score / (min_score * 2), 1.0)
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY, strength=round(strength, 4),
                    price=close, reason=f"多指标做多 score={score:.1f}",
                    metadata={"score": score, "rsi": rsi, "stoch": stoch, "cci": cci},
                ))
                self._peak[symbol] = close
            elif score <= -min_score:
                strength = min(abs(score) / (min_score * 2), 1.0)
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY, strength=round(strength, 4),
                    price=close, reason=f"多指标做空 score={score:.1f}",
                    metadata={"score": score, "rsi": rsi, "stoch": stoch, "cci": cci},
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
                    signal_type=exit_t, strength=0.6, price=close, reason=f"超时{max_hold}",
                ))
                self._bars_in_pos[symbol] = 0
            elif pos.side.value == "buy":
                self._peak[symbol] = max(self._peak.get(symbol, close), close)
                trail = self._peak[symbol] - atr * stop_mult
                if close < trail or score <= -min_score * 0.5:
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.LONG_EXIT, strength=0.8, price=close,
                        reason=f"MultiInd平多 score={score:.1f}",
                    ))
                    self._bars_in_pos[symbol] = 0
            elif pos.side.value == "sell":
                self._trough[symbol] = min(self._trough.get(symbol, close), close)
                trail = self._trough[symbol] + atr * stop_mult
                if close > trail or score >= min_score * 0.5:
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.SHORT_EXIT, strength=0.8, price=close,
                        reason=f"MultiInd平空 score={score:.1f}",
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
