"""订单流策略 — VWAP 回归 + OBV 背离 + 买卖力道对比。

SOTA 要点:
- VWAP (Volume Weighted Average Price): 机构基准价，偏离过大则回归
- OBV (On Balance Volume): 量价背离检测
- Taker Buy Ratio: 主买/主卖力量对比
- 成交量 profile: 识别高成交量价格区域
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
    "vwap_period": 20,
    "vwap_dev_entry": 2.0,   # VWAP 偏离标准差入场
    "vwap_dev_exit": 0.5,
    "obv_period": 14,
    "obv_divergence": True,
    "taker_ratio_period": 10,
    "taker_ratio_threshold": 0.6,
    "taker_weight": 1.0,
    "atr_period": 14,
    "trailing_stop_atr_mult": 2.0,
    "max_hold_bars": 100,
}


@auto_register("orderflow")
class OrderFlowStrategy(BaseStrategy):
    """订单流分析策略。"""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._close_buf: dict[str, deque[float]] = {}
        self._high_buf: dict[str, deque[float]] = {}
        self._low_buf: dict[str, deque[float]] = {}
        self._vol_buf: dict[str, deque[float]] = {}
        self._taker_buy_buf: dict[str, deque[float]] = {}
        self._obv: dict[str, float] = {}
        self._obv_history: dict[str, deque[float]] = {}
        self._bars_in_pos: dict[str, int] = {}
        self._peak: dict[str, float] = {}
        self._trough: dict[str, float] = {}

    def _ensure(self, symbol: str) -> None:
        if symbol not in self._close_buf:
            ml = 60
            self._close_buf[symbol] = deque(maxlen=ml)
            self._high_buf[symbol] = deque(maxlen=ml)
            self._low_buf[symbol] = deque(maxlen=ml)
            self._vol_buf[symbol] = deque(maxlen=ml)
            self._taker_buy_buf[symbol] = deque(maxlen=ml)
            self._obv[symbol] = 0.0
            self._obv_history[symbol] = deque(maxlen=ml)

    def _vwap(self, symbol: str) -> tuple[float, float] | None:
        """计算 VWAP 和标准差。"""
        period = int(self.get_param("vwap_period"))
        c = list(self._close_buf[symbol])
        v = list(self._vol_buf[symbol])
        h = list(self._high_buf[symbol])
        lo = list(self._low_buf[symbol])
        if len(c) < period:
            return None
        tp_v = [((h[i] + lo[i] + c[i]) / 3) * v[i] for i in range(-period, 0)]
        vol_sum = sum(v[-period:])
        if vol_sum == 0:
            return None
        vwap_val = sum(tp_v) / vol_sum
        # VWAP 标准差
        tp = [(h[i] + lo[i] + c[i]) / 3 for i in range(-period, 0)]
        var = sum((t - vwap_val) ** 2 * v[i] for i, t in zip(range(-period, 0), tp)) / vol_sum
        std = var ** 0.5 if var > 0 else 1e-10
        return (vwap_val, std)

    def _update_obv(self, symbol: str, close: float, prev_close: float, volume: float) -> None:
        if close > prev_close:
            self._obv[symbol] += volume
        elif close < prev_close:
            self._obv[symbol] -= volume
        self._obv_history[symbol].append(self._obv[symbol])

    def _obv_divergence_signal(self, symbol: str) -> int:
        """OBV 与价格背离: 价格新高但 OBV 未新高 = 看跌背离。"""
        obv_period = int(self.get_param("obv_period"))
        c = list(self._close_buf[symbol])
        obv_h = list(self._obv_history[symbol])
        if len(c) < obv_period or len(obv_h) < obv_period:
            return 0
        price_new_high = c[-1] >= max(c[-obv_period:])
        price_new_low = c[-1] <= min(c[-obv_period:])
        obv_new_high = obv_h[-1] >= max(obv_h[-obv_period:])
        obv_new_low = obv_h[-1] <= min(obv_h[-obv_period:])

        if price_new_high and not obv_new_high:
            return -1  # 看跌背离
        if price_new_low and not obv_new_low:
            return 1   # 看涨背离
        return 0

    def _taker_ratio(self, symbol: str) -> float | None:
        period = int(self.get_param("taker_ratio_period"))
        tb = list(self._taker_buy_buf[symbol])
        v = list(self._vol_buf[symbol])
        if len(tb) < period or len(v) < period:
            return None
        total_tb = sum(tb[-period:])
        total_v = sum(v[-period:])
        if total_v == 0:
            return None
        return total_tb / total_v

    def _calc_atr(self, symbol: str) -> float | None:
        return calc_atr(self._high_buf[symbol], self._low_buf[symbol], self._close_buf[symbol], int(self.get_param("atr_period")))

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._ensure(symbol)
        close = float(bar["close"])
        high = float(bar["high"])
        low = float(bar["low"])
        volume = float(bar.get("volume", 0))
        taker_buy = float(bar.get("taker_buy_volume", volume * 0.5))

        prev_close = list(self._close_buf[symbol])[-1] if self._close_buf[symbol] else close
        self._close_buf[symbol].append(close)
        self._high_buf[symbol].append(high)
        self._low_buf[symbol].append(low)
        self._vol_buf[symbol].append(volume)
        self._taker_buy_buf[symbol].append(taker_buy)
        self._update_obv(symbol, close, prev_close, volume)

        vwap_result = self._vwap(symbol)
        atr = self._calc_atr(symbol)
        if vwap_result is None or atr is None:
            return []

        vwap_val, vwap_std = vwap_result
        dev_entry = float(self.get_param("vwap_dev_entry"))
        dev_exit = float(self.get_param("vwap_dev_exit"))
        z = (close - vwap_val) / vwap_std if vwap_std > 0 else 0

        signals: list[Signal] = []
        pos = self.get_position(symbol)

        score = 0.0
        # VWAP 信号
        if z < -dev_entry:
            score += 1.0
        elif z > dev_entry:
            score -= 1.0

        # OBV 背离
        if self.get_param("obv_divergence"):
            obv_sig = self._obv_divergence_signal(symbol)
            score += obv_sig * 0.5

        # Taker ratio
        tr = self._taker_ratio(symbol)
        if tr is not None:
            threshold = float(self.get_param("taker_ratio_threshold"))
            tw = float(self.get_param("taker_weight"))
            if tr > threshold:
                score += tw * 0.5
            elif tr < 1 - threshold:
                score -= tw * 0.5

        if pos is None:
            if score >= 1.0:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY, strength=min(score / 2, 1.0),
                    price=close, reason=f"OrderFlow做多 z={z:.2f} score={score:.1f}",
                ))
                self._peak[symbol] = close
            elif score <= -1.0:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY, strength=min(abs(score) / 2, 1.0),
                    price=close, reason=f"OrderFlow做空 z={z:.2f} score={score:.1f}",
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
                if close < trail or abs(z) < dev_exit:
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.LONG_EXIT, strength=0.8, price=close,
                        reason=f"OF平多 z={z:.2f}",
                    ))
                    self._bars_in_pos[symbol] = 0
            elif pos.side.value == "sell":
                self._trough[symbol] = min(self._trough.get(symbol, close), close)
                trail = self._trough[symbol] + atr * stop_mult
                if close > trail or abs(z) < dev_exit:
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.SHORT_EXIT, strength=0.8, price=close,
                        reason=f"OF平空 z={z:.2f}",
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
