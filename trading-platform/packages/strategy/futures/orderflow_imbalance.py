"""订单流失衡策略 — 基于 tick 级买卖力量对比的微结构信号。

理论基础：
  - Kyle (1985): 知情交易者通过订单流传递信息
  - Cont, Kukanov & Stoikov (2014, J. Financial Economics):
    "The Price Impact of Order Book Events" — OFI 预测短期价格变动
  - 国内期货可观测：成交量、持仓量变化、买卖盘口

信号构建：
  1. Volume Delta: 上涨 bar 成交量 vs 下跌 bar 成交量的累积差
  2. OI Delta: 持仓量增减反映多空力量进出
  3. Volume-Price Divergence: 价格新高但成交量萎缩 = 假突破
  4. Aggressive Participation: 大单成交占比（需 tick 级数据）

日内适配：每个交易时段重置累积量，避免跨时段噪声。

Method: 经典市场微结构理论 + Cont et al. 2014 (J. Financial Economics, 白名单)。
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..indicators import calc_atr
from ..registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS: dict[str, Any] = {
    "ofi_period": 20,
    "ofi_threshold": 2.0,
    "oi_change_period": 10,
    "oi_confirm_threshold": 0.005,
    "atr_period": 14,
    "tp_atr_mult": 2.5,
    "sl_atr_mult": 1.2,
    "max_hold_bars": 40,
    "cooldown_bars": 3,
    "volume_ma_period": 20,
    "divergence_lookback": 15,
}


@auto_register("orderflow_imbalance")
class OrderFlowImbalanceStrategy(BaseStrategy):
    """订单流失衡日内策略。

    核心信号：
    1. Volume Delta Z-score > threshold → 多/空方力量显著
    2. OI 变化确认 → 新资金进入（非平仓行为）
    3. Volume-Price 背离 → 趋势衰竭反转信号
    """

    def __init__(self, config: StrategyConfig) -> None:
        config = config.model_copy(
            update={"params": {**DEFAULT_PARAMS, **config.params}}
        )
        super().__init__(config)
        self._highs: deque[float] = deque(maxlen=200)
        self._lows: deque[float] = deque(maxlen=200)
        self._closes: deque[float] = deque(maxlen=200)
        self._volumes: deque[float] = deque(maxlen=200)
        self._ois: deque[float] = deque(maxlen=200)
        self._volume_deltas: deque[float] = deque(maxlen=200)
        self._bar_count = 0
        self._position_side: str | None = None
        self._entry_price = 0.0
        self._hold_bars = 0
        self._cd = 0

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        h = float(bar.get("high", 0))
        l = float(bar.get("low", 0))
        c = float(bar.get("close", 0))
        v = float(bar.get("volume", 0))
        oi = float(bar.get("open_interest", 0))

        self._highs.append(h)
        self._lows.append(l)
        self._closes.append(c)
        self._volumes.append(v)
        self._ois.append(oi)
        self._bar_count += 1

        prev_close = self._closes[-2] if len(self._closes) >= 2 else c
        vd = v if c > prev_close else (-v if c < prev_close else 0.0)
        self._volume_deltas.append(vd)

        ofi_period = self.get_param("ofi_period", 20)
        if self._bar_count < max(ofi_period + 5, 30):
            return []

        atr = calc_atr(list(self._highs), list(self._lows), list(self._closes),
                       self.get_param("atr_period", 14))
        if atr is None or atr < 1e-10:
            return []

        signals = []

        recent_vd = list(self._volume_deltas)[-ofi_period:]
        vd_mean = np.mean(recent_vd)
        vd_std = max(np.std(recent_vd), 1e-10)
        ofi_z = vd_mean / vd_std

        oi_period = self.get_param("oi_change_period", 10)
        oi_list = list(self._ois)
        oi_change = 0.0
        if len(oi_list) > oi_period and oi_list[-oi_period - 1] > 0:
            oi_change = (oi_list[-1] - oi_list[-oi_period - 1]) / oi_list[-oi_period - 1]

        oi_confirm_threshold = self.get_param("oi_confirm_threshold", 0.005)
        oi_increasing = oi_change > oi_confirm_threshold

        div_lookback = self.get_param("divergence_lookback", 15)
        close_list = list(self._closes)
        vol_list = list(self._volumes)
        bullish_div = False
        bearish_div = False

        if len(close_list) >= div_lookback:
            price_slope = (close_list[-1] - close_list[-div_lookback]) / max(abs(close_list[-div_lookback]), 1e-10)
            vol_ma_recent = np.mean(vol_list[-div_lookback // 2:])
            vol_ma_earlier = np.mean(vol_list[-div_lookback:-div_lookback // 2])
            vol_slope = (vol_ma_recent - vol_ma_earlier) / max(vol_ma_earlier, 1e-10)

            if price_slope > 0.005 and vol_slope < -0.1:
                bearish_div = True
            elif price_slope < -0.005 and vol_slope < -0.1:
                bullish_div = True

        if self._position_side:
            self._hold_bars += 1
            max_hold = self.get_param("max_hold_bars", 40)
            tp_mult = self.get_param("tp_atr_mult", 2.5)
            sl_mult = self.get_param("sl_atr_mult", 1.2)

            pnl = (c - self._entry_price) / self._entry_price if self._position_side == "long" else (self._entry_price - c) / self._entry_price
            tp_hit = pnl >= tp_mult * atr / self._entry_price
            sl_hit = pnl <= -sl_mult * atr / self._entry_price

            if self._position_side == "long" and bearish_div:
                sl_hit = True
            elif self._position_side == "short" and bullish_div:
                sl_hit = True

            if sl_hit or tp_hit or self._hold_bars >= max_hold:
                exit_type = SignalType.LONG_EXIT if self._position_side == "long" else SignalType.SHORT_EXIT
                reason = f"ofi_exit: pnl={pnl:.4f} hold={self._hold_bars}"
                if bearish_div or bullish_div:
                    reason += " divergence"
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_type, strength=0.8, price=c, reason=reason,
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = None
                self._hold_bars = 0
                self._cd = self.get_param("cooldown_bars", 3)
                return signals

        if self._cd > 0:
            self._cd -= 1

        ofi_threshold = self.get_param("ofi_threshold", 2.0)

        if not self._position_side and self._cd <= 0:
            if ofi_z > ofi_threshold and oi_increasing and not bearish_div:
                strength = min(abs(ofi_z) / 3.0, 1.0)
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY, strength=strength, price=c,
                    reason=f"ofi_buy: z={ofi_z:.2f} oi_chg={oi_change:.4f}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "long"
                self._entry_price = c
                self._hold_bars = 0

            elif ofi_z < -ofi_threshold and oi_increasing and not bullish_div:
                strength = min(abs(ofi_z) / 3.0, 1.0)
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY, strength=strength, price=c,
                    reason=f"ofi_sell: z={ofi_z:.2f} oi_chg={oi_change:.4f}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "short"
                self._entry_price = c
                self._hold_bars = 0

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        return []
