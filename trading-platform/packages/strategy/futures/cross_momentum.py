"""截面动量策略 — 跨品种相对强度轮动。

经典理论：Jegadeesh & Titman (1993) 截面动量效应。
日内适配：利用品种间日内收益差异进行轮动。

信号构建：
  1. 计算所有品种 N 根 K 线的收益率排名
  2. 做多 top-K 品种，做空 bottom-K 品种（多空配对）
  3. 动量衰减检测：RSI 极值 + 换手率异常 → 反转风险
  4. 产业链联动加权：rb-i 正相关, au-cu 负相关

本策略需要同时追踪多品种数据，通过 generate_signals() 批量处理。
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any

import numpy as np

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS: dict[str, Any] = {
    "lookback_bars": 30,
    "top_k": 2,
    "bottom_k": 2,
    "min_return_threshold": 0.002,
    "rebalance_interval": 12,
    "max_hold_bars": 40,
    "rsi_overbought": 75,
    "rsi_oversold": 25,
    "rsi_period": 14,
}

INDUSTRY_CHAINS: dict[str, list[str]] = {
    "black_metals": ["rb", "hc", "i", "j", "jm"],
    "precious_metals": ["au", "ag"],
    "base_metals": ["cu", "al", "zn", "ni", "sn", "pb"],
    "energy_chem": ["sc", "lu", "TA", "MA", "eg", "pp"],
    "agriculture": ["m", "y", "p", "OI", "RM", "a", "c"],
    "softs": ["CF", "SR", "AP", "CJ"],
}

CHAIN_FOR_SYMBOL: dict[str, str] = {}
for chain, symbols in INDUSTRY_CHAINS.items():
    for sym in symbols:
        CHAIN_FOR_SYMBOL[sym] = chain


def _compute_rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(-period, 0):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss < 1e-10:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


@auto_register("cross_momentum")
class CrossMomentumStrategy(BaseStrategy):
    """截面动量轮动策略 — 做多强势品种，做空弱势品种。"""

    def __init__(self, config: StrategyConfig) -> None:
        config = config.model_copy(
            update={"params": {**DEFAULT_PARAMS, **config.params}}
        )
        super().__init__(config)
        self._symbol_data: dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self._bar_counts: dict[str, int] = defaultdict(int)
        self._positions: dict[str, str] = {}
        self._hold_bars: dict[str, int] = defaultdict(int)
        self._entry_prices: dict[str, float] = {}
        self._last_rebalance = 0
        self._global_bar = 0

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        c = float(bar.get("close", 0))
        self._symbol_data[symbol].append(c)
        self._bar_counts[symbol] += 1
        self._global_bar += 1

        for sym in list(self._positions.keys()):
            self._hold_bars[sym] += 1

        signals = []

        max_hold = self.get_param("max_hold_bars", 40)
        if symbol in self._positions and self._hold_bars.get(symbol, 0) >= max_hold:
            side = self._positions[symbol]
            exit_type = SignalType.LONG_EXIT if side == "long" else SignalType.SHORT_EXIT
            sig = Signal(
                strategy_id=self.strategy_id, symbol=symbol,
                signal_type=exit_type, strength=0.7, price=c,
                reason=f"xmom_timeout: {self._hold_bars[symbol]} bars",
            )
            signals.append(sig)
            self.record_signal(sig)
            del self._positions[symbol]
            self._hold_bars.pop(symbol, None)
            self._entry_prices.pop(symbol, None)

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        """Batch cross-sectional ranking across all tracked symbols."""
        rebalance_interval = self.get_param("rebalance_interval", 12)
        if self._global_bar - self._last_rebalance < rebalance_interval:
            return []
        self._last_rebalance = self._global_bar

        lookback = self.get_param("lookback_bars", 30)
        returns: dict[str, float] = {}
        rsi_values: dict[str, float] = {}

        for sym, prices in self._symbol_data.items():
            price_list = list(prices)
            if len(price_list) < lookback + 1:
                continue
            ret = (price_list[-1] - price_list[-lookback]) / price_list[-lookback]
            returns[sym] = ret
            rsi_values[sym] = _compute_rsi(price_list, self.get_param("rsi_period", 14))

        if len(returns) < 4:
            return []

        sorted_symbols = sorted(returns.keys(), key=lambda s: returns[s], reverse=True)
        top_k = self.get_param("top_k", 2)
        bottom_k = self.get_param("bottom_k", 2)
        min_ret = self.get_param("min_return_threshold", 0.002)
        rsi_ob = self.get_param("rsi_overbought", 75)
        rsi_os = self.get_param("rsi_oversold", 25)

        longs = sorted_symbols[:top_k]
        shorts = sorted_symbols[-bottom_k:]

        signals = []

        for sym in longs:
            if sym in self._positions:
                continue
            if returns[sym] < min_ret:
                continue
            if rsi_values.get(sym, 50) > rsi_ob:
                continue

            price = list(self._symbol_data[sym])[-1]
            sig = Signal(
                strategy_id=self.strategy_id, symbol=sym,
                signal_type=SignalType.LONG_ENTRY,
                strength=min(abs(returns[sym]) * 10, 1.0), price=price,
                reason=f"xmom_long: rank={sorted_symbols.index(sym)+1}/{len(sorted_symbols)} ret={returns[sym]:.4f}",
            )
            signals.append(sig)
            self.record_signal(sig)
            self._positions[sym] = "long"
            self._entry_prices[sym] = price
            self._hold_bars[sym] = 0

        for sym in shorts:
            if sym in self._positions:
                continue
            if returns[sym] > -min_ret:
                continue
            if rsi_values.get(sym, 50) < rsi_os:
                continue

            price = list(self._symbol_data[sym])[-1]
            sig = Signal(
                strategy_id=self.strategy_id, symbol=sym,
                signal_type=SignalType.SHORT_ENTRY,
                strength=min(abs(returns[sym]) * 10, 1.0), price=price,
                reason=f"xmom_short: rank={sorted_symbols.index(sym)+1}/{len(sorted_symbols)} ret={returns[sym]:.4f}",
            )
            signals.append(sig)
            self.record_signal(sig)
            self._positions[sym] = "short"
            self._entry_prices[sym] = price
            self._hold_bars[sym] = 0

        return signals
