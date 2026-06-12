"""产业链联动策略 — 利用上下游品种间的价格传导关系。

国内期货产业链关系：
  黑色链: 焦煤(jm) → 焦炭(j) → 铁矿(i) → 螺纹(rb)/热卷(hc)
  能化链: 原油(sc) → 燃油(lu)/沥青(bu) → PTA(TA)/MEG(eg) → 聚酯
  油脂链: 棕榈(p) / 豆油(y) / 菜油(OI) — 相互替代
  蛋白链: 豆粕(m) / 菜粕(RM) — 替代效应

信号构建：
  1. 上游品种先行突破 → 下游品种跟随（领先-滞后关系）
  2. 价差异常收敛/扩大 → 套利/趋势信号
  3. 比价关系（如 焦煤/焦炭比）偏离历史均值 → 回归

Method: 经典产业经济学 + 统计套利。产业链关系是经济学基础知识。
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
    "spread_lookback": 30,
    "spread_zscore_entry": 2.0,
    "spread_zscore_exit": 0.3,
    "lead_lag_lookback": 10,
    "lead_lag_threshold": 0.003,
    "max_hold_bars": 40,
    "cooldown_bars": 3,
}

CHAIN_PAIRS: list[tuple[str, str, str]] = [
    ("rb", "i", "black"),    # 螺纹 vs 铁矿
    ("rb", "hc", "black"),   # 螺纹 vs 热卷
    ("j", "jm", "coke"),     # 焦炭 vs 焦煤
    ("sc", "TA", "chem"),    # 原油 vs PTA
    ("sc", "lu", "energy"),  # 原油 vs 燃油
    ("p", "y", "oil_fat"),   # 棕榈 vs 豆油
    ("m", "RM", "protein"),  # 豆粕 vs 菜粕
    ("cu", "al", "base"),    # 铜 vs 铝
]


@auto_register("commodity_chain")
class CommodityChainStrategy(BaseStrategy):
    """产业链联动策略 — 跨品种价差回归 + 领先滞后跟随。"""

    def __init__(self, config: StrategyConfig) -> None:
        config = config.model_copy(
            update={"params": {**DEFAULT_PARAMS, **config.params}}
        )
        super().__init__(config)
        self._symbol_data: dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self._bar_count = 0
        self._positions: dict[str, str] = {}
        self._hold_bars: dict[str, int] = defaultdict(int)
        self._entry_prices: dict[str, float] = {}
        self._cd: dict[str, int] = defaultdict(int)

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        c = float(bar.get("close", 0))
        base = "".join(ch for ch in symbol if ch.isalpha())
        self._symbol_data[base].append(c)
        self._bar_count += 1

        for sym in list(self._positions.keys()):
            self._hold_bars[sym] += 1

        signals = []
        max_hold = self.get_param("max_hold_bars", 40)

        if base in self._positions and self._hold_bars.get(base, 0) >= max_hold:
            side = self._positions[base]
            exit_type = SignalType.LONG_EXIT if side == "long" else SignalType.SHORT_EXIT
            sig = Signal(
                strategy_id=self.strategy_id, symbol=symbol,
                signal_type=exit_type, strength=0.7, price=c,
                reason=f"chain_timeout: {self._hold_bars[base]} bars",
            )
            signals.append(sig)
            self.record_signal(sig)
            del self._positions[base]
            self._hold_bars.pop(base, None)
            self._cd[base] = self.get_param("cooldown_bars", 3)

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        """Cross-symbol chain analysis."""
        spread_lb = self.get_param("spread_lookback", 30)
        entry_z = self.get_param("spread_zscore_entry", 2.0)
        exit_z = self.get_param("spread_zscore_exit", 0.3)
        lead_lb = self.get_param("lead_lag_lookback", 10)
        lead_threshold = self.get_param("lead_lag_threshold", 0.003)

        signals = []

        for sym_a, sym_b, chain in CHAIN_PAIRS:
            data_a = list(self._symbol_data.get(sym_a, []))
            data_b = list(self._symbol_data.get(sym_b, []))

            if len(data_a) < spread_lb or len(data_b) < spread_lb:
                continue

            prices_a = np.array(data_a[-spread_lb:])
            prices_b = np.array(data_b[-spread_lb:])
            ratio = prices_a / np.maximum(prices_b, 1e-10)

            ratio_mean = np.mean(ratio)
            ratio_std = np.std(ratio)
            if ratio_std < 1e-10:
                continue

            current_ratio = ratio[-1]
            z = (current_ratio - ratio_mean) / ratio_std

            pair_key = f"{sym_a}_{sym_b}"
            if self._cd.get(pair_key, 0) > 0:
                self._cd[pair_key] -= 1
                continue

            if pair_key in self._positions:
                side = self._positions[pair_key]
                if (side == "long" and z > -exit_z) or (side == "short" and z < exit_z):
                    price_a = data_a[-1]
                    exit_type = SignalType.LONG_EXIT if side == "long" else SignalType.SHORT_EXIT
                    sig = Signal(
                        strategy_id=self.strategy_id, symbol=sym_a,
                        signal_type=exit_type, strength=0.7, price=price_a,
                        reason=f"chain_spread_exit: {pair_key} z={z:.2f} ratio={current_ratio:.4f}",
                    )
                    signals.append(sig)
                    self.record_signal(sig)
                    del self._positions[pair_key]
                    self._hold_bars.pop(pair_key, None)
                    self._cd[pair_key] = self.get_param("cooldown_bars", 3)
                continue

            if z < -entry_z:
                price_a = data_a[-1]
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=sym_a,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=min(abs(z) / 3.0, 1.0), price=price_a,
                    reason=f"chain_spread_buy: {pair_key} z={z:.2f} (ratio below mean, expect convergence)",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._positions[pair_key] = "long"
                self._entry_prices[pair_key] = price_a
                self._hold_bars[pair_key] = 0

            elif z > entry_z:
                price_a = data_a[-1]
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=sym_a,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=min(abs(z) / 3.0, 1.0), price=price_a,
                    reason=f"chain_spread_sell: {pair_key} z={z:.2f} (ratio above mean, expect reversion)",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._positions[pair_key] = "short"
                self._entry_prices[pair_key] = price_a
                self._hold_bars[pair_key] = 0

        return signals
