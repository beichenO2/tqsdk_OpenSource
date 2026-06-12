"""期货策略集成系统 — 多策略投票/加权组合。

将 18 个独立策略的信号统一为一个决策：
  1. 信号收集：每个子策略在 on_bar 中独立产生信号
  2. 分类汇总：按方向（多/空/平）分组计票
  3. 加权投票：根据策略历史表现动态调整权重
  4. 冲突解决：多空信号同时出现时的处理规则
  5. 仓位分配：信号一致性越高 → 仓位越大

策略分组（降低同类策略的重复投票权重）：
  - 趋势类：cta_trend, dual_ma, kalman_trend, chan_theory
  - 动量类：cross_momentum, regime_momentum, orderflow_imbalance
  - 回归类：bollinger_mr, intraday_reversal, har_volatility
  - 结构类：rbreaker, spread_arb, pairs_trading
  - ML类：dl_timeseries
  - 情绪类：news_sentiment
  - 量价类：vol_breakout, volume_price

Method: 集成学习 (ensemble methods) — 经典 ML 方法，Breiman 2001 Random Forests
概念的策略层面应用。投票机制类似 bagging。
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS: dict[str, Any] = {
    "min_consensus_ratio": 0.4,
    "strong_consensus_ratio": 0.6,
    "max_hold_bars": 40,
    "cooldown_bars": 3,
    "group_weight_cap": 0.3,
}

STRATEGY_GROUPS: dict[str, list[str]] = {
    "trend": ["cta_trend", "dual_ma", "kalman_trend", "chan_theory"],
    "momentum": ["cross_momentum", "regime_momentum", "orderflow_imbalance"],
    "reversion": ["bollinger_mr", "intraday_reversal", "har_volatility"],
    "structural": ["rbreaker", "spread_arb", "pairs_trading"],
    "ml": ["dl_timeseries"],
    "sentiment": ["news_sentiment"],
    "vol_price": ["vol_breakout", "volume_price", "adaptive_bollinger"],
}

GROUP_FOR_STRATEGY: dict[str, str] = {}
for group, strats in STRATEGY_GROUPS.items():
    for s in strats:
        GROUP_FOR_STRATEGY[s] = group


class SignalAggregator:
    """收集多策略信号并产生集成决策。"""

    def __init__(self, group_weight_cap: float = 0.3):
        self._signals: list[Signal] = []
        self._group_weight_cap = group_weight_cap

    def add_signal(self, signal: Signal) -> None:
        self._signals.append(signal)

    def clear(self) -> None:
        self._signals.clear()

    def aggregate(self) -> dict[str, Any]:
        """Compute weighted vote across all collected signals."""
        if not self._signals:
            return {"direction": "hold", "score": 0.0, "consensus": 0.0, "voters": 0}

        long_votes: dict[str, float] = defaultdict(float)
        short_votes: dict[str, float] = defaultdict(float)

        group_long_count: dict[str, int] = defaultdict(int)
        group_short_count: dict[str, int] = defaultdict(int)

        for sig in self._signals:
            strategy_name = sig.strategy_id
            group = GROUP_FOR_STRATEGY.get(strategy_name, "other")
            weight = sig.strength

            if sig.signal_type in (SignalType.LONG_ENTRY,):
                group_long_count[group] += 1
                intra_group_discount = 1.0 / max(group_long_count[group], 1)
                long_votes[strategy_name] = weight * intra_group_discount

            elif sig.signal_type in (SignalType.SHORT_ENTRY,):
                group_short_count[group] += 1
                intra_group_discount = 1.0 / max(group_short_count[group], 1)
                short_votes[strategy_name] = weight * intra_group_discount

        total_long = sum(long_votes.values())
        total_short = sum(short_votes.values())
        total_votes = len(long_votes) + len(short_votes)

        if total_votes == 0:
            return {"direction": "hold", "score": 0.0, "consensus": 0.0, "voters": 0}

        if total_long > total_short:
            direction = "long"
            score = total_long
            consensus = len(long_votes) / total_votes
        elif total_short > total_long:
            direction = "short"
            score = total_short
            consensus = len(short_votes) / total_votes
        else:
            direction = "hold"
            score = 0.0
            consensus = 0.0

        unique_groups_agreeing = len(set(
            GROUP_FOR_STRATEGY.get(s, "other")
            for s in (long_votes if direction == "long" else short_votes)
        ))

        return {
            "direction": direction,
            "score": score,
            "consensus": consensus,
            "voters": total_votes,
            "long_count": len(long_votes),
            "short_count": len(short_votes),
            "groups_agreeing": unique_groups_agreeing,
            "long_details": dict(long_votes),
            "short_details": dict(short_votes),
        }


@auto_register("ensemble_futures")
class EnsembleFuturesStrategy(BaseStrategy):
    """期货策略集成 — 多策略投票决策系统。

    用法：在 on_bar 中收集子策略信号，然后调用 _process_round 做集成决策。
    单独使用时作为一个框架，需要外部注入子策略信号。
    """

    def __init__(self, config: StrategyConfig) -> None:
        config = config.model_copy(
            update={"params": {**DEFAULT_PARAMS, **config.params}}
        )
        super().__init__(config)
        self._aggregator = SignalAggregator(
            group_weight_cap=self.get_param("group_weight_cap", 0.3),
        )
        self._position_side: str | None = None
        self._entry_price = 0.0
        self._hold_bars = 0
        self._cd = 0
        self._bar_count = 0

    def inject_sub_signals(self, signals: list[Signal]) -> None:
        """External injection point: add sub-strategy signals for aggregation."""
        for sig in signals:
            self._aggregator.add_signal(sig)

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        c = float(bar.get("close", 0))
        self._bar_count += 1

        result = self._aggregator.aggregate()
        self._aggregator.clear()

        signals = []

        if self._position_side:
            self._hold_bars += 1
            max_hold = self.get_param("max_hold_bars", 40)

            opposite = (self._position_side == "long" and result["direction"] == "short") or \
                      (self._position_side == "short" and result["direction"] == "long")

            if self._hold_bars >= max_hold or (opposite and result["consensus"] > 0.5):
                exit_type = SignalType.LONG_EXIT if self._position_side == "long" else SignalType.SHORT_EXIT
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_type, strength=0.8, price=c,
                    reason=f"ensemble_exit: hold={self._hold_bars} opposite={opposite}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = None
                self._hold_bars = 0
                self._cd = self.get_param("cooldown_bars", 3)
                return signals

        if self._cd > 0:
            self._cd -= 1

        min_consensus = self.get_param("min_consensus_ratio", 0.4)

        if not self._position_side and self._cd <= 0:
            if result["direction"] == "long" and result["consensus"] >= min_consensus and result["groups_agreeing"] >= 2:
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=min(result["score"], 1.0), price=c,
                    reason=f"ensemble_buy: {result['long_count']}L vs {result['short_count']}S, {result['groups_agreeing']} groups, consensus={result['consensus']:.2f}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "long"
                self._entry_price = c
                self._hold_bars = 0

            elif result["direction"] == "short" and result["consensus"] >= min_consensus and result["groups_agreeing"] >= 2:
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=min(result["score"], 1.0), price=c,
                    reason=f"ensemble_sell: {result['long_count']}L vs {result['short_count']}S, {result['groups_agreeing']} groups, consensus={result['consensus']:.2f}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "short"
                self._entry_price = c
                self._hold_bars = 0

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        return []
