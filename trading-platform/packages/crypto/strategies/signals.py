"""BTC 信号聚合器 - 将多策略信号合并为最终交易决策。"""

from __future__ import annotations

import logging

from strategy.base import Signal, SignalType

logger = logging.getLogger(__name__)


class BTCSignalAggregator:
    """聚合多个 BTC 策略的信号，通过加权投票产出最终信号。

    当多个策略对同一标的同时发出信号时，聚合器按权重投票决定最终操作。
    """

    def __init__(self, strategy_weights: dict[str, float] | None = None) -> None:
        self._weights = strategy_weights or {}
        self._signal_buffer: list[Signal] = []

    def set_weight(self, strategy_id: str, weight: float) -> None:
        self._weights[strategy_id] = weight

    def add_signal(self, signal: Signal) -> None:
        self._signal_buffer.append(signal)

    def add_signals(self, signals: list[Signal]) -> None:
        self._signal_buffer.extend(signals)

    def aggregate(self, symbol: str) -> Signal | None:
        """对指定标的的缓冲信号执行加权投票聚合。"""
        relevant = [s for s in self._signal_buffer if s.symbol == symbol]
        if not relevant:
            return None

        score_map: dict[SignalType, float] = {}
        for sig in relevant:
            w = self._weights.get(sig.strategy_id, 1.0)
            weighted = sig.strength * w
            score_map[sig.signal_type] = score_map.get(sig.signal_type, 0.0) + weighted

        if not score_map:
            return None

        best_type = max(score_map, key=lambda k: score_map[k])
        total_score = score_map[best_type]
        max_possible = sum(self._weights.get(s.strategy_id, 1.0) for s in relevant)
        normalized_strength = min(total_score / max_possible, 1.0) if max_possible > 0 else 0.0

        best_signals = [s for s in relevant if s.signal_type == best_type]
        reasons = [s.reason for s in best_signals if s.reason]

        aggregated = Signal(
            strategy_id="aggregator",
            symbol=symbol,
            signal_type=best_type,
            strength=round(normalized_strength, 4),
            reason=f"聚合({len(relevant)}信号): {'; '.join(reasons[:3])}",
            metadata={
                "source_count": len(relevant),
                "vote_scores": {k.value: round(v, 4) for k, v in score_map.items()},
                "contributing_strategies": [s.strategy_id for s in best_signals],
            },
        )

        logger.info(
            "信号聚合: symbol=%s type=%s strength=%.4f sources=%d",
            symbol, best_type.value, normalized_strength, len(relevant),
        )
        return aggregated

    def flush(self) -> None:
        """清空信号缓冲区。"""
        self._signal_buffer.clear()

    def flush_symbol(self, symbol: str) -> None:
        """清空指定标的的信号缓冲区。"""
        self._signal_buffer = [s for s in self._signal_buffer if s.symbol != symbol]
