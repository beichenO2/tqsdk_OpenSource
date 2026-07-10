"""Multi-asset crypto portfolio strategy with per-asset delegation and rebalancing."""

from __future__ import annotations

import logging
from typing import Any

from ..base import BaseStrategy, OrderSide, Position, Signal, SignalType, StrategyConfig
from ..registry import StrategyRegistry, auto_register

logger = logging.getLogger(__name__)

DEFAULT_UNIVERSE = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")

DEFAULT_PARAMS: dict[str, Any] = {
    "allocation_weights": {
        "BTCUSDT": 0.35,
        "ETHUSDT": 0.25,
        "SOLUSDT": 0.20,
        "BNBUSDT": 0.20,
    },
    "rebalance_interval_bars": 48,
    "rebalance_drift_threshold": 0.08,
    "sub_strategy_by_symbol": None,
    "sub_strategy_params_by_symbol": None,
}


def _normalize_weights(weights: dict[str, float], symbols: tuple[str, ...]) -> dict[str, float]:
    raw = {s: float(weights.get(s, 1.0 / len(symbols))) for s in symbols}
    total = sum(raw.values()) or 1.0
    return {s: raw[s] / total for s in symbols}


@auto_register("crypto_portfolio")
class PortfolioStrategy(BaseStrategy):
    """Manage >=3 crypto legs with target weights, child strategies, and periodic rebalance.

    Child strategies are keyed by registry name (e.g. ``btc_trend_following``). Position
    updates on the portfolio are mirrored to the active child for that symbol so
    trend/momentum style children stay consistent in simple backtests.
    """

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        symbols = tuple(config.symbols) if len(config.symbols) >= 3 else DEFAULT_UNIVERSE
        self._symbols: tuple[str, ...] = symbols

        weights_in = dict(self.get_param("allocation_weights") or {})
        self._weights = _normalize_weights(
            {k.upper(): v for k, v in weights_in.items()},
            self._symbols,
        )

        by_sym = self.get_param("sub_strategy_by_symbol") or {}
        params_by_sym = self.get_param("sub_strategy_params_by_symbol") or {}
        default_type = "funding_rate_alpha"

        self._children: dict[str, BaseStrategy] = {}
        for sym in self._symbols:
            reg_name = by_sym.get(sym) or by_sym.get(sym.upper()) or default_type
            extra = dict(params_by_sym.get(sym) or params_by_sym.get(sym.upper()) or {})
            child_cfg = StrategyConfig(
                name=f"{config.name}:{sym}",
                symbols=[sym],
                params=extra,
            )
            self._children[sym] = StrategyRegistry.create(reg_name, child_cfg)

        self._last_closes: dict[str, float] = {}
        self._generate_cycles: int = 0

    def update_position(self, position: Position) -> None:
        super().update_position(position)
        child = self._children.get(position.symbol)
        if child:
            child.update_position(position)

    def remove_position(self, symbol: str) -> None:
        super().remove_position(symbol)
        child = self._children.get(symbol)
        if child:
            child.remove_position(symbol)

    def _tag_signal(self, sig: Signal, symbol: str) -> Signal:
        w = self._weights.get(symbol, 0.0)
        meta = {**sig.metadata, "allocation_weight": w, "parent_strategy_id": self.strategy_id}
        return sig.model_copy(
            update={
                "strategy_id": self.strategy_id,
                "metadata": meta,
            }
        )

    async def _maybe_rebalance(self) -> list[Signal]:
        interval = int(self.get_param("rebalance_interval_bars") or 0)
        drift_thr = float(self.get_param("rebalance_drift_threshold") or 0.1)
        if interval <= 0 or self._generate_cycles % interval != 0:
            return []

        positions = {s: self.get_position(s) for s in self._symbols}
        active = {s: p for s, p in positions.items() if p is not None}
        if len(active) < 2:
            return []

        mv: dict[str, float] = {}
        for s, p in active.items():
            px = self._last_closes.get(s)
            if px is None or px <= 0:
                continue
            mv[s] = abs(p.qty) * px
        if not mv:
            return []

        total_mv = sum(mv.values())
        if total_mv <= 0:
            return []

        out: list[Signal] = []
        for s, p in active.items():
            if s not in mv:
                continue
            target_w = self._weights.get(s, 0.0)
            cur_w = mv[s] / total_mv
            if target_w <= 0:
                continue
            if cur_w <= target_w * (1.0 + drift_thr):
                continue
            px = self._last_closes[s]
            target_mv = total_mv * target_w
            excess_mv = mv[s] - target_mv
            if excess_mv <= 0 or px <= 0:
                continue
            trim_qty = excess_mv / px
            trim_qty = min(trim_qty, p.qty * 0.999)

            if p.side == OrderSide.BUY:
                sig = Signal(
                    strategy_id=self.strategy_id,
                    symbol=s,
                    signal_type=SignalType.LONG_EXIT,
                    strength=min(1.0, (cur_w - target_w) * 5),
                    price=px,
                    suggested_qty=trim_qty,
                    reason=(
                        f"组合再平衡: 权重 {cur_w:.2%} > 目标 {target_w:.2%} (+{drift_thr:.0%})"
                    ),
                    metadata={
                        "rebalance": True,
                        "current_weight": cur_w,
                        "target_weight": target_w,
                    },
                )
            else:
                sig = Signal(
                    strategy_id=self.strategy_id,
                    symbol=s,
                    signal_type=SignalType.SHORT_EXIT,
                    strength=min(1.0, (cur_w - target_w) * 5),
                    price=px,
                    suggested_qty=trim_qty,
                    reason=(
                        f"组合再平衡: 空头名义占比 {cur_w:.2%} > 目标 {target_w:.2%}"
                    ),
                    metadata={
                        "rebalance": True,
                        "current_weight": cur_w,
                        "target_weight": target_w,
                    },
                )
            out.append(sig)
            self.record_signal(sig)
        return out

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        if symbol not in self._children:
            return []

        close = float(bar["close"])
        self._last_closes[symbol] = close

        child = self._children[symbol]
        raw = await child.on_bar(symbol, bar)
        signals = [self._tag_signal(s, symbol) for s in raw]
        for s in signals:
            self.record_signal(s)
        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        ordered = [s for s in self._symbols if s in market_data]
        all_sig: list[Signal] = []
        for sym in ordered:
            bar = market_data[sym]
            if bar:
                all_sig.extend(await self.on_bar(sym, bar))
        self._generate_cycles += 1
        all_sig.extend(await self._maybe_rebalance())
        return all_sig


def _load_child_strategy_modules_for_registry() -> None:
    """Ensure active strategy modules are imported for registry registration.

    meta_labeling / patch_tst_strategy moved to _archived — do not import here.
    """
    from . import funding_rate_alpha as _funding  # noqa: F401
    from . import cross_sectional_momentum as _momentum  # noqa: F401
    from . import funding_meta_ensemble as _ensemble  # noqa: F401


_load_child_strategy_modules_for_registry()
