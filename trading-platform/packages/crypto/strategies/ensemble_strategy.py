"""Combine >=3 crypto sub-strategies by vote or confidence-weighted score."""

from __future__ import annotations

import logging
from typing import Any

from strategy.base import BaseStrategy, Position, Signal, SignalType, StrategyConfig
from strategy.registry import StrategyRegistry, auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS: dict[str, Any] = {
    "ensemble_mode": "vote",
    "sub_strategy_types": (
        "funding_rate_alpha",
        "time_series_momentum",
        "funding_meta_ensemble",
    ),
    "sub_strategy_weights": None,
    "action_threshold": 0.45,
}


def _dominant_entry(signals: list[Signal]) -> tuple[str, float]:
    long_s = max(
        (s.strength for s in signals if s.signal_type == SignalType.LONG_ENTRY),
        default=0.0,
    )
    short_s = max(
        (s.strength for s in signals if s.signal_type == SignalType.SHORT_ENTRY),
        default=0.0,
    )
    if long_s <= 0 and short_s <= 0:
        return "hold", 0.0
    if long_s >= short_s:
        return "long", float(long_s)
    return "short", float(short_s)


def _dominant_exit(signals: list[Signal], pos_side: str | None) -> Signal | None:
    if pos_side == "long":
        exits = [s for s in signals if s.signal_type == SignalType.LONG_EXIT]
    elif pos_side == "short":
        exits = [s for s in signals if s.signal_type == SignalType.SHORT_EXIT]
    else:
        return None
    if not exits:
        return None
    return max(exits, key=lambda s: s.strength)


@auto_register("crypto_ensemble")
class EnsembleStrategy(BaseStrategy):
    """Aggregate >=3 registered sub-strategies on the same symbol.

    Modes:
    - ``vote``: weighted vote on discrete long / short / hold from each child.
    - ``weighted``: sum of signed strengths (long positive, short negative), thresholded.
    """

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        types: tuple[str, ...] = tuple(self.get_param("sub_strategy_types"))
        if len(types) < 3:
            raise ValueError("EnsembleStrategy requires at least 3 sub_strategy_types")

        weights = self.get_param("sub_strategy_weights")
        if weights is None:
            w = 1.0 / len(types)
            self._weights = [w] * len(types)
        else:
            wlist = [float(x) for x in weights]
            if len(wlist) != len(types):
                raise ValueError("sub_strategy_weights must match sub_strategy_types length")
            s = sum(wlist) or 1.0
            self._weights = [x / s for x in wlist]

        symbol = config.symbols[0] if config.symbols else "BTCUSDT"
        self._trade_symbol = symbol

        self._children: list[BaseStrategy] = []
        for i, reg_name in enumerate(types):
            child_cfg = StrategyConfig(
                name=f"{config.name}:sub{i}",
                symbols=[self._trade_symbol],
                params={},
            )
            self._children.append(StrategyRegistry.create(reg_name, child_cfg))

    def update_position(self, position: Position) -> None:
        super().update_position(position)
        for c in self._children:
            c.update_position(position)

    def remove_position(self, symbol: str) -> None:
        super().remove_position(symbol)
        for c in self._children:
            c.remove_position(symbol)

    def _pos_side(self) -> str | None:
        p = self.get_position(self._trade_symbol)
        if p is None:
            return None
        if p.side.value == "buy":
            return "long"
        return "short"

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        if symbol != self._trade_symbol:
            return []

        per_child: list[list[Signal]] = []
        for child in self._children:
            per_child.append(await child.on_bar(symbol, bar))

        pos_side = self._pos_side()
        threshold = float(self.get_param("action_threshold"))
        mode = str(self.get_param("ensemble_mode")).lower()

        exit_sig = None
        for raw in per_child:
            cand = _dominant_exit(raw, pos_side)
            if cand is not None and (exit_sig is None or cand.strength > exit_sig.strength):
                exit_sig = cand

        if pos_side and exit_sig is not None and exit_sig.strength >= threshold:
            out = exit_sig.model_copy(
                update={
                    "strategy_id": self.strategy_id,
                    "metadata": {
                        **exit_sig.metadata,
                        "ensemble_mode": mode,
                        "ensemble_exit": True,
                    },
                }
            )
            self.record_signal(out)
            return [out]

        if pos_side:
            return []

        entries: list[tuple[str, float, float]] = []
        for raw, w in zip(per_child, self._weights, strict=True):
            d, st = _dominant_entry(raw)
            entries.append((d, st, w))

        close = float(bar["close"])

        if mode == "weighted":
            long_score = sum(st * w for d, st, w in entries if d == "long")
            short_score = sum(st * w for d, st, w in entries if d == "short")
            net = long_score - short_score
            if abs(net) < threshold:
                return []
            if net > 0:
                sig_type = SignalType.LONG_ENTRY
                strength = min(1.0, abs(net))
                reason = f"集成(weighted): net={net:.3f}"
            else:
                sig_type = SignalType.SHORT_ENTRY
                strength = min(1.0, abs(net))
                reason = f"集成(weighted): net={net:.3f}"
        else:
            w_long = sum(w for d, _st, w in entries if d == "long")
            w_short = sum(w for d, _st, w in entries if d == "short")
            if w_long == w_short:
                return []
            if w_long > w_short:
                margin = (w_long - w_short) / (w_long + w_short + 1e-9)
                if margin < threshold:
                    return []
                sig_type = SignalType.LONG_ENTRY
                strength = min(1.0, margin + 0.25)
                reason = f"集成(vote): long={w_long:.2f} short={w_short:.2f}"
            else:
                margin = (w_short - w_long) / (w_long + w_short + 1e-9)
                if margin < threshold:
                    return []
                sig_type = SignalType.SHORT_ENTRY
                strength = min(1.0, margin + 0.25)
                reason = f"集成(vote): short={w_short:.2f} long={w_long:.2f}"

        sig = Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            signal_type=sig_type,
            strength=round(strength, 4),
            price=close,
            reason=reason,
            metadata={
                "ensemble_mode": mode,
                "per_child": [
                    {"dir": d, "strength": round(st, 4)}
                    for d, st, _w in entries
                ],
            },
        )
        self.record_signal(sig)
        return [sig]

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        bar = market_data.get(self._trade_symbol)
        if not bar:
            return []
        return await self.on_bar(self._trade_symbol, bar)


def _load_child_strategy_modules_for_registry() -> None:
    from . import funding_rate_alpha as _funding  # noqa: F401
    from . import cross_sectional_momentum as _ts_mom  # noqa: F401
    from . import funding_meta_ensemble as _fund_meta  # noqa: F401
    from . import scalp_momentum as _scalp  # noqa: F401
    from . import volatility_breakout_scalp as _vol_scalp  # noqa: F401


_load_child_strategy_modules_for_registry()
