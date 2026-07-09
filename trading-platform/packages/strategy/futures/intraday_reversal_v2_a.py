"""intraday_reversal_v2_a — ATR-normalized tight stops variant.

2号位 gate: 原 intraday_reversal OOS sharpe=0.45 就差 0.354 到阈值，回报 +83.8%
但 per-trade 波动太大。假说：收紧 sl_mult (1.0→0.6) + tp_mult (1.5→1.0) 让单笔
赢/输幅度双压缩，std 下降幅度快于 mean，sharpe 上升。

Params delta vs base:
  - sl_atr_mult: 1.0 → 0.6
  - tp_atr_mult: 1.5 → 1.0
  - max_hold_bars: 20 → 12
"""

from __future__ import annotations

from typing import Any

from ..registry import auto_register
from .intraday_reversal import DEFAULT_PARAMS as _BASE_PARAMS, IntradayReversalStrategy

PARAMS_V2_A: dict[str, Any] = {
    **_BASE_PARAMS,
    "sl_atr_mult": 0.6,
    "tp_atr_mult": 1.0,
    "max_hold_bars": 12,
}


@auto_register("intraday_reversal_v2_a")
class IntradayReversalV2AStrategy(IntradayReversalStrategy):
    """变体 A：ATR 归一化紧止损 + 紧止盈 + 短持仓."""

    def __init__(self, config) -> None:
        merged = config.model_copy(update={"params": {**PARAMS_V2_A, **config.params}})
        super().__init__(merged)
