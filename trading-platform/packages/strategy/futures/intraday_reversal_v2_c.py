"""intraday_reversal_v2_c — A (紧止损) + B (vol filter) 叠加变体.

2号位 gate: 假说：A（紧止损让单笔 std 缩）+ B（vol filter 跳过高波动入场）
同时生效 → 整体 std 双重压制，期望 sharpe 从 0.45 跳到 0.8+。

Params delta vs base:
  - sl_atr_mult: 1.0 → 0.6          (from A)
  - tp_atr_mult: 1.5 → 1.1          (略宽于 A，让赢利有空间)
  - max_hold_bars: 20 → 15          (折中)
  - vol_regime_mult: 1.5            (from B)
"""

from __future__ import annotations

from typing import Any

from ..registry import auto_register
from .intraday_reversal_v2_b import IntradayReversalV2BStrategy

PARAMS_V2_C: dict[str, Any] = {
    "sl_atr_mult": 0.6,
    "tp_atr_mult": 1.1,
    "max_hold_bars": 15,
    "vol_regime_window": 20,
    "vol_regime_mult": 1.5,
}


@auto_register("intraday_reversal_v2_c")
class IntradayReversalV2CStrategy(IntradayReversalV2BStrategy):
    """变体 C：A+B 合并 — A 的紧止损 + B 的 vol filter."""

    def __init__(self, config) -> None:
        merged = config.model_copy(update={"params": {**PARAMS_V2_C, **config.params}})
        super().__init__(merged)
