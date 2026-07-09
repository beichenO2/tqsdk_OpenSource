"""intraday_reversal_v2_b — volatility-regime filter variant.

2号位 gate: 原 sharpe 0.45 是因为回报波动大。假说：在高波动时段（ATR > 1.5×
20-bar ATR 均值）直接不进场，避开最吃波动的 20-30% 交易，把 std 砍下去。

Implementation delta vs base:
  - 维护 ATR(14) 20-bar 滚窗；若当前 ATR > 1.5× 均值则跳过入场（exit 不受影响）
  - 保留原止损/止盈参数
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from ..base import Signal, SignalType
from ..registry import auto_register
from .intraday_reversal import DEFAULT_PARAMS as _BASE_PARAMS, IntradayReversalStrategy

PARAMS_V2_B: dict[str, Any] = {
    **_BASE_PARAMS,
    "vol_regime_window": 20,
    "vol_regime_mult": 1.5,
}


@auto_register("intraday_reversal_v2_b")
class IntradayReversalV2BStrategy(IntradayReversalStrategy):
    """变体 B：高波动时段抑制入场；exit 逻辑不变."""

    def __init__(self, config) -> None:
        merged = config.model_copy(update={"params": {**PARAMS_V2_B, **config.params}})
        super().__init__(merged)
        self._recent_atrs: deque[float] = deque(maxlen=merged.params.get("vol_regime_window", 20))

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        signals = await super().on_bar(symbol, bar)
        if not signals:
            return signals

        from ..indicators import calc_atr
        atr = calc_atr(list(self._highs), list(self._lows), list(self._closes),
                       self.get_param("atr_period", 14))
        if atr is None or atr < 1e-10:
            return signals

        self._recent_atrs.append(atr)
        window = self.get_param("vol_regime_window", 20)
        if len(self._recent_atrs) < window:
            return signals

        vol_mean = float(np.mean(self._recent_atrs))
        vol_mult = self.get_param("vol_regime_mult", 1.5)
        if vol_mean <= 0:
            return signals

        if atr <= vol_mult * vol_mean:
            return signals

        filtered: list[Signal] = []
        for sig in signals:
            if sig.signal_type in (SignalType.LONG_ENTRY, SignalType.SHORT_ENTRY):
                # 高波动：拒绝入场，回滚内部状态
                if self._position_side is not None:
                    self._position_side = None
                    self._entry_price = 0.0
                    self._hold_bars = 0
            else:
                filtered.append(sig)
        return filtered
