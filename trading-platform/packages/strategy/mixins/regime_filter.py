"""EMASlopeRegimeMixin — 轻量 EMA 斜率趋势方向过滤.

Root: whale_detector / keltner_channel / supertrend 在明显上涨行情里仍会产出
大量 SHORT（inbox/pos1/strategy-signal-bias）。加一个 EMA(period) 斜率判断：

  regime = 'up' if ema_slope > thresh, 'down' if < -thresh, 'flat' otherwise

用法：
    class MyStrategy(EMASlopeRegimeMixin, BaseStrategy):
        EMA_REGIME_PERIOD = 30
        EMA_REGIME_SLOPE_THRESH = 0.0005

        async def on_bar(...):
            self._regime_update(close)
            if signal_type == SignalType.SHORT_ENTRY and self._regime() == 'up':
                return []  # 压制逆势信号
"""

from __future__ import annotations

from collections import deque
from typing import Any


class EMASlopeRegimeMixin:
    """维护一条 EMA；用最近 5 bar 的斜率判 regime."""

    EMA_REGIME_PERIOD: int = 30
    EMA_REGIME_SLOPE_THRESH: float = 0.0005
    EMA_REGIME_WARMUP: int = 10

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._regime_ema: float | None = None
        self._regime_ema_hist: deque[float] = deque(maxlen=5)
        self._regime_bars_seen = 0

    def _regime_update(self, close: float) -> None:
        if close <= 0:
            return
        self._regime_bars_seen += 1
        alpha = 2.0 / (self.EMA_REGIME_PERIOD + 1.0)
        if self._regime_ema is None:
            self._regime_ema = close
        else:
            self._regime_ema = alpha * close + (1 - alpha) * self._regime_ema
        self._regime_ema_hist.append(self._regime_ema)

    def _regime(self) -> str:
        if self._regime_bars_seen < self.EMA_REGIME_WARMUP or len(self._regime_ema_hist) < 5 or not self._regime_ema:
            return "flat"
        first = self._regime_ema_hist[0]
        last = self._regime_ema_hist[-1]
        if first <= 0:
            return "flat"
        slope = (last - first) / first
        if slope > self.EMA_REGIME_SLOPE_THRESH:
            return "up"
        if slope < -self.EMA_REGIME_SLOPE_THRESH:
            return "down"
        return "flat"

    def _regime_allow(self, signal_type: Any) -> bool:
        """True 则允许此方向入场，False 则被 regime 过滤."""
        name = getattr(signal_type, "name", str(signal_type))
        regime = self._regime()
        if regime == "up" and name == "SHORT_ENTRY":
            return False
        if regime == "down" and name == "LONG_ENTRY":
            return False
        return True
