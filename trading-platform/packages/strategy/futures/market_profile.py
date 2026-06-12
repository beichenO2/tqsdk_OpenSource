"""市场剖面图策略 — TPO (Time-Price-Opportunity) 分析。

理论：Market Profile (J. Peter Steidlmayer, 1980s, CBOT, 经典)
  - 价格-时间分布分析：找到 Value Area（70% 时间分布区域）
  - POC (Point of Control)：最多时间停留的价格 = "公平价格"
  - 价格在 VA 外 → 趋势/突破信号
  - 价格回到 POC → 均值回归信号

日内适配：
  - 用 5min K 线构建当日 TPO 分布
  - 实时更新 POC/VA 边界
  - VA High/Low 作为日内支撑/阻力
"""

from __future__ import annotations

import logging
from collections import Counter, deque
from typing import Any

import numpy as np

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..indicators import calc_atr
from ..registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS: dict[str, Any] = {
    "profile_bins": 30,
    "va_pct": 0.70,
    "breakout_confirm_bars": 2,
    "atr_period": 14,
    "tp_atr_mult": 2.0,
    "sl_atr_mult": 1.0,
    "max_hold_bars": 30,
    "cooldown_bars": 3,
}


class IntradayProfile:
    """实时市场剖面图计算器。"""

    def __init__(self, n_bins: int = 30, va_pct: float = 0.70):
        self.n_bins = n_bins
        self.va_pct = va_pct
        self.prices: list[float] = []
        self.poc: float = 0.0
        self.va_high: float = 0.0
        self.va_low: float = 0.0

    def update(self, close: float) -> None:
        self.prices.append(close)
        if len(self.prices) < 10:
            return
        self._compute()

    def _compute(self) -> None:
        prices = np.array(self.prices)
        p_min, p_max = prices.min(), prices.max()
        if p_max - p_min < 1e-10:
            self.poc = p_min
            self.va_high = p_max
            self.va_low = p_min
            return

        bin_edges = np.linspace(p_min, p_max, self.n_bins + 1)
        counts, _ = np.histogram(prices, bins=bin_edges)

        poc_idx = np.argmax(counts)
        self.poc = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2

        total = len(prices)
        target = int(total * self.va_pct)
        lo, hi = poc_idx, poc_idx
        accumulated = counts[poc_idx]

        while accumulated < target and (lo > 0 or hi < len(counts) - 1):
            expand_lo = counts[lo - 1] if lo > 0 else 0
            expand_hi = counts[hi + 1] if hi < len(counts) - 1 else 0

            if expand_lo >= expand_hi and lo > 0:
                lo -= 1
                accumulated += counts[lo]
            elif hi < len(counts) - 1:
                hi += 1
                accumulated += counts[hi]
            else:
                lo -= 1
                accumulated += counts[lo]

        self.va_low = bin_edges[lo]
        self.va_high = bin_edges[hi + 1]

    def reset(self) -> None:
        self.prices.clear()
        self.poc = 0.0
        self.va_high = 0.0
        self.va_low = 0.0


@auto_register("market_profile")
class MarketProfileStrategy(BaseStrategy):
    """市场剖面图策略 — POC/VA 边界信号。"""

    def __init__(self, config: StrategyConfig) -> None:
        config = config.model_copy(
            update={"params": {**DEFAULT_PARAMS, **config.params}}
        )
        super().__init__(config)
        self._profile = IntradayProfile(
            n_bins=self.get_param("profile_bins", 30),
            va_pct=self.get_param("va_pct", 0.70),
        )
        self._closes: deque[float] = deque(maxlen=200)
        self._highs: deque[float] = deque(maxlen=200)
        self._lows: deque[float] = deque(maxlen=200)
        self._bar_count = 0
        self._position_side: str | None = None
        self._entry_price = 0.0
        self._hold_bars = 0
        self._cd = 0
        self._breakout_count = 0
        self._last_session_reset = -1

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        h = float(bar.get("high", 0))
        l = float(bar.get("low", 0))
        c = float(bar.get("close", 0))

        self._highs.append(h)
        self._lows.append(l)
        self._closes.append(c)
        self._bar_count += 1

        dt = bar.get("datetime")
        if dt is not None:
            try:
                import pandas as pd
                ts = pd.Timestamp(dt)
                if ts.hour == 9 and ts.minute < 10 and self._bar_count - self._last_session_reset > 5:
                    self._profile.reset()
                    self._last_session_reset = self._bar_count
                    self._breakout_count = 0
            except Exception:
                pass

        self._profile.update(c)

        if self._bar_count < 20 or self._profile.poc == 0:
            return []

        atr = calc_atr(list(self._highs), list(self._lows), list(self._closes),
                       self.get_param("atr_period", 14))
        if atr is None or atr < 1e-10:
            return []

        signals = []

        if self._position_side:
            self._hold_bars += 1
            max_hold = self.get_param("max_hold_bars", 30)
            tp_mult = self.get_param("tp_atr_mult", 2.0)
            sl_mult = self.get_param("sl_atr_mult", 1.0)

            pnl = (c - self._entry_price) / self._entry_price if self._position_side == "long" else (self._entry_price - c) / self._entry_price
            if pnl >= tp_mult * atr / self._entry_price or pnl <= -sl_mult * atr / self._entry_price or self._hold_bars >= max_hold:
                exit_type = SignalType.LONG_EXIT if self._position_side == "long" else SignalType.SHORT_EXIT
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_type, strength=0.8, price=c,
                    reason=f"mp_exit: hold={self._hold_bars}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = None
                self._hold_bars = 0
                self._cd = self.get_param("cooldown_bars", 3)
                return signals

        if self._cd > 0:
            self._cd -= 1

        confirm_bars = self.get_param("breakout_confirm_bars", 2)

        if not self._position_side and self._cd <= 0:
            if c > self._profile.va_high:
                self._breakout_count += 1
                if self._breakout_count >= confirm_bars:
                    sig = Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.LONG_ENTRY,
                        strength=min((c - self._profile.va_high) / atr, 1.0), price=c,
                        reason=f"mp_breakout_up: c={c:.1f} VA=[{self._profile.va_low:.1f},{self._profile.va_high:.1f}] POC={self._profile.poc:.1f}",
                    )
                    signals.append(sig)
                    self.record_signal(sig)
                    self._position_side = "long"
                    self._entry_price = c
                    self._hold_bars = 0
                    self._breakout_count = 0

            elif c < self._profile.va_low:
                self._breakout_count += 1
                if self._breakout_count >= confirm_bars:
                    sig = Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.SHORT_ENTRY,
                        strength=min((self._profile.va_low - c) / atr, 1.0), price=c,
                        reason=f"mp_breakout_down: c={c:.1f} VA=[{self._profile.va_low:.1f},{self._profile.va_high:.1f}] POC={self._profile.poc:.1f}",
                    )
                    signals.append(sig)
                    self.record_signal(sig)
                    self._position_side = "short"
                    self._entry_price = c
                    self._hold_bars = 0
                    self._breakout_count = 0
            else:
                self._breakout_count = 0

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        return []
