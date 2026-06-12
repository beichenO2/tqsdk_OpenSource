"""开盘冲击因子策略 — 捕捉开盘 15 分钟的价格动量与成交量爆发。

核心逻辑：
  1. 开盘跳空方向确认（前收盘→开盘价，大跳空=趋势延续信号）
  2. 开盘 15 分钟成交量爆发检测（与前日同时段对比）
  3. 成交量加权动量信号（VWAP slope 方向 + OI 变化确认）
  4. 集合竞价信息利用（开盘价 vs 前日收盘的隐含信息）

时段依赖：
  - FuturesSessionType.MORNING_OPEN (09:00-09:15)
  - FuturesSessionType.NIGHT_OPEN (21:00-21:15)
  - 仅在开盘时段生成入场信号，其他时段只管理持仓

Method: 经典技术分析（开盘区间突破）+ 行为金融学开盘效应
        (Amihud & Mendelson 1987 开盘价格发现, Stoll & Whaley 1990 日内模式)
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..indicators import calc_atr
from ..registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS: dict[str, Any] = {
    "opening_window_bars": 3,
    "gap_threshold_atr": 0.5,
    "gap_continuation_min": 0.3,
    "volume_surge_mult": 2.0,
    "momentum_bars": 3,
    "momentum_threshold": 0.003,
    "oi_confirmation": True,
    "oi_change_min": 0.0,
    "atr_period": 14,
    "tp_atr_mult": 2.5,
    "sl_atr_mult": 1.0,
    "max_hold_bars": 40,
    "cooldown_bars": 10,
    "trail_atr_mult": 1.5,
    "trail_activate_atr": 1.0,
    "max_daily_trades": 2,
}


@auto_register("opening_rush")
class OpeningRushStrategy(BaseStrategy):
    """开盘冲击因子策略 — 利用开盘时段的价格发现与成交量集中特征。"""

    def __init__(self, config: StrategyConfig) -> None:
        config = config.model_copy(
            update={"params": {**DEFAULT_PARAMS, **config.params}}
        )
        super().__init__(config)
        self._highs: deque[float] = deque(maxlen=300)
        self._lows: deque[float] = deque(maxlen=300)
        self._closes: deque[float] = deque(maxlen=300)
        self._volumes: deque[float] = deque(maxlen=300)
        self._ois: deque[float] = deque(maxlen=300)
        self._tp_volumes: deque[float] = deque(maxlen=300)
        self._bar_count = 0

        self._position_side: str | None = None
        self._entry_price = 0.0
        self._hold_bars = 0
        self._cd = 0
        self._trail_peak = 0.0

        self._prev_session_close: float | None = None
        self._session_open_price: float | None = None
        self._session_bars: list[dict] = []
        self._current_session_id: str = ""
        self._daily_trades = 0
        self._last_trade_date: str = ""

        self._prev_opening_volumes: deque[float] = deque(maxlen=20)

    def _detect_session_start(self, bar: dict) -> str | None:
        """Detect if this bar starts a new trading session. Returns session id or None."""
        dt = bar.get("datetime")
        if dt is None:
            return None
        try:
            import pandas as pd
            ts = pd.Timestamp(dt)
            h, m = ts.hour, ts.minute
            if h == 9 and m < 10:
                return f"day_{ts.strftime('%Y%m%d')}"
            if h == 21 and m < 10:
                return f"night_{ts.strftime('%Y%m%d')}"
        except Exception:
            pass
        return None

    def _is_opening_window(self) -> bool:
        window = self.get_param("opening_window_bars", 3)
        return len(self._session_bars) <= window and len(self._session_bars) > 0

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        h = float(bar.get("high", 0))
        l = float(bar.get("low", 0))
        c = float(bar.get("close", 0))
        v = float(bar.get("volume", 0))
        oi = float(bar.get("open_interest", 0))

        self._highs.append(h)
        self._lows.append(l)
        self._closes.append(c)
        self._volumes.append(v)
        self._ois.append(oi)
        self._tp_volumes.append(c * v)
        self._bar_count += 1

        session_id = self._detect_session_start(bar)
        if session_id and session_id != self._current_session_id:
            if self._session_bars:
                total_vol = sum(b.get("volume", 0) for b in self._session_bars[:self.get_param("opening_window_bars", 3)])
                if total_vol > 0:
                    self._prev_opening_volumes.append(total_vol)

            if self._closes and len(self._closes) >= 2:
                self._prev_session_close = list(self._closes)[-2]
            self._session_open_price = c
            self._session_bars = []
            self._current_session_id = session_id

            dt = bar.get("datetime")
            if dt is not None:
                try:
                    import pandas as pd
                    date_str = pd.Timestamp(dt).strftime('%Y%m%d')
                    if date_str != self._last_trade_date:
                        self._daily_trades = 0
                        self._last_trade_date = date_str
                except Exception:
                    pass

        if self._current_session_id:
            self._session_bars.append(bar)

        if self._cd > 0:
            self._cd -= 1

        if self._bar_count < 30:
            return []

        atr = calc_atr(list(self._highs), list(self._lows), list(self._closes),
                       self.get_param("atr_period", 14))
        if atr is None or atr < 1e-10:
            return []

        signals = []

        if self._position_side:
            self._hold_bars += 1
            max_hold = self.get_param("max_hold_bars", 40)
            tp_mult = self.get_param("tp_atr_mult", 2.5)
            sl_mult = self.get_param("sl_atr_mult", 1.0)
            trail_mult = self.get_param("trail_atr_mult", 1.5)
            trail_activate = self.get_param("trail_activate_atr", 1.0)

            if self._position_side == "long":
                pnl = (c - self._entry_price) / self._entry_price
                self._trail_peak = max(self._trail_peak, h)
                trail_active = (self._trail_peak - self._entry_price) >= trail_activate * atr
                trail_hit = trail_active and l <= (self._trail_peak - trail_mult * atr)
            else:
                pnl = (self._entry_price - c) / self._entry_price
                self._trail_peak = min(self._trail_peak, l)
                trail_active = (self._entry_price - self._trail_peak) >= trail_activate * atr
                trail_hit = trail_active and h >= (self._trail_peak + trail_mult * atr)

            tp_hit = pnl >= tp_mult * atr / max(self._entry_price, 1e-10)
            sl_hit = pnl <= -sl_mult * atr / max(self._entry_price, 1e-10)

            if sl_hit or tp_hit or self._hold_bars >= max_hold or trail_hit:
                exit_type = SignalType.LONG_EXIT if self._position_side == "long" else SignalType.SHORT_EXIT
                reason = "tp" if tp_hit else "sl" if sl_hit else "trail" if trail_hit else "max_hold"
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_type, strength=0.8, price=c,
                    metadata={"reason": reason, "pnl": pnl, "hold_bars": self._hold_bars},
                )
                signals.append(sig)
                self._position_side = None
                self._cd = self.get_param("cooldown_bars", 10)
                return signals
            return signals

        if self._position_side is not None or self._cd > 0:
            return signals

        max_daily = self.get_param("max_daily_trades", 2)
        if self._daily_trades >= max_daily:
            return signals

        if not self._is_opening_window():
            return signals

        if self._session_open_price is None or self._prev_session_close is None:
            return signals

        gap = (self._session_open_price - self._prev_session_close) / max(atr, 1e-10)
        gap_threshold = self.get_param("gap_threshold_atr", 0.5)

        n_bars = len(self._session_bars)
        if n_bars < 2:
            return signals

        opening_prices = [float(b.get("close", 0)) for b in self._session_bars]
        opening_momentum = (opening_prices[-1] - opening_prices[0]) / max(atr, 1e-10)

        opening_vol = sum(float(b.get("volume", 0)) for b in self._session_bars)
        vol_surge = False
        if self._prev_opening_volumes:
            avg_prev_vol = np.mean(list(self._prev_opening_volumes))
            vol_surge = avg_prev_vol > 0 and opening_vol / avg_prev_vol >= self.get_param("volume_surge_mult", 2.0)
        else:
            vol_avg = np.mean(list(self._volumes)[-60:]) if len(self._volumes) >= 60 else v
            vol_surge = vol_avg > 0 and opening_vol / max(n_bars, 1) >= vol_avg * self.get_param("volume_surge_mult", 2.0)

        oi_ok = True
        if self.get_param("oi_confirmation", True) and len(self._ois) >= 2:
            oi_delta = self._ois[-1] - self._ois[-2]
            oi_ok = oi_delta >= self.get_param("oi_change_min", 0.0)

        momentum_threshold = self.get_param("momentum_threshold", 0.003)
        close_list = list(self._closes)
        if len(close_list) >= self.get_param("momentum_bars", 3) + 1:
            mb = self.get_param("momentum_bars", 3)
            mom = (close_list[-1] - close_list[-mb - 1]) / max(close_list[-mb - 1], 1e-10)
        else:
            mom = 0.0

        gap_continuation = self.get_param("gap_continuation_min", 0.3)

        is_bullish = (
            gap > gap_threshold
            and opening_momentum > gap_continuation
            and vol_surge
            and oi_ok
            and mom > momentum_threshold
        )

        is_bearish = (
            gap < -gap_threshold
            and opening_momentum < -gap_continuation
            and vol_surge
            and oi_ok
            and mom < -momentum_threshold
        )

        if is_bullish:
            strength = min(1.0, 0.5 + abs(gap) * 0.1 + abs(opening_momentum) * 0.1)
            sig = Signal(
                strategy_id=self.strategy_id, symbol=symbol,
                signal_type=SignalType.LONG_ENTRY, strength=strength, price=c,
                metadata={
                    "gap_atr": round(gap, 2),
                    "opening_momentum": round(opening_momentum, 2),
                    "volume_surge_ratio": round(opening_vol / max(np.mean(list(self._prev_opening_volumes)) if self._prev_opening_volumes else 1, 1e-10), 2),
                    "session": self._current_session_id,
                },
            )
            signals.append(sig)
            self._position_side = "long"
            self._entry_price = c
            self._hold_bars = 0
            self._trail_peak = h
            self._daily_trades += 1

        elif is_bearish:
            strength = min(1.0, 0.5 + abs(gap) * 0.1 + abs(opening_momentum) * 0.1)
            sig = Signal(
                strategy_id=self.strategy_id, symbol=symbol,
                signal_type=SignalType.SHORT_ENTRY, strength=strength, price=c,
                metadata={
                    "gap_atr": round(gap, 2),
                    "opening_momentum": round(opening_momentum, 2),
                    "volume_surge_ratio": round(opening_vol / max(np.mean(list(self._prev_opening_volumes)) if self._prev_opening_volumes else 1, 1e-10), 2),
                    "session": self._current_session_id,
                },
            )
            signals.append(sig)
            self._position_side = "short"
            self._entry_price = c
            self._hold_bars = 0
            self._trail_peak = l
            self._daily_trades += 1

        return signals
