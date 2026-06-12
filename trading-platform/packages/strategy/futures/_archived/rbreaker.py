"""R-Breaker 日内策略 — 前一日 OHLC 计算六档 pivot，趋势突破与反转信号。"""

from __future__ import annotations

import logging
from collections import deque
from datetime import date, datetime
from typing import Any

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..indicators import calc_atr
from ..registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "f1": 0.35,
    "f2": 0.07,
    "f3": 0.25,
    "position_size": 1.0,
    "atr_period": 14,
    "trailing_stop_atr_mult": 2.0,
}


def _bar_calendar_day(bar: dict[str, Any]) -> date | None:
    """从 K 线字典解析交易日（自然日）。"""
    raw = bar.get("datetime") or bar.get("date")
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    if isinstance(raw, (int, float)):
        # 秒级时间戳
        try:
            return datetime.utcfromtimestamp(float(raw)).date()
        except (OSError, OverflowError, ValueError):
            return None
    if isinstance(raw, str):
        s = raw.strip()
        if len(s) >= 10:
            try:
                return date.fromisoformat(s[:10])
            except ValueError:
                pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(s[:19], fmt).date()
            except (ValueError, IndexError):
                continue
    return None


def _calc_rbreaker_levels(
    high: float, low: float, close: float, f1: float, f2: float, f3: float
) -> dict[str, float]:
    """按指定公式计算 R-Breaker 价位。f2/f3 预留扩展，与经典变体兼容。"""
    _ = f2, f3  # 当前公式仅使用 f1；保留参数供配置对齐
    pivot = (high + low + close) / 3.0
    rng = high - low
    ssetup = pivot + rng * f1
    senter = 2.0 * pivot - low
    benter = 2.0 * pivot - high
    bsetup = pivot - rng * f1
    sbreak = bsetup - f1 * (benter - bsetup)
    bbreak = ssetup + f1 * (ssetup - senter)
    return {
        "pivot": pivot,
        "ssetup": ssetup,
        "senter": senter,
        "benter": benter,
        "bsetup": bsetup,
        "sbreak": sbreak,
        "bbreak": bbreak,
    }


@auto_register("rbreaker")
class RBreakerStrategy(BaseStrategy):
    """R-Breaker 日内模型。

    - 使用前一日完整 OHLC 生成 pivot 六档
    - 趋势：向上突破 bbreak 做多，向下跌破 sbreak 做空
    - 反转：曾触及 ssetup 后回落至 senter 下方做多；曾触及 bsetup 后反弹至 benter 上方做空
    - 持仓 ATR 追踪止损
    """

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._close_history: dict[str, deque[float]] = {}
        self._high_history: dict[str, deque[float]] = {}
        self._low_history: dict[str, deque[float]] = {}

        self._prev_day: dict[str, date | None] = {}
        self._day_open: dict[str, float | None] = {}
        self._day_high: dict[str, float] = {}
        self._day_low: dict[str, float] = {}
        self._day_close: dict[str, float] = {}
        self._last_ohlc: dict[str, tuple[float, float, float] | None] = {}

        self._touched_ssetup: dict[str, bool] = {}
        self._touched_bsetup: dict[str, bool] = {}
        self._peak_price: dict[str, float] = {}
        self._trough_price: dict[str, float] = {}

    def _ensure_buffers(self, symbol: str) -> None:
        atr_p = int(self.get_param("atr_period")) + 15
        if symbol not in self._close_history:
            self._close_history[symbol] = deque(maxlen=atr_p)
            self._high_history[symbol] = deque(maxlen=atr_p)
            self._low_history[symbol] = deque(maxlen=atr_p)

    def _calc_atr(self, symbol: str) -> float | None:
        return calc_atr(
            self._high_history[symbol],
            self._low_history[symbol],
            self._close_history[symbol],
            int(self.get_param("atr_period")),
        )

    def _roll_session(self, symbol: str, day: date, o: float, h: float, l: float, c: float) -> None:
        """在新交易日开始时把上一日 OHLC 写入 last_ohlc。"""
        prev = self._prev_day.get(symbol)
        if prev is not None and day != prev:
            dh = self._day_high.get(symbol)
            dl = self._day_low.get(symbol)
            dc = self._day_close.get(symbol)
            if dh is not None and dl is not None and dc is not None and dh >= dl:
                self._last_ohlc[symbol] = (float(dh), float(dl), float(dc))
                logger.debug("[%s] 更新前日 OHLC H=%.4f L=%.4f C=%.4f", symbol, dh, dl, dc)

        if prev is None or day != prev:
            self._day_open[symbol] = o
            self._day_high[symbol] = h
            self._day_low[symbol] = l
            self._day_close[symbol] = c
            self._prev_day[symbol] = day
            self._touched_ssetup[symbol] = False
            self._touched_bsetup[symbol] = False
        else:
            self._day_high[symbol] = max(self._day_high.get(symbol, h), h)
            self._day_low[symbol] = min(self._day_low.get(symbol, l), l)
            self._day_close[symbol] = c

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        close = bar.get("close")
        high = bar.get("high")
        low = bar.get("low")
        open_ = bar.get("open", close)
        if close is None or high is None or low is None:
            return []

        fc = float(close)
        fh = float(high)
        fl = float(low)
        fo = float(open_) if open_ is not None else fc

        day = _bar_calendar_day(bar)
        if day is None:
            logger.warning("[%s] K 线缺少 datetime/date，R-Breaker 无法分日，跳过", symbol)
            return []

        self._ensure_buffers(symbol)
        self._roll_session(symbol, day, fo, fh, fl, fc)

        self._close_history[symbol].append(fc)
        self._high_history[symbol].append(fh)
        self._low_history[symbol].append(fl)

        ohlc = self._last_ohlc.get(symbol)
        signals: list[Signal] = []

        f1 = float(self.get_param("f1"))
        f2 = float(self.get_param("f2"))
        f3 = float(self.get_param("f3"))
        pos_sz = float(self.get_param("position_size") or 1.0)
        if pos_sz <= 0:
            pos_sz = 1.0

        levels: dict[str, float] | None = None
        if ohlc is not None:
            ph, pl, pc = ohlc
            if ph > pl and ph > 0:
                levels = _calc_rbreaker_levels(ph, pl, pc, f1, f2, f3)

        if levels:
            if fh >= levels["ssetup"]:
                self._touched_ssetup[symbol] = True
            if fl <= levels["bsetup"]:
                self._touched_bsetup[symbol] = True

        pos = self.get_position(symbol)
        atr = self._calc_atr(symbol)

        if levels and pos is None:
            bbreak = levels["bbreak"]
            sbreak = levels["sbreak"]
            levels["ssetup"]
            senter = levels["senter"]
            levels["bsetup"]
            benter = levels["benter"]

            trend_long = fh > bbreak
            trend_short = fl < sbreak
            rev_long = self._touched_ssetup.get(symbol, False) and fl < senter
            rev_short = self._touched_bsetup.get(symbol, False) and fh > benter

            if trend_long:
                sig = Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=0.85,
                    price=fc,
                    suggested_qty=pos_sz,
                    reason=f"R-Breaker 趋势做多 突破 bbreak={bbreak:.4f}",
                    metadata={**levels, "mode": "trend_long"},
                )
                signals.append(sig)
                self.record_signal(sig)
                self._peak_price[symbol] = fc
                logger.info("[%s] %s", symbol, sig.reason)
            elif trend_short:
                sig = Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=0.85,
                    price=fc,
                    suggested_qty=pos_sz,
                    reason=f"R-Breaker 趋势做空 跌破 sbreak={sbreak:.4f}",
                    metadata={**levels, "mode": "trend_short"},
                )
                signals.append(sig)
                self.record_signal(sig)
                self._trough_price[symbol] = fc
                logger.info("[%s] %s", symbol, sig.reason)
            elif rev_long and not trend_short:
                sig = Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=0.75,
                    price=fc,
                    suggested_qty=pos_sz,
                    reason="R-Breaker 反转做多 曾达 ssetup 后跌破 senter",
                    metadata={**levels, "mode": "reversal_long"},
                )
                signals.append(sig)
                self.record_signal(sig)
                self._peak_price[symbol] = fc
                logger.info("[%s] %s", symbol, sig.reason)
            elif rev_short and not trend_long:
                sig = Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=0.75,
                    price=fc,
                    suggested_qty=pos_sz,
                    reason="R-Breaker 反转做空 曾达 bsetup 后突破 benter",
                    metadata={**levels, "mode": "reversal_short"},
                )
                signals.append(sig)
                self.record_signal(sig)
                self._trough_price[symbol] = fc
                logger.info("[%s] %s", symbol, sig.reason)

        if pos is not None and atr is not None and atr > 0:
            stop_mult = float(self.get_param("trailing_stop_atr_mult") or 2.0)
            if pos.side.value == "buy":
                self._peak_price[symbol] = max(self._peak_price.get(symbol, fc), fc)
                trail = self._peak_price[symbol] - atr * stop_mult
                if fc < trail:
                    sig = Signal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=SignalType.LONG_EXIT,
                        strength=0.9,
                        price=fc,
                        reason=f"ATR 追踪止损 trail={trail:.4f}",
                        metadata={"trailing_stop": trail},
                    )
                    signals.append(sig)
                    self.record_signal(sig)
                    self._peak_price.pop(symbol, None)
                    logger.info("[%s] 平多 %s", symbol, sig.reason)
            elif pos.side.value == "sell":
                self._trough_price[symbol] = min(self._trough_price.get(symbol, fc), fc)
                trail = self._trough_price[symbol] + atr * stop_mult
                if fc > trail:
                    sig = Signal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=SignalType.SHORT_EXIT,
                        strength=0.9,
                        price=fc,
                        reason=f"ATR 追踪止损 trail={trail:.4f}",
                        metadata={"trailing_stop": trail},
                    )
                    signals.append(sig)
                    self.record_signal(sig)
                    self._trough_price.pop(symbol, None)
                    logger.info("[%s] 平空 %s", symbol, sig.reason)

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        all_signals: list[Signal] = []
        for symbol in self.config.symbols:
            bar = market_data.get(symbol)
            if bar:
                sigs = await self.on_bar(symbol, bar)
                all_signals.extend(sigs)
        return all_signals
