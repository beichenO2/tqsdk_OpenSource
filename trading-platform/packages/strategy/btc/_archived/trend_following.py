"""BTC 趋势跟随策略 - 基于多时间框架 EMA + ADX 趋势确认的顺势策略。"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..indicators import calc_atr, ema_update
from ..registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "ema_fast": 8,
    "ema_slow": 29,
    "ema_trend": 52,
    "adx_period": 15,
    "adx_threshold": 33.6,
    "atr_period": 14,
    "risk_per_trade_pct": 0.02,
    "trailing_stop_atr_mult": 2.12,
    "partial_take_profit_atr_mult": 5.43,
    "partial_close_pct": 0.41,
    "volume_ma_period": 20,
    "volume_confirm_ratio": 0.77,
}


_ema_update = ema_update


@auto_register("btc_trend_following")
class BTCTrendFollowingStrategy(BaseStrategy):
    """BTC 趋势跟随策略。

    核心逻辑:
    - EMA12/26 交叉定方向，EMA50 确认主趋势
    - ADX > 阈值时确认趋势强度，过滤震荡行情
    - ATR 动态止损 + 分批止盈
    """

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._ema_fast: dict[str, float | None] = {}
        self._ema_slow: dict[str, float | None] = {}
        self._ema_trend: dict[str, float | None] = {}

        self._high_history: dict[str, deque[float]] = {}
        self._low_history: dict[str, deque[float]] = {}
        self._close_history: dict[str, deque[float]] = {}

        self._prev_plus_dm: dict[str, float] = {}
        self._prev_minus_dm: dict[str, float] = {}
        self._prev_tr: dict[str, float] = {}
        self._adx: dict[str, float | None] = {}
        self._adx_history: dict[str, deque[float]] = {}

        self._bar_count: dict[str, int] = {}
        self._peak_price: dict[str, float] = {}
        self._trough_price: dict[str, float] = {}
        self._volume_history: dict[str, deque[float]] = {}

    def _ensure_buffers(self, symbol: str) -> None:
        buf_len = max(
            self.get_param("ema_trend"),
            self.get_param("adx_period"),
            self.get_param("atr_period"),
        ) + 10
        if symbol not in self._close_history:
            self._close_history[symbol] = deque(maxlen=buf_len)
            self._high_history[symbol] = deque(maxlen=buf_len)
            self._low_history[symbol] = deque(maxlen=buf_len)
            self._adx_history[symbol] = deque(maxlen=buf_len)
            self._volume_history[symbol] = deque(maxlen=buf_len)
            self._bar_count[symbol] = 0

    def _update_emas(self, symbol: str, close: float) -> None:
        self._ema_fast[symbol] = _ema_update(
            self._ema_fast.get(symbol), close, self.get_param("ema_fast")
        )
        self._ema_slow[symbol] = _ema_update(
            self._ema_slow.get(symbol), close, self.get_param("ema_slow")
        )
        self._ema_trend[symbol] = _ema_update(
            self._ema_trend.get(symbol), close, self.get_param("ema_trend")
        )

    def _update_adx(self, symbol: str) -> float | None:
        """Wilder smoothed ADX calculation."""
        period = self.get_param("adx_period")
        highs = list(self._high_history[symbol])
        lows = list(self._low_history[symbol])
        closes = list(self._close_history[symbol])

        if len(highs) < 2:
            return None

        high_diff = highs[-1] - highs[-2]
        low_diff = lows[-2] - lows[-1]
        plus_dm = max(high_diff, 0.0) if high_diff > low_diff else 0.0
        minus_dm = max(low_diff, 0.0) if low_diff > high_diff else 0.0

        tr = max(
            highs[-1] - lows[-1],
            abs(highs[-1] - closes[-2]),
            abs(lows[-1] - closes[-2]),
        )

        if symbol in self._prev_tr and self._prev_tr[symbol] > 0:
            smooth_tr = self._prev_tr[symbol] - self._prev_tr[symbol] / period + tr
            smooth_plus = self._prev_plus_dm[symbol] - self._prev_plus_dm[symbol] / period + plus_dm
            smooth_minus = self._prev_minus_dm[symbol] - self._prev_minus_dm[symbol] / period + minus_dm
        else:
            if self._bar_count[symbol] < period + 1:
                self._prev_tr[symbol] = self._prev_tr.get(symbol, 0) + tr
                self._prev_plus_dm[symbol] = self._prev_plus_dm.get(symbol, 0) + plus_dm
                self._prev_minus_dm[symbol] = self._prev_minus_dm.get(symbol, 0) + minus_dm
                return None
            smooth_tr = self._prev_tr[symbol]
            smooth_plus = self._prev_plus_dm[symbol]
            smooth_minus = self._prev_minus_dm[symbol]

        self._prev_tr[symbol] = smooth_tr
        self._prev_plus_dm[symbol] = smooth_plus
        self._prev_minus_dm[symbol] = smooth_minus

        if smooth_tr == 0:
            return None

        plus_di = 100 * smooth_plus / smooth_tr
        minus_di = 100 * smooth_minus / smooth_tr
        di_sum = plus_di + minus_di
        dx = 100 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0

        self._adx_history[symbol].append(dx)

        if len(self._adx_history[symbol]) < period:
            return None

        prev_adx = self._adx.get(symbol)
        if prev_adx is None:
            adx_val = sum(list(self._adx_history[symbol])[-period:]) / period
        else:
            adx_val = (prev_adx * (period - 1) + dx) / period

        self._adx[symbol] = adx_val
        return adx_val

    def _calc_atr(self, symbol: str) -> float | None:
        return calc_atr(
            self._high_history[symbol],
            self._low_history[symbol],
            self._close_history[symbol],
            self.get_param("atr_period"),
        )

    def _volume_confirmed(self, symbol: str) -> bool:
        """Check if current volume is above its moving average."""
        vol_ma_p = self.get_param("volume_ma_period")
        ratio = self.get_param("volume_confirm_ratio")
        vols = self._volume_history.get(symbol)
        if not vols or len(vols) < vol_ma_p:
            return True
        vol_ma = sum(list(vols)[-vol_ma_p:]) / vol_ma_p
        return vols[-1] >= vol_ma * ratio if vol_ma > 0 else True

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._ensure_buffers(symbol)
        close = bar["close"]
        high = bar["high"]
        low = bar["low"]
        volume = bar.get("volume", 0.0)

        self._close_history[symbol].append(close)
        self._high_history[symbol].append(high)
        self._low_history[symbol].append(low)
        self._volume_history[symbol].append(volume)
        self._bar_count[symbol] = self._bar_count.get(symbol, 0) + 1

        self._update_emas(symbol, close)
        adx = self._update_adx(symbol)
        atr = self._calc_atr(symbol)

        ema_f = self._ema_fast.get(symbol)
        ema_s = self._ema_slow.get(symbol)
        ema_t = self._ema_trend.get(symbol)

        if any(v is None for v in (ema_f, ema_s, ema_t, adx, atr)):
            return []

        signals: list[Signal] = []
        threshold = self.get_param("adx_threshold")
        trend_strong = adx >= threshold

        bullish_cross = ema_f > ema_s
        above_trend = close > ema_t
        bearish_cross = ema_f < ema_s
        below_trend = close < ema_t
        vol_ok = self._volume_confirmed(symbol)

        pos = self.get_position(symbol)

        if trend_strong and bullish_cross and above_trend and vol_ok and pos is None:
            strength = min((adx - threshold) / 30 + 0.4, 1.0)
            sig = Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_type=SignalType.LONG_ENTRY,
                strength=round(strength, 4),
                price=close,
                reason=(
                    f"趋势做多: EMA{self.get_param('ema_fast')}"
                    f">{self.get_param('ema_slow')}"
                    f", 价格>EMA{self.get_param('ema_trend')}"
                    f", ADX={adx:.1f}"
                ),
                metadata={
                    "ema_fast": ema_f, "ema_slow": ema_s, "ema_trend": ema_t,
                    "adx": adx, "atr": atr,
                },
            )
            signals.append(sig)
            self.record_signal(sig)
            self._peak_price[symbol] = close

        elif trend_strong and bearish_cross and below_trend and vol_ok and pos is None:
            strength = min((adx - threshold) / 30 + 0.4, 1.0)
            sig = Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_type=SignalType.SHORT_ENTRY,
                strength=round(strength, 4),
                price=close,
                reason=(
                    f"趋势做空: EMA{self.get_param('ema_fast')}"
                    f"<{self.get_param('ema_slow')}"
                    f", 价格<EMA{self.get_param('ema_trend')}"
                    f", ADX={adx:.1f}"
                ),
                metadata={
                    "ema_fast": ema_f, "ema_slow": ema_s, "ema_trend": ema_t,
                    "adx": adx, "atr": atr,
                },
            )
            signals.append(sig)
            self.record_signal(sig)
            self._trough_price[symbol] = close

        if pos is not None and atr > 0:
            stop_mult = self.get_param("trailing_stop_atr_mult")
            tp_mult = self.get_param("partial_take_profit_atr_mult")

            if pos.side.value == "buy":
                self._peak_price[symbol] = max(
                    self._peak_price.get(symbol, close), close
                )
                trailing_stop = self._peak_price[symbol] - atr * stop_mult

                if close < trailing_stop:
                    sig = Signal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=SignalType.LONG_EXIT,
                        strength=0.9,
                        price=close,
                        reason=f"趋势止损: 价格跌破追踪止损线 {trailing_stop:.2f}",
                        metadata={"trailing_stop": trailing_stop, "peak": self._peak_price[symbol]},
                    )
                    signals.append(sig)
                    self.record_signal(sig)
                    self._peak_price.pop(symbol, None)

                elif close >= pos.avg_price + atr * tp_mult:
                    sig = Signal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=SignalType.LONG_EXIT,
                        strength=0.5,
                        price=close,
                        suggested_qty=pos.qty * self.get_param("partial_close_pct"),
                        reason=f"趋势分批止盈: 利润达 {tp_mult}x ATR",
                        metadata={"profit_atr_mult": tp_mult},
                    )
                    signals.append(sig)
                    self.record_signal(sig)

            elif pos.side.value == "sell":
                self._trough_price[symbol] = min(
                    self._trough_price.get(symbol, close), close
                )
                trailing_stop = self._trough_price[symbol] + atr * stop_mult

                if close > trailing_stop:
                    sig = Signal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=SignalType.SHORT_EXIT,
                        strength=0.9,
                        price=close,
                        reason=f"趋势止损: 价格突破追踪止损线 {trailing_stop:.2f}",
                        metadata={"trailing_stop": trailing_stop, "trough": self._trough_price[symbol]},
                    )
                    signals.append(sig)
                    self.record_signal(sig)
                    self._trough_price.pop(symbol, None)

                elif close <= pos.avg_price - atr * tp_mult:
                    sig = Signal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=SignalType.SHORT_EXIT,
                        strength=0.5,
                        price=close,
                        suggested_qty=pos.qty * self.get_param("partial_close_pct"),
                        reason=f"趋势分批止盈: 利润达 {tp_mult}x ATR",
                        metadata={"profit_atr_mult": tp_mult},
                    )
                    signals.append(sig)
                    self.record_signal(sig)

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        all_signals: list[Signal] = []
        for symbol in self.config.symbols:
            bar = market_data.get(symbol)
            if bar:
                sigs = await self.on_bar(symbol, bar)
                all_signals.extend(sigs)
        return all_signals
