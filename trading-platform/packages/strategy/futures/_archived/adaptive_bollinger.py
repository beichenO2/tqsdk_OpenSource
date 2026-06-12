"""自适应布林均值回归 — 基于波动率状态动态调整参数。

研究来源:
- IEEE: "A learning adaptive Bollinger band system" — 动态 period/std
- TradingView: Adaptive Bollinger Bands — 波动率归一化
- FMZ: Multi-Regime Adaptive Strategy — ADX 趋势过滤
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..registry import auto_register

DEFAULT_PARAMS = {
    "bb_period_base": 20,
    "bb_period_min": 12,
    "bb_period_max": 30,
    "bb_num_std_base": 2.0,
    "bb_num_std_min": 1.5,
    "bb_num_std_max": 2.8,
    "vol_lookback": 50,
    "rsi_period": 14,
    "rsi_oversold": 32.0,
    "rsi_overbought": 68.0,
    "atr_period": 14,
    "atr_stop_mult": 2.0,
    "adx_period": 14,
    "adx_trend_threshold": 45.0,
    "max_hold_bars": 120,
    "min_bars": 30,
}


@auto_register("adaptive_bollinger")
class AdaptiveBollingerStrategy(BaseStrategy):
    """波动率自适应布林均值回归:
    - 高波动: 短 period + 宽 bands → 更快响应
    - 低波动: 长 period + 窄 bands → 过滤噪声
    - ADX 趋势过滤: ADX > 阈值时暂停均值回归 (趋势市场不适合)
    - ATR 追踪止损: 防止极端行情吃掉利润
    - 持仓时间限制: 控制交易频率
    """

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)
        self._close: dict[str, deque[float]] = {}
        self._high: dict[str, deque[float]] = {}
        self._low: dict[str, deque[float]] = {}
        self._hold_bars: dict[str, int] = {}
        self._entry_atr: dict[str, float] = {}

    def _ensure(self, symbol: str) -> None:
        mx = int(self.get_param("vol_lookback")) + int(self.get_param("bb_period_max")) + 20
        if symbol not in self._close:
            self._close[symbol] = deque(maxlen=mx)
            self._high[symbol] = deque(maxlen=mx)
            self._low[symbol] = deque(maxlen=mx)

    @staticmethod
    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals)

    def _vol_percentile(self, closes: list[float]) -> float:
        """Compute current volatility as percentile of historical range (0-1)."""
        lb = int(self.get_param("vol_lookback"))
        if len(closes) < lb + 2:
            return 0.5
        returns = [closes[i] / closes[i - 1] - 1 for i in range(-lb, 0)]
        current_vol = math.sqrt(sum(r * r for r in returns[-10:]) / 10)
        all_vols = []
        for j in range(lb - 10):
            window = returns[j:j + 10]
            all_vols.append(math.sqrt(sum(r * r for r in window) / len(window)))
        if not all_vols:
            return 0.5
        all_vols.sort()
        rank = sum(1 for v in all_vols if v <= current_vol) / len(all_vols)
        return max(0.0, min(1.0, rank))

    def _adaptive_params(self, vol_pct: float) -> tuple[int, float]:
        """Map volatility percentile to dynamic bb_period and bb_num_std."""
        p_min = int(self.get_param("bb_period_min"))
        p_max = int(self.get_param("bb_period_max"))
        s_min = float(self.get_param("bb_num_std_min"))
        s_max = float(self.get_param("bb_num_std_max"))
        period = int(p_max - (p_max - p_min) * vol_pct)
        num_std = s_min + (s_max - s_min) * vol_pct
        return max(p_min, min(p_max, period)), max(s_min, min(s_max, num_std))

    def _bbands(self, closes: list[float], n: int, k: float) -> tuple[float, float, float] | None:
        if len(closes) < n:
            return None
        w = closes[-n:]
        mid = self._mean(w)
        var = sum((x - mid) ** 2 for x in w) / max(len(w) - 1, 1)
        sd = math.sqrt(max(var, 0.0))
        return mid - k * sd, mid, mid + k * sd

    def _rsi(self, closes: list[float], n: int) -> float | None:
        if len(closes) < n + 1:
            return None
        gains = losses = 0.0
        for i in range(-n, 0):
            d = closes[i] - closes[i - 1]
            if d >= 0:
                gains += d
            else:
                losses -= d
        ag, al = gains / n, losses / n
        if al == 0:
            return 100.0 if ag > 0 else 50.0
        return 100.0 - 100.0 / (1.0 + ag / al)

    def _atr(self, symbol: str) -> float | None:
        n = int(self.get_param("atr_period"))
        h, l, c = list(self._high[symbol]), list(self._low[symbol]), list(self._close[symbol])
        if len(h) < n + 1:
            return None
        trs = [max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])) for i in range(-n, 0)]
        return sum(trs) / len(trs)

    def _adx(self, symbol: str) -> float | None:
        n = int(self.get_param("adx_period"))
        h, l, c = list(self._high[symbol]), list(self._low[symbol]), list(self._close[symbol])
        if len(h) < 2 * n + 1:
            return None
        plus_dm = []
        minus_dm = []
        tr_list = []
        for i in range(-2 * n, 0):
            up = h[i] - h[i - 1]
            down = l[i - 1] - l[i]
            plus_dm.append(max(up, 0) if up > down else 0)
            minus_dm.append(max(down, 0) if down > up else 0)
            tr_list.append(max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])))
        sp = sum(plus_dm[:n])
        sm = sum(minus_dm[:n])
        st = sum(tr_list[:n])
        dx_vals: list[float] = []
        for i in range(n, len(plus_dm)):
            sp = sp - sp / n + plus_dm[i]
            sm = sm - sm / n + minus_dm[i]
            st = st - st / n + tr_list[i]
            if st <= 0:
                continue
            pdi = 100 * sp / st
            mdi = 100 * sm / st
            di_sum = pdi + mdi
            dx_vals.append(100 * abs(pdi - mdi) / di_sum if di_sum > 0 else 0)
        if not dx_vals:
            return None
        return sum(dx_vals[-n:]) / min(len(dx_vals), n)

    def _hold_signal(self, symbol: str, price: float | None, reason: str) -> Signal:
        return Signal(strategy_id=self.strategy_id, symbol=symbol, signal_type=SignalType.HOLD, strength=0.0, price=price, reason=reason)

    def generate_signal(self, symbol: str, bar: dict[str, Any]) -> Signal:
        c, h, l = bar.get("close"), bar.get("high"), bar.get("low")
        if c is None or h is None or l is None:
            return self._hold_signal(symbol, None, "incomplete")

        self._ensure(symbol)
        fc, fh, fl = float(c), float(h), float(l)
        self._close[symbol].append(fc)
        self._high[symbol].append(fh)
        self._low[symbol].append(fl)

        if len(self._close[symbol]) < int(self.get_param("min_bars")):
            return self._hold_signal(symbol, fc, "warming up")

        closes = list(self._close[symbol])

        vol_pct = self._vol_percentile(closes)
        bb_period, bb_std = self._adaptive_params(vol_pct)
        bb = self._bbands(closes, bb_period, bb_std)
        if bb is None:
            return self._hold_signal(symbol, fc, "bbands")
        lower, mid, upper = bb

        rsi_val = self._rsi(closes, int(self.get_param("rsi_period")))
        if rsi_val is None:
            return self._hold_signal(symbol, fc, "rsi")

        atr = self._atr(symbol)
        adx = self._adx(symbol)

        rsi_os = float(self.get_param("rsi_oversold"))
        rsi_ob = float(self.get_param("rsi_overbought"))
        adx_thresh = float(self.get_param("adx_trend_threshold"))
        max_hold = int(self.get_param("max_hold_bars"))
        stop_mult = float(self.get_param("atr_stop_mult"))

        pos = self.get_position(symbol)

        if pos is not None:
            self._hold_bars[symbol] = self._hold_bars.get(symbol, 0) + 1
            is_long = pos.side.value == "buy"

            if is_long and fc >= mid:
                self._hold_bars[symbol] = 0
                return self._exit_signal(symbol, fc, "LONG_EXIT", f"回归中轨 mid={mid:.2f}")

            if not is_long and fc <= mid:
                self._hold_bars[symbol] = 0
                return self._exit_signal(symbol, fc, "SHORT_EXIT", f"回归中轨 mid={mid:.2f}")

            if atr and self._entry_atr.get(symbol):
                ea = self._entry_atr[symbol]
                if is_long and fc < pos.avg_price - ea * stop_mult:
                    self._hold_bars[symbol] = 0
                    return self._exit_signal(symbol, fc, "LONG_EXIT", f"ATR止损 stop={pos.avg_price - ea * stop_mult:.2f}")
                if not is_long and fc > pos.avg_price + ea * stop_mult:
                    self._hold_bars[symbol] = 0
                    return self._exit_signal(symbol, fc, "SHORT_EXIT", f"ATR止损 stop={pos.avg_price + ea * stop_mult:.2f}")

            if self._hold_bars.get(symbol, 0) >= max_hold:
                self._hold_bars[symbol] = 0
                exit_type = "LONG_EXIT" if is_long else "SHORT_EXIT"
                return self._exit_signal(symbol, fc, exit_type, f"持仓超时 {max_hold}根")

            return self._hold_signal(symbol, fc, "holding")

        if adx is not None and adx > adx_thresh:
            return self._hold_signal(symbol, fc, f"ADX趋势过滤 adx={adx:.1f}>{adx_thresh}")

        touch_lower = fl <= lower or fc <= lower
        touch_upper = fh >= upper or fc >= upper

        if touch_lower and rsi_val < rsi_os:
            strength = min(0.5 + (rsi_os - rsi_val) / max(rsi_os, 1.0) * 0.5, 1.0)
            if atr:
                self._entry_atr[symbol] = atr
            sig = Signal(
                strategy_id=self.strategy_id, symbol=symbol,
                signal_type=SignalType.LONG_ENTRY, strength=round(strength, 4),
                price=fc,
                reason=f"自适应BB下轨(p={bb_period},k={bb_std:.2f})+RSI{rsi_val:.0f}<{rsi_os} vol={vol_pct:.0%}",
                metadata={"lower": lower, "mid": mid, "upper": upper, "rsi": rsi_val, "vol_pct": vol_pct, "adx": adx or 0},
            )
            self.record_signal(sig)
            return sig

        if touch_upper and rsi_val > rsi_ob:
            strength = min(0.5 + (rsi_val - rsi_ob) / max(100.0 - rsi_ob, 1.0) * 0.5, 1.0)
            if atr:
                self._entry_atr[symbol] = atr
            sig = Signal(
                strategy_id=self.strategy_id, symbol=symbol,
                signal_type=SignalType.SHORT_ENTRY, strength=round(strength, 4),
                price=fc,
                reason=f"自适应BB上轨(p={bb_period},k={bb_std:.2f})+RSI{rsi_val:.0f}>{rsi_ob} vol={vol_pct:.0%}",
                metadata={"lower": lower, "mid": mid, "upper": upper, "rsi": rsi_val, "vol_pct": vol_pct, "adx": adx or 0},
            )
            self.record_signal(sig)
            return sig

        return self._hold_signal(symbol, fc, "no signal")

    def _exit_signal(self, symbol: str, price: float, exit_type: str, reason: str) -> Signal:
        st = SignalType.LONG_EXIT if exit_type == "LONG_EXIT" else SignalType.SHORT_EXIT
        sig = Signal(strategy_id=self.strategy_id, symbol=symbol, signal_type=st, strength=0.8, price=price, reason=f"布林: {reason}")
        self.record_signal(sig)
        return sig

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        sig = self.generate_signal(symbol, bar)
        return [] if sig.signal_type == SignalType.HOLD else [sig]

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        for sym in self.config.symbols:
            b = market_data.get(sym)
            if b:
                out.extend(await self.on_bar(sym, b))
        return out
