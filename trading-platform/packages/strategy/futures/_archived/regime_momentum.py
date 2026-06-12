"""Regime-Adaptive EMA Momentum — HMM 市场状态识别 + EMA 交叉动量。

趋势市(TRENDING): 激进跟趋势，快慢EMA靠近，宽ATR止损
震荡市(CHOPPY):   不开新仓，加速平仓
常规市(NORMAL):   中性参数，标准动量

研究来源:
- RegimeForecast: "HMM for Market Regimes"
- Abdullah-BA: "Regime-Switching Momentum"
- PyQuantLab: "GMM Regime-Switching Momentum"
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..indicators import calc_atr
from ..registry import auto_register
from ..regime.hmm_detector import HMMRegimeDetector, MarketRegime

logger = logging.getLogger(__name__)

REGIME_PARAMS = {
    MarketRegime.TRENDING: {
        "fast_ema": 12,
        "slow_ema": 26,
        "signal_ema": 9,
        "atr_stop_mult": 3.0,
        "position_scale": 1.0,
        "max_hold_bars": 480,
        "profit_target_atr": 5.0,
    },
    MarketRegime.NORMAL: {
        "fast_ema": 20,
        "slow_ema": 50,
        "signal_ema": 14,
        "atr_stop_mult": 2.0,
        "position_scale": 0.6,
        "max_hold_bars": 240,
        "profit_target_atr": 3.5,
    },
    MarketRegime.CHOPPY: {
        "fast_ema": 20,
        "slow_ema": 50,
        "signal_ema": 14,
        "atr_stop_mult": 1.5,
        "position_scale": 0.0,
        "max_hold_bars": 60,
        "profit_target_atr": 2.0,
    },
    MarketRegime.UNKNOWN: {
        "fast_ema": 20,
        "slow_ema": 50,
        "signal_ema": 14,
        "atr_stop_mult": 2.0,
        "position_scale": 0.3,
        "max_hold_bars": 120,
        "profit_target_atr": 3.0,
    },
}

DEFAULT_PARAMS = {
    "atr_period": 48,
    "adx_period": 48,
    "adx_min_strength": 22.0,
    "rsi_period": 14,
    "rsi_overbought": 75.0,
    "rsi_oversold": 25.0,
    "hmm_lookback": 480,
    "hmm_retrain_interval": 120,
    "regime_confidence_threshold": 0.4,
    "min_bars": 60,
}


@auto_register("regime_momentum")
class RegimeMomentumStrategy(BaseStrategy):
    """EMA crossover momentum with HMM-driven parameter switching.

    MACD-like signal: fast_ema - slow_ema, smoothed by signal_ema.
    Entries: MACD histogram crosses zero (confirmed by ADX filter).
    Exits: ATR trailing stop, profit target, hold-time limit, regime shift to CHOPPY.
    """

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._hmm_lookback = int(merged["hmm_lookback"])
        self._hmm_retrain = int(merged["hmm_retrain_interval"])
        self._detectors: dict[str, HMMRegimeDetector] = {}
        self._closes: dict[str, deque[float]] = {}
        self._highs: dict[str, deque[float]] = {}
        self._lows: dict[str, deque[float]] = {}
        self._peak_price: dict[str, float] = {}
        self._trough_price: dict[str, float] = {}
        self._entry_atr: dict[str, float] = {}
        self._entry_price: dict[str, float] = {}
        self._bars_in_pos: dict[str, int] = {}
        self._bar_count: dict[str, int] = {}
        self._prev_hist: dict[str, float | None] = {}

    def _ensure_symbol(self, symbol: str) -> None:
        if symbol not in self._closes:
            self._closes[symbol] = deque(maxlen=600)
            self._highs[symbol] = deque(maxlen=600)
            self._lows[symbol] = deque(maxlen=600)
            self._detectors[symbol] = HMMRegimeDetector(
                lookback=self._hmm_lookback,
                retrain_interval=self._hmm_retrain,
            )
            self._bar_count[symbol] = 0
            self._prev_hist[symbol] = None

    def _active_params(self, symbol: str) -> dict[str, Any]:
        state = self._detectors[symbol].current_regime
        threshold = float(self.get_param("regime_confidence_threshold"))
        if state.confidence >= threshold:
            return REGIME_PARAMS[state.regime]
        return REGIME_PARAMS[MarketRegime.UNKNOWN]

    def _ema(self, data: list[float], period: int) -> list[float]:
        if len(data) < period:
            return []
        alpha = 2.0 / (period + 1)
        result = [sum(data[:period]) / period]
        for val in data[period:]:
            result.append(alpha * val + (1 - alpha) * result[-1])
        return result

    def _calc_macd(self, symbol: str, params: dict[str, Any]) -> tuple[float, float, float] | None:
        closes = list(self._closes[symbol])
        fast_p = params["fast_ema"]
        slow_p = params["slow_ema"]
        sig_p = params["signal_ema"]
        if len(closes) < slow_p + sig_p:
            return None

        fast = self._ema(closes, fast_p)
        slow = self._ema(closes, slow_p)
        offset = len(fast) - len(slow)
        macd_line = [f - s for f, s in zip(fast[offset:], slow)]
        if len(macd_line) < sig_p:
            return None

        signal_line = self._ema(macd_line, sig_p)
        len(macd_line) - len(signal_line)
        histogram = macd_line[-1] - signal_line[-1]
        return macd_line[-1], signal_line[-1], histogram

    def _calc_atr(self, symbol: str) -> float | None:
        return calc_atr(self._highs[symbol], self._lows[symbol], self._closes[symbol], int(self.get_param("atr_period")))

    def _calc_adx(self, symbol: str) -> float | None:
        period = int(self.get_param("adx_period"))
        n = period * 2 + 1
        if len(self._highs[symbol]) < n:
            return None
        highs = list(self._highs[symbol])
        lows = list(self._lows[symbol])
        closes = list(self._closes[symbol])

        plus_dms, minus_dms, trs = [], [], []
        for i in range(-period * 2, 0):
            up = highs[i] - highs[i - 1]
            down = lows[i - 1] - lows[i]
            plus_dms.append(up if up > down and up > 0 else 0)
            minus_dms.append(down if down > up and down > 0 else 0)
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
            trs.append(tr)

        sp = sum(plus_dms[:period])
        sm = sum(minus_dms[:period])
        st = sum(trs[:period])

        dxs = []
        for i in range(period, len(plus_dms)):
            sp = sp - sp / period + plus_dms[i]
            sm = sm - sm / period + minus_dms[i]
            st = st - st / period + trs[i]
            if st <= 0:
                continue
            pdi = 100 * sp / st
            mdi = 100 * sm / st
            di_sum = pdi + mdi
            dxs.append(100 * abs(pdi - mdi) / di_sum if di_sum > 0 else 0)

        if not dxs:
            return None
        return sum(dxs[-period:]) / min(len(dxs), period)

    def _calc_rsi(self, symbol: str) -> float | None:
        period = int(self.get_param("rsi_period"))
        if len(self._closes[symbol]) < period + 1:
            return None
        closes = list(self._closes[symbol])[-(period + 1):]
        gains, losses = 0.0, 0.0
        for i in range(1, len(closes)):
            d = closes[i] - closes[i - 1]
            if d > 0:
                gains += d
            else:
                losses -= d
        avg_gain = gains / period
        avg_loss = losses / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1 + rs)

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        close = bar.get("close")
        high = bar.get("high")
        low = bar.get("low")
        if close is None or high is None or low is None:
            return []

        self._ensure_symbol(symbol)
        fc, fh, fl = float(close), float(high), float(low)
        self._closes[symbol].append(fc)
        self._highs[symbol].append(fh)
        self._lows[symbol].append(fl)
        self._bar_count[symbol] += 1

        self._detectors[symbol].update(fc)

        if self._bar_count[symbol] < int(self.get_param("min_bars")):
            return []

        params = self._active_params(symbol)
        atr = self._calc_atr(symbol)
        macd_result = self._calc_macd(symbol, params)
        if atr is None or atr <= 0 or macd_result is None:
            return []

        _macd_val, _signal_val, histogram = macd_result

        pos = self.get_position(symbol)
        signals: list[Signal] = []

        if symbol in self._bars_in_pos:
            self._bars_in_pos[symbol] += 1

        if pos is not None:
            signals.extend(self._check_exits(symbol, fc, atr, pos, params))
        elif params["position_scale"] > 0:
            signals.extend(self._check_entries(symbol, fc, atr, histogram, params))

        self._prev_hist[symbol] = histogram
        return signals

    def _check_entries(
        self, symbol: str, close: float, atr: float,
        histogram: float, params: dict[str, Any],
    ) -> list[Signal]:
        if self._prev_hist.get(symbol) is None:
            return []

        adx = self._calc_adx(symbol)
        min_adx = float(self.get_param("adx_min_strength"))
        if adx is not None and adx < min_adx:
            return []

        rsi = self._calc_rsi(symbol)
        ob = float(self.get_param("rsi_overbought"))
        os_ = float(self.get_param("rsi_oversold"))

        regime = self._detectors[symbol].current_regime.regime.value
        signals = []

        prev_h = self._prev_hist[symbol]
        if prev_h is not None and prev_h <= 0 < histogram:
            if rsi is not None and rsi > ob:
                return []
            strength = min(abs(histogram) / atr * 5 + 0.4, 1.0)
            sig = Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_type=SignalType.LONG_ENTRY,
                strength=round(strength * params["position_scale"], 4),
                price=close,
                reason=f"MACD多头交叉(regime={regime}) hist={histogram:.4f}",
                metadata={"regime": regime, "adx": adx, "rsi": rsi, "atr": atr},
            )
            signals.append(sig)
            self.record_signal(sig)
            self._peak_price[symbol] = close
            self._entry_atr[symbol] = atr
            self._entry_price[symbol] = close
            self._bars_in_pos[symbol] = 0

        elif prev_h is not None and prev_h >= 0 > histogram:
            if rsi is not None and rsi < os_:
                return []
            strength = min(abs(histogram) / atr * 5 + 0.4, 1.0)
            sig = Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_type=SignalType.SHORT_ENTRY,
                strength=round(strength * params["position_scale"], 4),
                price=close,
                reason=f"MACD空头交叉(regime={regime}) hist={histogram:.4f}",
                metadata={"regime": regime, "adx": adx, "rsi": rsi, "atr": atr},
            )
            signals.append(sig)
            self.record_signal(sig)
            self._trough_price[symbol] = close
            self._entry_atr[symbol] = atr
            self._entry_price[symbol] = close
            self._bars_in_pos[symbol] = 0

        return signals

    def _check_exits(
        self, symbol: str, close: float, atr: float, pos: Any, params: dict[str, Any],
    ) -> list[Signal]:
        stop_mult = params["atr_stop_mult"]
        max_hold = params["max_hold_bars"]
        profit_target = params["profit_target_atr"]
        bars_held = self._bars_in_pos.get(symbol, 0)
        entry_atr = self._entry_atr.get(symbol, atr)
        entry_px = self._entry_price.get(symbol, close)
        is_long = pos.side.value == "buy"

        signals = []

        if is_long:
            self._peak_price[symbol] = max(self._peak_price.get(symbol, close), close)
            trail_stop = self._peak_price[symbol] - atr * stop_mult
            profit_hit = (close - entry_px) >= entry_atr * profit_target

            should_exit = False
            reason = ""
            if bars_held >= max_hold:
                should_exit, reason = True, f"持仓超限 {bars_held}>={max_hold}"
            elif profit_hit:
                should_exit, reason = True, f"止盈 pnl={close - entry_px:.2f} >= {entry_atr * profit_target:.2f}"
            elif close < trail_stop:
                should_exit, reason = True, f"ATR止损 trail={trail_stop:.2f}"

            if should_exit:
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_EXIT, strength=0.85, price=close,
                    reason=reason, metadata={"bars_held": bars_held},
                )
                signals.append(sig)
                self.record_signal(sig)
                self._cleanup_pos(symbol)
        else:
            self._trough_price[symbol] = min(self._trough_price.get(symbol, close), close)
            trail_stop = self._trough_price[symbol] + atr * stop_mult
            profit_hit = (entry_px - close) >= entry_atr * profit_target

            should_exit = False
            reason = ""
            if bars_held >= max_hold:
                should_exit, reason = True, f"持仓超限 {bars_held}>={max_hold}"
            elif profit_hit:
                should_exit, reason = True, f"止盈 pnl={entry_px - close:.2f} >= {entry_atr * profit_target:.2f}"
            elif close > trail_stop:
                should_exit, reason = True, f"ATR止损 trail={trail_stop:.2f}"

            if should_exit:
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_EXIT, strength=0.85, price=close,
                    reason=reason, metadata={"bars_held": bars_held},
                )
                signals.append(sig)
                self.record_signal(sig)
                self._cleanup_pos(symbol)

        return signals

    def _cleanup_pos(self, symbol: str) -> None:
        self._peak_price.pop(symbol, None)
        self._trough_price.pop(symbol, None)
        self._entry_atr.pop(symbol, None)
        self._entry_price.pop(symbol, None)
        self._bars_in_pos.pop(symbol, None)

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        all_signals: list[Signal] = []
        for symbol in self.config.symbols:
            bar = market_data.get(symbol)
            if bar:
                sigs = await self.on_bar(symbol, bar)
                all_signals.extend(sigs)
        return all_signals
