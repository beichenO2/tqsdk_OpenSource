"""Multi-factor BTC strategy — combines VWAP, OBV, fund flow, volatility regime.

Each factor produces a score in [-1, 1]. The composite score determines
signal direction and strength. Factor weights are configurable and can be
dynamically adjusted by the regime detector.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..indicators import calc_atr, ema_update
from ..registry import auto_register
from .regime_detector import MarketRegime, MarketRegimeDetector

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "vwap_period": 19,
    "obv_ma_period": 14,
    "fund_flow_period": 8,
    "rsi_period": 19,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "vol_period": 20,
    "composite_entry_threshold": 0.305,
    "composite_exit_threshold": -0.288,
    "atr_period": 15,
    "stop_loss_atr_mult": 3.42,
    "take_profit_atr_mult": 6.89,
    "weight_vwap": 0.118,
    "weight_obv": 0.178,
    "weight_fund_flow": 0.249,
    "weight_rsi": 0.326,
    "weight_macd": 0.056,
    "weight_vol_regime": 0.087,
    "enable_regime_adaptation": True,
}


_ema = ema_update


@auto_register("btc_multifactor")
class BTCMultiFactorStrategy(BaseStrategy):
    """Multi-factor composite strategy with regime-aware weighting."""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        buf = 100
        self._close: dict[str, deque[float]] = {}
        self._high: dict[str, deque[float]] = {}
        self._low: dict[str, deque[float]] = {}
        self._volume: dict[str, deque[float]] = {}
        self._taker_buy_vol: dict[str, deque[float]] = {}
        self._obv: dict[str, float] = {}
        self._obv_history: dict[str, deque[float]] = {}
        self._macd_fast_ema: dict[str, float | None] = {}
        self._macd_slow_ema: dict[str, float | None] = {}
        self._macd_signal_ema: dict[str, float | None] = {}
        self._regime_detector: dict[str, MarketRegimeDetector] = {}
        self._buf = buf

    def _ensure_buffers(self, symbol: str) -> None:
        if symbol not in self._close:
            self._close[symbol] = deque(maxlen=self._buf)
            self._high[symbol] = deque(maxlen=self._buf)
            self._low[symbol] = deque(maxlen=self._buf)
            self._volume[symbol] = deque(maxlen=self._buf)
            self._taker_buy_vol[symbol] = deque(maxlen=self._buf)
            self._obv[symbol] = 0.0
            self._obv_history[symbol] = deque(maxlen=self._buf)
            self._macd_fast_ema[symbol] = None
            self._macd_slow_ema[symbol] = None
            self._macd_signal_ema[symbol] = None
            self._regime_detector[symbol] = MarketRegimeDetector()

    def _calc_vwap_score(self, symbol: str, price: float) -> float:
        """VWAP deviation: positive when price > VWAP (bullish), negative below."""
        period = self.get_param("vwap_period")
        closes = list(self._close[symbol])
        volumes = list(self._volume[symbol])
        if len(closes) < period:
            return 0.0
        cum_pv = sum(c * v for c, v in zip(closes[-period:], volumes[-period:]))
        cum_v = sum(volumes[-period:])
        if cum_v == 0:
            return 0.0
        vwap = cum_pv / cum_v
        deviation = (price - vwap) / vwap
        return max(min(deviation * 10, 1.0), -1.0)

    def _calc_obv_score(self, symbol: str) -> float:
        """OBV trend: positive when OBV above its MA."""
        period = self.get_param("obv_ma_period")
        obv_hist = list(self._obv_history[symbol])
        if len(obv_hist) < period:
            return 0.0
        obv_ma = sum(obv_hist[-period:]) / period
        current = obv_hist[-1]
        if obv_ma == 0:
            return 0.0
        deviation = (current - obv_ma) / abs(obv_ma)
        return max(min(deviation * 5, 1.0), -1.0)

    def _calc_fund_flow_score(self, symbol: str) -> float:
        """Taker buy ratio: > 0.5 means buyers dominate (bullish)."""
        period = self.get_param("fund_flow_period")
        vols = list(self._volume[symbol])
        taker_vols = list(self._taker_buy_vol[symbol])
        if len(vols) < period:
            return 0.0
        total_vol = sum(vols[-period:])
        total_taker = sum(taker_vols[-period:])
        if total_vol == 0:
            return 0.0
        ratio = total_taker / total_vol
        return max(min((ratio - 0.5) * 4, 1.0), -1.0)

    def _calc_rsi_score(self, symbol: str) -> float:
        """RSI momentum score: positive when RSI > 50 (bullish momentum)."""
        period = self.get_param("rsi_period")
        closes = list(self._close[symbol])
        if len(closes) < period + 1:
            return 0.0
        gains, losses = [], []
        for i in range(-period, 0):
            d = closes[i] - closes[i - 1]
            gains.append(max(d, 0))
            losses.append(max(-d, 0))
        avg_g = sum(gains) / period
        avg_l = sum(losses) / period
        if avg_l == 0:
            rsi = 100.0
        else:
            rsi = 100 - 100 / (1 + avg_g / avg_l)
        return max(min((rsi - 50) / 50, 1.0), -1.0)

    def _calc_macd_score(self, symbol: str, close: float) -> float:
        fast_p = self.get_param("macd_fast")
        slow_p = self.get_param("macd_slow")
        sig_p = self.get_param("macd_signal")

        self._macd_fast_ema[symbol] = _ema(self._macd_fast_ema[symbol], close, fast_p)
        self._macd_slow_ema[symbol] = _ema(self._macd_slow_ema[symbol], close, slow_p)

        fe = self._macd_fast_ema[symbol]
        se = self._macd_slow_ema[symbol]
        if fe is None or se is None:
            return 0.0

        macd_line = fe - se
        self._macd_signal_ema[symbol] = _ema(self._macd_signal_ema[symbol], macd_line, sig_p)
        sig_ema = self._macd_signal_ema[symbol]
        if sig_ema is None:
            return 0.0

        histogram = macd_line - sig_ema
        if close == 0:
            return 0.0
        normalized = histogram / close * 100
        return max(min(normalized * 10, 1.0), -1.0)

    def _calc_vol_regime_score(self, symbol: str) -> float:
        """Returns positive for low vol (favorable for entry), negative for high vol."""
        regime = self._regime_detector[symbol].current_regime
        score_map = {
            MarketRegime.STRONG_TREND: 0.6,
            MarketRegime.WEAK_TREND: 0.3,
            MarketRegime.RANGING: -0.2,
            MarketRegime.HIGH_VOLATILITY: -0.6,
            MarketRegime.BREAKOUT: 0.4,
            MarketRegime.UNKNOWN: 0.0,
        }
        return score_map.get(regime, 0.0)

    def _get_weights(self, symbol: str) -> dict[str, float]:
        """Return factor weights, optionally adjusted by market regime."""
        base = {
            "vwap": self.get_param("weight_vwap"),
            "obv": self.get_param("weight_obv"),
            "fund_flow": self.get_param("weight_fund_flow"),
            "rsi": self.get_param("weight_rsi"),
            "macd": self.get_param("weight_macd"),
            "vol_regime": self.get_param("weight_vol_regime"),
        }

        if not getattr(self, "_has_taker_data", True):
            base["fund_flow"] = 0.0

        total = sum(base.values())
        if total > 0:
            base = {k: v / total for k, v in base.items()}

        if not self.get_param("enable_regime_adaptation"):
            return base

        regime = self._regime_detector[symbol].current_regime
        if regime == MarketRegime.STRONG_TREND:
            base["macd"] *= 1.3
            base["vwap"] *= 1.2
            base["rsi"] *= 0.7
        elif regime == MarketRegime.RANGING:
            base["rsi"] *= 1.4
            base["macd"] *= 0.6
            base["fund_flow"] *= 1.2
        elif regime == MarketRegime.HIGH_VOLATILITY:
            base["vol_regime"] *= 1.5
            base["macd"] *= 0.5
            base["fund_flow"] *= 0.8

        total = sum(base.values())
        if total > 0:
            base = {k: v / total for k, v in base.items()}
        return base

    def _calc_atr(self, symbol: str) -> float | None:
        return calc_atr(
            self._high[symbol],
            self._low[symbol],
            self._close[symbol],
            self.get_param("atr_period"),
        )

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._ensure_buffers(symbol)

        close = bar["close"]
        high = bar["high"]
        low = bar["low"]
        vol = bar.get("volume", 0.0)
        taker = bar.get("taker_buy_volume")
        self._has_taker_data = taker is not None
        if taker is None:
            taker = vol * 0.5

        self._close[symbol].append(close)
        self._high[symbol].append(high)
        self._low[symbol].append(low)
        self._volume[symbol].append(vol)
        self._taker_buy_vol[symbol].append(taker)

        prev_close = list(self._close[symbol])[-2] if len(self._close[symbol]) > 1 else close
        if close >= prev_close:
            self._obv[symbol] += vol
        else:
            self._obv[symbol] -= vol
        self._obv_history[symbol].append(self._obv[symbol])

        self._regime_detector[symbol].update(high, low, close)

        scores = {
            "vwap": self._calc_vwap_score(symbol, close),
            "obv": self._calc_obv_score(symbol),
            "fund_flow": self._calc_fund_flow_score(symbol),
            "rsi": self._calc_rsi_score(symbol),
            "macd": self._calc_macd_score(symbol, close),
            "vol_regime": self._calc_vol_regime_score(symbol),
        }

        weights = self._get_weights(symbol)
        composite = sum(scores[k] * weights.get(k, 0) for k in scores)

        signals: list[Signal] = []
        entry_threshold = self.get_param("composite_entry_threshold")
        exit_threshold = self.get_param("composite_exit_threshold")
        # Long exit when composite drops to exit_threshold (negative);
        # short exit when composite rises to the symmetric opposite.
        long_exit_level = exit_threshold          # e.g. -0.1
        short_exit_level = -exit_threshold        # e.g.  0.1
        pos = self.get_position(symbol)

        if pos is None:
            if composite >= entry_threshold:
                strength = min(composite, 1.0)
                sig = Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=round(strength, 4),
                    price=close,
                    reason=f"多因子做多(composite={composite:.3f},regime={self._regime_detector[symbol].current_regime.value})",
                    metadata={
                        "scores": {k: round(v, 4) for k, v in scores.items()},
                        "weights": {k: round(v, 4) for k, v in weights.items()},
                        "composite": round(composite, 4),
                        "regime": self._regime_detector[symbol].current_regime.value,
                    },
                )
                signals.append(sig)
                self.record_signal(sig)

            elif composite <= -entry_threshold:
                strength = min(abs(composite), 1.0)
                sig = Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=round(strength, 4),
                    price=close,
                    reason=f"多因子做空(composite={composite:.3f},regime={self._regime_detector[symbol].current_regime.value})",
                    metadata={
                        "scores": {k: round(v, 4) for k, v in scores.items()},
                        "weights": {k: round(v, 4) for k, v in weights.items()},
                        "composite": round(composite, 4),
                        "regime": self._regime_detector[symbol].current_regime.value,
                    },
                )
                signals.append(sig)
                self.record_signal(sig)

        else:
            atr = self._calc_atr(symbol)
            if atr and atr > 0:
                regime_params = self._regime_detector[symbol].get_params()
                sl_mult = regime_params.get("stop_loss_mult", self.get_param("stop_loss_atr_mult"))
                tp_mult = regime_params.get("take_profit_mult", self.get_param("take_profit_atr_mult"))

                if pos.side.value == "buy":
                    if close < pos.avg_price - atr * sl_mult:
                        sig = Signal(
                            strategy_id=self.strategy_id, symbol=symbol,
                            signal_type=SignalType.LONG_EXIT, strength=0.9, price=close,
                            reason=f"多因子止损(loss={atr*sl_mult:.2f})",
                        )
                        signals.append(sig)
                        self.record_signal(sig)
                    elif close > pos.avg_price + atr * tp_mult:
                        sig = Signal(
                            strategy_id=self.strategy_id, symbol=symbol,
                            signal_type=SignalType.LONG_EXIT, strength=0.7, price=close,
                            reason=f"多因子止盈(profit={atr*tp_mult:.2f})",
                        )
                        signals.append(sig)
                        self.record_signal(sig)
                    elif composite <= long_exit_level:
                        sig = Signal(
                            strategy_id=self.strategy_id, symbol=symbol,
                            signal_type=SignalType.LONG_EXIT, strength=0.5, price=close,
                            reason=f"多因子反转平仓(composite={composite:.3f})",
                        )
                        signals.append(sig)
                        self.record_signal(sig)

                elif pos.side.value == "sell":
                    if close > pos.avg_price + atr * sl_mult:
                        sig = Signal(
                            strategy_id=self.strategy_id, symbol=symbol,
                            signal_type=SignalType.SHORT_EXIT, strength=0.9, price=close,
                            reason=f"多因子止损(loss={atr*sl_mult:.2f})",
                        )
                        signals.append(sig)
                        self.record_signal(sig)
                    elif close < pos.avg_price - atr * tp_mult:
                        sig = Signal(
                            strategy_id=self.strategy_id, symbol=symbol,
                            signal_type=SignalType.SHORT_EXIT, strength=0.7, price=close,
                            reason=f"多因子止盈(profit={atr*tp_mult:.2f})",
                        )
                        signals.append(sig)
                        self.record_signal(sig)
                    elif composite >= short_exit_level:
                        sig = Signal(
                            strategy_id=self.strategy_id, symbol=symbol,
                            signal_type=SignalType.SHORT_EXIT, strength=0.5, price=close,
                            reason=f"多因子反转平仓(composite={composite:.3f})",
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
