"""Trend Following v2 — regime-adaptive upgrade.

Improvements over the archived trend_following:
1. Regime-adaptive ATR multipliers (tight in ranging, wide in trending)
2. Multi-timeframe trend alignment (uses slow EMA as proxy for higher TF)
3. Chandelier Exit instead of simple trailing stop
4. Volume surge confirmation (>1.5x 20-bar average)
5. Cooldown mechanism to prevent overtrading

Based on C2 improvement plan. Original trend_following had:
  Return +25.0%, Sharpe 0.578, WinRate 47.1%, PF 1.576 (BTC 4h)
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any

from strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig
from strategy.indicators import calc_atr, ema_update
from strategy.registry import auto_register
from .regime_detector import MarketRegimeDetector, MarketRegime

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "ema_fast": 8,
    "ema_slow": 29,
    "ema_trend": 52,
    "adx_period": 15,
    "adx_threshold": 30.0,
    "atr_period": 14,
    "volume_ma_period": 20,
    "volume_surge_ratio": 1.8,
    "cooldown_bars": 8,
    "max_hold_bars": 96,
    "regime_atr_sl": {
        "strong_trend": 2.5,
        "weak_trend": 2.0,
        "ranging": 1.5,
        "high_volatility": 3.0,
        "breakout": 2.0,
    },
    "regime_atr_tp": {
        "strong_trend": 5.0,
        "weak_trend": 3.5,
        "ranging": 2.5,
        "high_volatility": 4.0,
        "breakout": 4.5,
    },
    "chandelier_period": 22,
    "use_regime_filter": True,
    "blocked_regimes": ["ranging"],
}


@auto_register("trend_following_v2")
class TrendFollowingV2Strategy(BaseStrategy):
    """Regime-adaptive trend following with Chandelier Exit."""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._closes: dict[str, deque[float]] = {}
        self._highs: dict[str, deque[float]] = {}
        self._lows: dict[str, deque[float]] = {}
        self._volumes: dict[str, deque[float]] = {}
        self._ema_f: dict[str, float | None] = {}
        self._ema_s: dict[str, float | None] = {}
        self._ema_t: dict[str, float | None] = {}
        self._regime: dict[str, MarketRegimeDetector] = {}
        self._hold_bars: dict[str, int] = {}
        self._cooldown: dict[str, int] = {}
        self._peak: dict[str, float] = {}
        self._trough: dict[str, float] = {}
        self._entry_price: dict[str, float] = {}
        self._buf = 200

    def _init(self, s: str) -> None:
        if s not in self._closes:
            self._closes[s] = deque(maxlen=self._buf)
            self._highs[s] = deque(maxlen=self._buf)
            self._lows[s] = deque(maxlen=self._buf)
            self._volumes[s] = deque(maxlen=self._buf)
            self._regime[s] = MarketRegimeDetector()

    def _volume_surge(self, s: str) -> bool:
        vols = list(self._volumes[s])
        p = self.get_param("volume_ma_period")
        if len(vols) < p:
            return True
        vol_ma = sum(vols[-p:]) / p
        return vols[-1] >= vol_ma * self.get_param("volume_surge_ratio") if vol_ma > 0 else True

    def _chandelier_exit(self, s: str, side: str) -> float | None:
        """Chandelier Exit: highest high/lowest low minus ATR multiplier."""
        period = self.get_param("chandelier_period")
        regime = self._regime[s].current_regime
        atr_map = self.get_param("regime_atr_sl")
        mult = atr_map.get(regime.value, 2.0)

        atr = calc_atr(self._highs[s], self._lows[s], self._closes[s], self.get_param("atr_period"))
        if atr is None or atr <= 0:
            return None

        highs = list(self._highs[s])
        lows = list(self._lows[s])
        if len(highs) < period:
            return None

        if side == "buy":
            highest = max(highs[-period:])
            return highest - atr * mult
        else:
            lowest = min(lows[-period:])
            return lowest + atr * mult

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._init(symbol)
        c, h, l = bar["close"], bar["high"], bar["low"]
        vol = bar.get("volume", 0.0)

        self._closes[symbol].append(c)
        self._highs[symbol].append(h)
        self._lows[symbol].append(l)
        self._volumes[symbol].append(vol)

        self._ema_f[symbol] = ema_update(self._ema_f.get(symbol), c, self.get_param("ema_fast"))
        self._ema_s[symbol] = ema_update(self._ema_s.get(symbol), c, self.get_param("ema_slow"))
        self._ema_t[symbol] = ema_update(self._ema_t.get(symbol), c, self.get_param("ema_trend"))
        self._regime[symbol].update(h, l, c)

        ef = self._ema_f[symbol]
        es = self._ema_s[symbol]
        et = self._ema_t[symbol]
        if ef is None or es is None or et is None:
            return []

        regime = self._regime[symbol].current_regime
        atr = calc_atr(self._highs[symbol], self._lows[symbol], self._closes[symbol], self.get_param("atr_period"))
        if atr is None or atr <= 0:
            return []

        signals: list[Signal] = []
        pos = self.get_position(symbol)
        self._cooldown[symbol] = max(self._cooldown.get(symbol, 0) - 1, 0)

        if pos is None:
            if self._cooldown.get(symbol, 0) > 0:
                return signals

            if self.get_param("use_regime_filter"):
                blocked = self.get_param("blocked_regimes")
                if regime.value in blocked:
                    return signals

            bullish = ef > es and c > et
            bearish = ef < es and c < et
            vol_ok = self._volume_surge(symbol)

            if bullish and vol_ok:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY, strength=0.85, price=c,
                    reason=f"TFv2 LONG [{regime.value}] vol_surge",
                    metadata={"regime": regime.value, "atr": atr},
                ))
                self._hold_bars[symbol] = 0
                self._peak[symbol] = h
                self._entry_price[symbol] = c

            elif bearish and vol_ok:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY, strength=0.85, price=c,
                    reason=f"TFv2 SHORT [{regime.value}] vol_surge",
                    metadata={"regime": regime.value, "atr": atr},
                ))
                self._hold_bars[symbol] = 0
                self._trough[symbol] = l
                self._entry_price[symbol] = c

        else:
            self._hold_bars[symbol] = self._hold_bars.get(symbol, 0) + 1
            should_exit = False
            reason = ""

            if self._hold_bars[symbol] >= self.get_param("max_hold_bars"):
                should_exit, reason = True, "timeout"
            else:
                chandelier = self._chandelier_exit(symbol, pos.side.value)
                if chandelier is not None:
                    if pos.side.value == "buy" and l <= chandelier:
                        should_exit, reason = True, f"Chandelier SL @{chandelier:.1f}"
                    elif pos.side.value == "sell" and h >= chandelier:
                        should_exit, reason = True, f"Chandelier SL @{chandelier:.1f}"

                if not should_exit:
                    entry = self._entry_price.get(symbol, c)
                    tp_map = self.get_param("regime_atr_tp")
                    tp_mult = tp_map.get(regime.value, 4.0)
                    if pos.side.value == "buy" and c >= entry + atr * tp_mult:
                        should_exit, reason = True, f"TP [{regime.value}] {tp_mult}x ATR"
                    elif pos.side.value == "sell" and c <= entry - atr * tp_mult:
                        should_exit, reason = True, f"TP [{regime.value}] {tp_mult}x ATR"

            if should_exit:
                exit_type = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_type, strength=0.9, price=c,
                    reason=f"TFv2: {reason}",
                ))
                self._cooldown[symbol] = self.get_param("cooldown_bars")

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        for s in self.config.symbols:
            if s in market_data:
                out.extend(await self.on_bar(s, market_data[s]))
        return out
