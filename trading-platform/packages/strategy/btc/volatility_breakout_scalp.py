"""1m Counter-Trend Momentum Reversion.

Entry: when 1H move exceeds threshold AGAINST the 4H trend,
bet on snapback (mean reversion to 4H trend).

Complement to scalp_momentum: enters when price surges against
the major trend, expecting a reversal back.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..indicators import ema_update
from ..registry import auto_register
from .regime_detector import MarketRegimeDetector

DEFAULT_PARAMS = {
    "ema_4h_fast": 480,
    "ema_4h_slow": 1260,

    "surge_window": 60,
    "surge_min_pct": 1.0,
    "min_close_ratio": 0.7,

    "stop_atr_mult": 1.5,
    "trail_activate_atr_mult": 3.0,
    "trail_atr_mult": 1.5,

    "max_hold_bars": 720,
    "cooldown_bars": 240,
    "max_trades_per_4h": 1,

    "use_regime_filter": True,
    "allowed_regimes": ["ranging", "high_volatility", "weak_trend"],
}


@auto_register("vol_breakout_scalp")
class VolatilityBreakoutScalpStrategy(BaseStrategy):

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._c: dict[str, deque[float]] = {}
        self._h: dict[str, deque[float]] = {}
        self._l: dict[str, deque[float]] = {}
        self._o: dict[str, deque[float]] = {}
        self._tr: dict[str, deque[float]] = {}
        self._ema: dict[str, dict[str, float | None]] = {}

        self._hb: dict[str, int] = {}
        self._peak: dict[str, float] = {}
        self._cd: dict[str, int] = {}
        self._trail_active: dict[str, bool] = {}
        self._4h_trades: dict[str, int] = {}
        self._4h_cnt: dict[str, int] = {}
        self._entry_atr: dict[str, float] = {}
        self._regime: dict[str, MarketRegimeDetector] = {}

    def _init(self, s: str) -> None:
        if s not in self._c:
            self._c[s] = deque(maxlen=1500)
            self._h[s] = deque(maxlen=1500)
            self._l[s] = deque(maxlen=1500)
            self._o[s] = deque(maxlen=1500)
            self._tr[s] = deque(maxlen=300)
            self._ema[s] = {}
            self._regime[s] = MarketRegimeDetector()

    def _calc_atr(self, s: str) -> float:
        trs = list(self._tr[s])
        p = min(120, len(trs))
        return sum(trs[-p:]) / p if p >= 10 else 0.0

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._init(symbol)
        c, h, l, o = bar["close"], bar["high"], bar["low"], bar["open"]

        self._c[symbol].append(c)
        self._h[symbol].append(h)
        self._l[symbol].append(l)
        self._o[symbol].append(o)

        prev_c = list(self._c[symbol])[-2] if len(self._c[symbol]) >= 2 else c
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        self._tr[symbol].append(tr)

        self._regime[symbol].update(h, l, c)
        atr = self._calc_atr(symbol)

        emas = self._ema[symbol]
        for key, period in [
            ("4hf", self.get_param("ema_4h_fast")),
            ("4hs", self.get_param("ema_4h_slow")),
        ]:
            emas[key] = ema_update(emas.get(key), c, period)

        cnt = self._4h_cnt.get(symbol, 0) + 1
        self._4h_cnt[symbol] = cnt
        if cnt >= 240:
            self._4h_cnt[symbol] = 0
            self._4h_trades[symbol] = 0

        signals: list[Signal] = []
        pos = self.get_position(symbol)
        self._cd[symbol] = max(self._cd.get(symbol, 0) - 1, 0)

        if pos is None:
            if self._cd.get(symbol, 0) > 0:
                return signals
            if self._4h_trades.get(symbol, 0) >= self.get_param("max_trades_per_4h"):
                return signals

            if self.get_param("use_regime_filter"):
                current_regime = self._regime[symbol].current_regime.value
                allowed = self.get_param("allowed_regimes")
                if current_regime not in allowed:
                    return signals

            e4hf = emas.get("4hf", 0)
            e4hs = emas.get("4hs", 0)
            if not all([e4hf, e4hs]):
                return signals

            closes = list(self._c[symbol])
            sw = self.get_param("surge_window")
            if len(closes) < sw + 1:
                return signals

            surge_start = closes[-(sw + 1)]
            surge_pct = (c - surge_start) / surge_start * 100

            surge_threshold = self.get_param("surge_min_pct")
            highs = list(self._h[symbol])
            lows = list(self._l[symbol])
            period_high = max(highs[-sw:])
            period_low = min(lows[-sw:])
            rng = period_high - period_low
            if rng < 1e-10:
                return signals

            regime_info = self._regime[symbol].current_regime.value

            if surge_pct < -surge_threshold and e4hf > e4hs:
                close_ratio = (c - period_low) / rng
                if close_ratio <= (1 - self.get_param("min_close_ratio")) and c > o:
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.LONG_ENTRY, strength=0.85, price=c,
                        reason=f"CT BUY dip={surge_pct:.2f}% [{regime_info}]",
                        metadata={"regime": regime_info, "atr": atr},
                    ))
                    self._hb[symbol] = 0
                    self._peak[symbol] = h
                    self._trail_active[symbol] = False
                    self._entry_atr[symbol] = atr
                    self._4h_trades[symbol] = self._4h_trades.get(symbol, 0) + 1

            elif surge_pct > surge_threshold and e4hf < e4hs:
                close_ratio = (period_high - c) / rng
                if close_ratio <= (1 - self.get_param("min_close_ratio")) and c < o:
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.SHORT_ENTRY, strength=0.85, price=c,
                        reason=f"CT SELL rip={surge_pct:.2f}% [{regime_info}]",
                        metadata={"regime": regime_info, "atr": atr},
                    ))
                    self._hb[symbol] = 0
                    self._peak[symbol] = l
                    self._trail_active[symbol] = False
                    self._entry_atr[symbol] = atr
                    self._4h_trades[symbol] = self._4h_trades.get(symbol, 0) + 1

        else:
            self._hb[symbol] = self._hb.get(symbol, 0) + 1

            ea = self._entry_atr.get(symbol, atr) or atr
            stop_d = ea * self.get_param("stop_atr_mult") if ea > 0 else c * 0.003
            trail_act = ea * self.get_param("trail_activate_atr_mult") if ea > 0 else c * 0.05
            trail_d = ea * self.get_param("trail_atr_mult") if ea > 0 else c * 0.025

            ex, reason = False, ""

            if self._hb[symbol] >= self.get_param("max_hold_bars"):
                ex, reason = True, "timeout"
            elif pos.side.value == "buy":
                self._peak[symbol] = max(self._peak.get(symbol, c), h)
                pdist = self._peak[symbol] - pos.avg_price
                if pdist >= trail_act:
                    self._trail_active[symbol] = True
                if self._trail_active.get(symbol, False):
                    ts = self._peak[symbol] - trail_d
                    if l <= ts:
                        pfit = (c - pos.avg_price) / pos.avg_price * 100
                        ex, reason = True, f"trail({pfit:.1f}%)"
                if c < pos.avg_price - stop_d:
                    ex, reason = True, "SL"
            elif pos.side.value == "sell":
                self._peak[symbol] = min(self._peak.get(symbol, c), l)
                pdist = pos.avg_price - self._peak[symbol]
                if pdist >= trail_act:
                    self._trail_active[symbol] = True
                if self._trail_active.get(symbol, False):
                    ts = self._peak[symbol] + trail_d
                    if h >= ts:
                        pfit = (pos.avg_price - c) / pos.avg_price * 100
                        ex, reason = True, f"trail({pfit:.1f}%)"
                if c > pos.avg_price + stop_d:
                    ex, reason = True, "SL"

            if ex:
                et = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=et, strength=0.9, price=c,
                    reason=f"CT: {reason}",
                ))
                self._cd[symbol] = self.get_param("cooldown_bars")

        return signals

    async def generate_signals(self, md: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        for s in self.config.symbols:
            if s in md:
                out.extend(await self.on_bar(s, md[s]))
        return out
