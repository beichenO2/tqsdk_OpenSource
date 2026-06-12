"""量价策略 — 滚动 VWAP、OBV 与量价动量确认。"""

from __future__ import annotations

from collections import deque
from typing import Any

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..registry import auto_register

DEFAULT_PARAMS = {
    "vwap_window": 30,
    "obv_ma_period": 12,
    "volume_ma_period": 20,
    "min_bars": 35,
    "divergence_bars": 5,
    "momentum_eps": 1e-9,
}


@auto_register("volume_price")
class VolumePriceStrategy(BaseStrategy):
    """OBV 与滚动 VWAP 结合：VWAP 上穿/下穿配合 OBV 相对均线与短期量价动量。"""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._close: dict[str, deque[float]] = {}
        self._high: dict[str, deque[float]] = {}
        self._low: dict[str, deque[float]] = {}
        self._volume: dict[str, deque[float]] = {}
        self._obv: dict[str, float] = {}
        self._obv_hist: dict[str, deque[float]] = {}
        self._vwap_hist: dict[str, deque[float]] = {}
        self._prev_close: dict[str, float | None] = {}

    def _ensure(self, symbol: str) -> None:
        mx = int(self.get_param("min_bars")) + 25
        if symbol not in self._close:
            self._close[symbol] = deque(maxlen=mx)
            self._high[symbol] = deque(maxlen=mx)
            self._low[symbol] = deque(maxlen=mx)
            self._volume[symbol] = deque(maxlen=mx)
            self._obv_hist[symbol] = deque(maxlen=mx)
            self._vwap_hist[symbol] = deque(maxlen=mx)
            self._obv[symbol] = 0.0
            self._prev_close[symbol] = None

    @staticmethod
    def _sma(vals: list[float], n: int) -> float | None:
        if n <= 0 or len(vals) < n:
            return None
        return sum(vals[-n:]) / n

    def _rolling_vwap(self, symbol: str) -> float | None:
        w = int(self.get_param("vwap_window"))
        highs = list(self._high[symbol])
        lows = list(self._low[symbol])
        closes = list(self._close[symbol])
        vols = list(self._volume[symbol])
        if len(closes) < w:
            return None
        tpv = 0.0
        vv = 0.0
        for i in range(-w, 0):
            tp = (highs[i] + lows[i] + closes[i]) / 3.0
            v = max(vols[i], 0.0)
            tpv += tp * v
            vv += v
        if vv <= 0:
            return None
        return tpv / vv

    def _hold(self, symbol: str, price: float | None, reason: str) -> Signal:
        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            signal_type=SignalType.HOLD,
            strength=0.0,
            price=price,
            reason=reason,
        )

    def generate_signal(self, symbol: str, bar: dict[str, Any]) -> Signal:
        c, h, l, v = bar.get("close"), bar.get("high"), bar.get("low"), bar.get("volume")
        if c is None or h is None or l is None:
            return self._hold(symbol, None, "incomplete bar")

        self._ensure(symbol)
        fc, fh, fl = float(c), float(h), float(l)
        fv = float(v or 0.0)

        prev = self._prev_close[symbol]
        obv = self._obv[symbol]
        if prev is not None:
            if fc > prev:
                obv += fv
            elif fc < prev:
                obv -= fv
        self._obv[symbol] = obv
        self._prev_close[symbol] = fc

        self._close[symbol].append(fc)
        self._high[symbol].append(fh)
        self._low[symbol].append(fl)
        self._volume[symbol].append(fv)
        self._obv_hist[symbol].append(obv)

        min_bars = int(self.get_param("min_bars"))
        if len(self._close[symbol]) < min_bars:
            return self._hold(symbol, fc, "warming up")

        vwap = self._rolling_vwap(symbol)
        if vwap is None:
            return self._hold(symbol, fc, "no vwap")

        self._vwap_hist[symbol].append(vwap)
        vh = list(self._vwap_hist[symbol])
        prev_vwap = vh[-2] if len(vh) >= 2 else vwap

        obv_ma_p = int(self.get_param("obv_ma_period"))
        obv_list = list(self._obv_hist[symbol])
        obv_ma = self._sma(obv_list, obv_ma_p)

        vol_ma_p = int(self.get_param("volume_ma_period"))
        vol_list = list(self._volume[symbol])
        vol_ma = self._sma(vol_list, vol_ma_p)

        if obv_ma is None or vol_ma is None or vol_ma <= 0:
            return self._hold(symbol, fc, "obv/vol ma")

        pos = self.get_position(symbol)
        eps = float(self.get_param("momentum_eps") or 1e-9)

        # --- exits: simple opposite VWAP cross ---
        if pos is not None:
            if pos.side.value == "buy" and fc < vwap and prev is not None and prev >= prev_vwap:
                sig = Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=SignalType.LONG_EXIT,
                    strength=0.75,
                    price=fc,
                    reason="量价: 跌破 VWAP 离场",
                    metadata={"vwap": vwap, "obv": obv},
                )
                self.record_signal(sig)
                return sig
            if pos.side.value == "sell" and fc > vwap and prev is not None and prev <= prev_vwap:
                sig = Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=SignalType.SHORT_EXIT,
                    strength=0.75,
                    price=fc,
                    reason="量价: 上穿 VWAP 空头离场",
                    metadata={"vwap": vwap, "obv": obv},
                )
                self.record_signal(sig)
                return sig
            return self._hold(symbol, fc, "holding")

        # Volume-weighted momentum: price vs VWAP
        vw_mom = (fc - vwap) / max(abs(vwap), eps)
        div_n = int(self.get_param("divergence_bars"))
        closes = list(self._close[symbol])
        obvs = list(self._obv_hist[symbol])
        price_down = len(closes) > div_n and closes[-1] < closes[-div_n]
        obv_up = len(obvs) > div_n and obvs[-1] > obvs[-div_n]
        bull_div = price_down and obv_up and fc > vwap
        price_up = len(closes) > div_n and closes[-1] > closes[-div_n]
        obv_down = len(obvs) > div_n and obvs[-1] < obvs[-div_n]
        bear_div = price_up and obv_down and fc < vwap

        vol_ok = fv >= vol_ma

        # Entry: VWAP cross + OBV vs its MA + volume; or OBV–VWAP divergence
        cross_up = prev is not None and prev <= prev_vwap and fc > vwap
        cross_dn = prev is not None and prev >= prev_vwap and fc < vwap

        if (cross_up and obv > obv_ma and vol_ok) or (bull_div and vol_ok):
            strength = min(0.45 + abs(vw_mom) * 5.0, 1.0)
            sig = Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_type=SignalType.LONG_ENTRY,
                strength=round(strength, 4),
                price=fc,
                reason="量价: VWAP 上穿/底背离 + OBV 与放量",
                metadata={
                    "vwap": vwap,
                    "obv": obv,
                    "obv_ma": obv_ma,
                    "vw_momentum": vw_mom,
                    "bull_div": bull_div,
                },
            )
            self.record_signal(sig)
            return sig

        if (cross_dn and obv < obv_ma and vol_ok) or (bear_div and vol_ok):
            strength = min(0.45 + abs(vw_mom) * 5.0, 1.0)
            sig = Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_type=SignalType.SHORT_ENTRY,
                strength=round(strength, 4),
                price=fc,
                reason="量价: VWAP 下穿/顶背离 + OBV 与放量",
                metadata={
                    "vwap": vwap,
                    "obv": obv,
                    "obv_ma": obv_ma,
                    "vw_momentum": vw_mom,
                    "bear_div": bear_div,
                },
            )
            self.record_signal(sig)
            return sig

        return self._hold(symbol, fc, "no signal")

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        sig = self.generate_signal(symbol, bar)
        if sig.signal_type == SignalType.HOLD:
            return []
        return [sig]

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        for sym in self.config.symbols:
            b = market_data.get(sym)
            if b:
                out.extend(await self.on_bar(sym, b))
        return out
