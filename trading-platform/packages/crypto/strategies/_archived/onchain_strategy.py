"""On-chain data driven BTC strategy.

Uses derivatives market metrics (open interest, funding rates, liquidations,
long/short ratio) as contrarian and momentum signals. These metrics come
from CoinAnk and are not available from traditional OHLCV data.

Signal logic:
- Extreme funding rate → contrarian signal (everyone leaning one way)
- OI spike + price divergence → exhaustion/breakout signal
- Liquidation cascade → capitulation bottom / forced selling top
- L/S ratio extreme → crowding indicator
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig
from strategy.registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "funding_extreme_threshold": 0.001,
    "funding_contrarian_strength": 0.7,
    "oi_lookback": 20,
    "oi_surge_ratio": 1.3,
    "oi_divergence_threshold": 0.02,
    "liquidation_surge_ratio": 3.0,
    "ls_ratio_extreme_low": 0.4,
    "ls_ratio_extreme_high": 2.5,
    "confirmation_window": 3,
    "cooldown_bars": 5,
}


@auto_register("btc_onchain")
class BTCOnChainStrategy(BaseStrategy):
    """BTC strategy driven by on-chain and derivatives market data.

    Combines four signal sources:
    1. Funding rate extremes (contrarian)
    2. Open interest divergence from price
    3. Liquidation cascades
    4. Long/short ratio crowding
    """

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        buf = max(self.get_param("oi_lookback"), 50) + 10
        self._funding_history: dict[str, deque[float]] = {}
        self._oi_history: dict[str, deque[float]] = {}
        self._price_history: dict[str, deque[float]] = {}
        self._liq_long_history: dict[str, deque[float]] = {}
        self._liq_short_history: dict[str, deque[float]] = {}
        self._ls_ratio_history: dict[str, deque[float]] = {}
        self._buf_len = buf
        self._cooldown: dict[str, int] = {}

    def _ensure_buffers(self, symbol: str) -> None:
        if symbol not in self._funding_history:
            self._funding_history[symbol] = deque(maxlen=self._buf_len)
            self._oi_history[symbol] = deque(maxlen=self._buf_len)
            self._price_history[symbol] = deque(maxlen=self._buf_len)
            self._liq_long_history[symbol] = deque(maxlen=self._buf_len)
            self._liq_short_history[symbol] = deque(maxlen=self._buf_len)
            self._ls_ratio_history[symbol] = deque(maxlen=self._buf_len)
            self._cooldown[symbol] = 0

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._ensure_buffers(symbol)

        price = bar["close"]
        self._price_history[symbol].append(price)

        funding = bar.get("funding_rate", 0.0)
        oi = bar.get("open_interest", 0.0)
        liq_long = bar.get("liquidation_long", 0.0)
        liq_short = bar.get("liquidation_short", 0.0)
        ls_ratio = bar.get("long_short_ratio", 1.0)

        self._funding_history[symbol].append(funding)
        self._oi_history[symbol].append(oi)
        self._liq_long_history[symbol].append(liq_long)
        self._liq_short_history[symbol].append(liq_short)
        self._ls_ratio_history[symbol].append(ls_ratio)

        if self._cooldown[symbol] > 0:
            self._cooldown[symbol] -= 1
            return []

        candidates: list[Signal] = []

        sig = self._check_funding_signal(symbol, price, funding)
        if sig:
            candidates.append(sig)

        sig = self._check_oi_divergence(symbol, price)
        if sig:
            candidates.append(sig)

        sig = self._check_liquidation_cascade(symbol, price, liq_long, liq_short)
        if sig:
            candidates.append(sig)

        sig = self._check_ls_ratio(symbol, price, ls_ratio)
        if sig:
            candidates.append(sig)

        if not candidates:
            return []

        # Resolve conflicting directions: keep only the strongest signal
        best = max(candidates, key=lambda s: s.strength)
        self._cooldown[symbol] = self.get_param("cooldown_bars")
        self.record_signal(best)
        return [best]

    def _check_funding_signal(
        self, symbol: str, price: float, funding: float
    ) -> Signal | None:
        threshold = self.get_param("funding_extreme_threshold")
        if abs(funding) < threshold:
            return None

        if funding > threshold:
            return Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_type=SignalType.SHORT_ENTRY,
                strength=self.get_param("funding_contrarian_strength"),
                price=price,
                reason=f"资金费率极高({funding:.4%})→空头反向信号",
                metadata={"funding_rate": funding, "signal_source": "funding"},
            )
        else:
            return Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_type=SignalType.LONG_ENTRY,
                strength=self.get_param("funding_contrarian_strength"),
                price=price,
                reason=f"资金费率极低({funding:.4%})→多头反向信号",
                metadata={"funding_rate": funding, "signal_source": "funding"},
            )

    def _check_oi_divergence(self, symbol: str, price: float) -> Signal | None:
        lookback = self.get_param("oi_lookback")
        oi_hist = list(self._oi_history[symbol])
        price_hist = list(self._price_history[symbol])

        if len(oi_hist) < lookback or len(price_hist) < lookback:
            return None

        oi_window = oi_hist[-lookback:]
        price_window = price_hist[-lookback:]

        oi_mean = sum(oi_window) / len(oi_window)
        if oi_mean == 0:
            return None

        oi_change = (oi_hist[-1] - oi_mean) / oi_mean
        price_change = (price_hist[-1] - price_window[0]) / price_window[0] if price_window[0] != 0 else 0
        threshold = self.get_param("oi_divergence_threshold")
        surge = self.get_param("oi_surge_ratio")

        if oi_hist[-1] > oi_mean * surge:
            if price_change > threshold and oi_change > 0:
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=min(oi_change * 2, 0.9),
                    price=price,
                    reason=f"OI放大({oi_change:.1%})+价格上涨({price_change:.1%})→趋势强化",
                    metadata={"oi_change": oi_change, "price_change": price_change, "signal_source": "oi"},
                )
            elif price_change < -threshold and oi_change > 0:
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=min(oi_change * 2, 0.9),
                    price=price,
                    reason=f"OI放大({oi_change:.1%})+价格下跌({price_change:.1%})→空头趋势",
                    metadata={"oi_change": oi_change, "price_change": price_change, "signal_source": "oi"},
                )
        return None

    def _check_liquidation_cascade(
        self, symbol: str, price: float, liq_long: float, liq_short: float
    ) -> Signal | None:
        liq_hist_l = list(self._liq_long_history[symbol])
        liq_hist_s = list(self._liq_short_history[symbol])

        if len(liq_hist_l) < 5:
            return None

        avg_liq_long = sum(liq_hist_l[-20:]) / min(len(liq_hist_l), 20) or 1
        avg_liq_short = sum(liq_hist_s[-20:]) / min(len(liq_hist_s), 20) or 1
        surge = self.get_param("liquidation_surge_ratio")

        if liq_long > avg_liq_long * surge:
            return Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_type=SignalType.LONG_ENTRY,
                strength=min(liq_long / avg_liq_long / 5, 0.85),
                price=price,
                reason=f"多头爆仓潮({liq_long / avg_liq_long:.1f}x均值)→触底反转信号",
                metadata={
                    "liq_long": liq_long, "avg_liq_long": avg_liq_long,
                    "signal_source": "liquidation",
                },
            )

        if liq_short > avg_liq_short * surge:
            return Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_type=SignalType.SHORT_ENTRY,
                strength=min(liq_short / avg_liq_short / 5, 0.85),
                price=price,
                reason=f"空头爆仓潮({liq_short / avg_liq_short:.1f}x均值)→顶部反转信号",
                metadata={
                    "liq_short": liq_short, "avg_liq_short": avg_liq_short,
                    "signal_source": "liquidation",
                },
            )

        return None

    def _check_ls_ratio(
        self, symbol: str, price: float, ls_ratio: float
    ) -> Signal | None:
        low = self.get_param("ls_ratio_extreme_low")
        high = self.get_param("ls_ratio_extreme_high")

        if ls_ratio <= low:
            return Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_type=SignalType.LONG_ENTRY,
                strength=0.65,
                price=price,
                reason=f"多空比极低({ls_ratio:.2f})→过度看空反转",
                metadata={"ls_ratio": ls_ratio, "signal_source": "ls_ratio"},
            )
        elif ls_ratio >= high:
            return Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_type=SignalType.SHORT_ENTRY,
                strength=0.65,
                price=price,
                reason=f"多空比极高({ls_ratio:.2f})→过度看多反转",
                metadata={"ls_ratio": ls_ratio, "signal_source": "ls_ratio"},
            )
        return None

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        all_signals: list[Signal] = []
        for symbol in self.config.symbols:
            bar = market_data.get(symbol)
            if bar:
                sigs = await self.on_bar(symbol, bar)
                all_signals.extend(sigs)
        return all_signals
