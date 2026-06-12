"""CTA 趋势跟踪 — 唐奇安通道突破 + ATR 波动过滤 + ATR 仓位与追踪止损。"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..indicators import calc_atr
from ..registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "entry_period": 20,
    "exit_period": 10,
    "atr_period": 14,
    "atr_filter_mult": 1.0,
    "risk_per_trade": 0.02,
    "trailing_stop_atr_mult": 2.5,
    "contract_multiplier": 1.0,
}


@auto_register("cta_trend")
class CTATrendStrategy(BaseStrategy):
    """唐奇安通道趋势策略。

    - 入场：收盘价突破 N 日（不含当根）高点做多 / 低点做空
    - 出场：更短周期对侧通道破位（多仓跌破短周期下轨等）
    - 仅当 ATR 不低于历史均量×倍数时开仓，过滤窄幅震荡
    - 建议仓位：`(权益 × risk_per_trade) / ATR`，再按合约乘数缩放
    """

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._close_history: dict[str, deque[float]] = {}
        self._high_history: dict[str, deque[float]] = {}
        self._low_history: dict[str, deque[float]] = {}
        self._atr_history: dict[str, deque[float]] = {}
        self._peak_price: dict[str, float] = {}
        self._trough_price: dict[str, float] = {}
        self._context_equity: float | None = None

    def _ensure_buffers(self, symbol: str) -> None:
        entry_p = int(self.get_param("entry_period"))
        atr_p = int(self.get_param("atr_period"))
        max_len = max(entry_p, atr_p) * 3 + 20
        if symbol not in self._close_history:
            self._close_history[symbol] = deque(maxlen=max_len)
            self._high_history[symbol] = deque(maxlen=max_len)
            self._low_history[symbol] = deque(maxlen=max_len)
            self._atr_history[symbol] = deque(maxlen=atr_p * 4 + 10)

    def _calc_atr(self, symbol: str) -> float | None:
        return calc_atr(
            self._high_history[symbol],
            self._low_history[symbol],
            self._close_history[symbol],
            int(self.get_param("atr_period")),
        )

    def _donchian_high_excl_last(self, symbol: str, n: int) -> float | None:
        highs = list(self._high_history[symbol])
        if len(highs) <= n:
            return None
        window = highs[-(n + 1) : -1]
        if len(window) < n:
            return None
        return max(window)

    def _donchian_low_excl_last(self, symbol: str, n: int) -> float | None:
        lows = list(self._low_history[symbol])
        if len(lows) <= n:
            return None
        window = lows[-(n + 1) : -1]
        if len(window) < n:
            return None
        return min(window)

    def _atr_volatility_ok(self, symbol: str, atr: float) -> bool:
        hist = self._atr_history[symbol]
        atr_p = int(self.get_param("atr_period"))
        mult = float(self.get_param("atr_filter_mult") or 1.0)
        if mult <= 0:
            return True
        if len(hist) < atr_p:
            return False
        recent = list(hist)[-atr_p:]
        mean_atr = sum(recent) / len(recent)
        if mean_atr <= 0:
            return False
        return atr >= mult * mean_atr

    def _suggested_qty(self, symbol: str, close: float, atr: float) -> float | None:
        equity = self._context_equity
        if equity is None or equity <= 0 or atr <= 0:
            return None
        risk = float(self.get_param("risk_per_trade") or 0.0)
        mult = float(self.get_param("contract_multiplier") or 1.0)
        if mult <= 0:
            mult = 1.0
        # 风险预算 / 每点波动风险（简化：ATR 代表单合约大致波动尺度）
        budget = float(equity) * risk
        denom = max(atr * mult, 1e-12)
        qty = budget / denom
        return max(qty, 0.0)

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        close = bar.get("close")
        high = bar.get("high")
        low = bar.get("low")
        if close is None or high is None or low is None:
            logger.debug("CTA 跳过不完整 K 线: %s", symbol)
            return []

        self._ensure_buffers(symbol)
        fc, fh, fl = float(close), float(high), float(low)

        self._close_history[symbol].append(fc)
        self._high_history[symbol].append(fh)
        self._low_history[symbol].append(fl)

        atr = self._calc_atr(symbol)
        if atr is not None and atr > 0:
            self._atr_history[symbol].append(atr)

        entry_n = int(self.get_param("entry_period"))
        exit_n = int(self.get_param("exit_period"))

        signals: list[Signal] = []
        pos = self.get_position(symbol)

        d_hi = self._donchian_high_excl_last(symbol, entry_n)
        d_lo = self._donchian_low_excl_last(symbol, entry_n)
        x_hi = self._donchian_high_excl_last(symbol, exit_n)
        x_lo = self._donchian_low_excl_last(symbol, exit_n)

        if pos is None and atr is not None and atr > 0 and d_hi is not None and d_lo is not None:
            vol_ok = self._atr_volatility_ok(symbol, atr)
            if not vol_ok:
                pass
            elif vol_ok and fc > d_hi and d_hi > 0:
                strength = min((fc - d_hi) / max(d_hi, 1e-12) * 10 + 0.4, 1.0)
                sq = self._suggested_qty(symbol, fc, atr)
                sig = Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=round(strength, 4),
                    price=fc,
                    suggested_qty=sq,
                    reason=f"唐奇安上破 entry={entry_n} 高点={d_hi:.4f} ATR={atr:.4f}",
                    metadata={
                        "donchian_high": d_hi,
                        "atr": atr,
                        "entry_period": entry_n,
                    },
                )
                signals.append(sig)
                self.record_signal(sig)
                self._peak_price[symbol] = fc
                logger.info("[%s] 多头入场 %s", symbol, sig.reason)

            elif vol_ok and fc < d_lo:
                strength = min((d_lo - fc) / max(abs(d_lo), 1e-12) * 10 + 0.4, 1.0)
                sq = self._suggested_qty(symbol, fc, atr)
                sig = Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=round(strength, 4),
                    price=fc,
                    suggested_qty=sq,
                    reason=f"唐奇安下破 entry={entry_n} 低点={d_lo:.4f} ATR={atr:.4f}",
                    metadata={
                        "donchian_low": d_lo,
                        "atr": atr,
                        "entry_period": entry_n,
                    },
                )
                signals.append(sig)
                self.record_signal(sig)
                self._trough_price[symbol] = fc
                logger.info("[%s] 空头入场 %s", symbol, sig.reason)

        if pos is not None and atr is not None and atr > 0:
            stop_mult = float(self.get_param("trailing_stop_atr_mult") or 2.5)

            if pos.side.value == "buy":
                self._peak_price[symbol] = max(self._peak_price.get(symbol, fc), fc)
                trail = self._peak_price[symbol] - atr * stop_mult
                exit_channel = x_lo is not None and fc < x_lo

                if exit_channel:
                    sig = Signal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=SignalType.LONG_EXIT,
                        strength=0.85,
                        price=fc,
                        reason=f"短周期({exit_n})下轨离场 close={fc:.4f} < 通道低={x_lo:.4f}",
                        metadata={"exit_low": x_lo},
                    )
                    signals.append(sig)
                    self.record_signal(sig)
                    self._peak_price.pop(symbol, None)
                    logger.info("[%s] 平多(通道) %s", symbol, sig.reason)
                elif fc < trail:
                    sig = Signal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=SignalType.LONG_EXIT,
                        strength=0.9,
                        price=fc,
                        reason=f"ATR 追踪止损 trail={trail:.4f}",
                        metadata={"trailing_stop": trail, "atr": atr},
                    )
                    signals.append(sig)
                    self.record_signal(sig)
                    self._peak_price.pop(symbol, None)
                    logger.info("[%s] 平多(ATR) %s", symbol, sig.reason)

            elif pos.side.value == "sell":
                self._trough_price[symbol] = min(self._trough_price.get(symbol, fc), fc)
                trail = self._trough_price[symbol] + atr * stop_mult
                exit_channel = x_hi is not None and fc > x_hi

                if exit_channel:
                    sig = Signal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=SignalType.SHORT_EXIT,
                        strength=0.85,
                        price=fc,
                        reason=f"短周期({exit_n})上轨离场 close={fc:.4f} > 通道高={x_hi:.4f}",
                        metadata={"exit_high": x_hi},
                    )
                    signals.append(sig)
                    self.record_signal(sig)
                    self._trough_price.pop(symbol, None)
                    logger.info("[%s] 平空(通道) %s", symbol, sig.reason)
                elif fc > trail:
                    sig = Signal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=SignalType.SHORT_EXIT,
                        strength=0.9,
                        price=fc,
                        reason=f"ATR 追踪止损 trail={trail:.4f}",
                        metadata={"trailing_stop": trail, "atr": atr},
                    )
                    signals.append(sig)
                    self.record_signal(sig)
                    self._trough_price.pop(symbol, None)
                    logger.info("[%s] 平空(ATR) %s", symbol, sig.reason)

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        eq = market_data.get("equity")
        self._context_equity = float(eq) if eq is not None else None

        all_signals: list[Signal] = []
        for symbol in self.config.symbols:
            bar = market_data.get(symbol)
            if bar:
                sigs = await self.on_bar(symbol, bar)
                all_signals.extend(sigs)
        return all_signals
