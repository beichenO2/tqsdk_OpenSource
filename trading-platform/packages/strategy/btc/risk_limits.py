"""BTC 专用风控规则 - 扩展 Ch27 的 RiskLimit 基类。

针对加密货币市场特点增加的风控规则:
- 波动率过高暂停交易
- 交易所价差过大拒绝下单
- 资金费率极端时限制方向
"""

from __future__ import annotations

import logging
import math
from collections import deque
from decimal import Decimal

from core.enums.direction import Direction, Offset
from core.enums.order_type import OrderType
from execution.order_manager import OrderRequest
from risk.limits import RiskContext, RiskLimit

logger = logging.getLogger(__name__)


class VolatilityCircuitBreaker(RiskLimit):
    """波动率熔断: 当价格短期波动率超过阈值时拒绝新开仓。"""

    def __init__(
        self,
        max_volatility_pct: Decimal = Decimal("0.10"),
        lookback_bars: int = 20,
        recovery_pct: Decimal | None = None,
        cooldown_bars: int = 5,
    ) -> None:
        self._max_volatility = max_volatility_pct
        self._lookback = lookback_bars
        self._recovery_pct = recovery_pct if recovery_pct is not None else max_volatility_pct * Decimal("0.8")
        self._cooldown_bars = cooldown_bars
        self._price_history: dict[str, deque[Decimal]] = {}
        self._tripped: dict[str, int] = {}

    @property
    def name(self) -> str:
        return "BTC_VolatilityCircuitBreaker"

    def feed_price(self, instrument: str, price: Decimal) -> None:
        if instrument not in self._price_history:
            self._price_history[instrument] = deque(maxlen=self._lookback + 1)
        self._price_history[instrument].append(price)

    def _calc_volatility(self, instrument: str) -> Decimal | None:
        prices = self._price_history.get(instrument)
        if prices is None or len(prices) < 3:
            return None
        price_list = list(prices)
        returns = [
            float((price_list[i] - price_list[i - 1]) / price_list[i - 1])
            for i in range(1, len(price_list))
            if price_list[i - 1] != 0
        ]
        if not returns:
            return None
        mean_r = sum(returns) / len(returns)
        variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
        return Decimal(str(math.sqrt(variance)))

    def check(self, request: OrderRequest, context: RiskContext) -> tuple[bool, str]:
        if request.offset != Offset.OPEN:
            return True, ""

        sym = request.symbol
        vol = self._calc_volatility(sym)

        remaining = self._tripped.get(sym, 0)
        if remaining > 0:
            if vol is not None and vol <= self._recovery_pct:
                self._tripped[sym] = 0
                logger.info("波动率熔断恢复: %s vol=%.4f", sym, float(vol))
            else:
                self._tripped[sym] = remaining - 1
                return False, (
                    f"BTC 波动率熔断冷却中 (剩余 {remaining} bars)"
                )

        if vol is not None and vol > self._max_volatility:
            self._tripped[sym] = self._cooldown_bars
            logger.warning(
                "波动率熔断: %s vol=%.4f > max=%.4f",
                sym, float(vol), float(self._max_volatility),
            )
            return False, (
                f"BTC 波动率 {float(vol):.2%} 超过阈值 {float(self._max_volatility):.2%}"
            )
        return True, ""


class SpreadLimit(RiskLimit):
    """买卖价差限制: 当盘口价差过大时拒绝市价单。"""

    def __init__(self, max_spread_pct: Decimal = Decimal("0.005")) -> None:
        self._max_spread = max_spread_pct
        self._spreads: dict[str, Decimal] = {}

    @property
    def name(self) -> str:
        return "BTC_SpreadLimit"

    def update_spread(self, instrument: str, bid: Decimal, ask: Decimal) -> None:
        mid = (bid + ask) / 2
        if mid > 0:
            self._spreads[instrument] = (ask - bid) / mid

    def check(self, request: OrderRequest, context: RiskContext) -> tuple[bool, str]:
        if getattr(request, "order_type", None) != OrderType.MARKET:
            return True, ""

        spread = self._spreads.get(request.symbol)
        if spread is not None and spread > self._max_spread:
            return False, (
                f"盘口价差 {float(spread):.4%} 超过阈值 {float(self._max_spread):.4%}，"
                f"拒绝市价单"
            )
        return True, ""


class FundingRateLimit(RiskLimit):
    """资金费率限制: 当资金费率极端时限制不利方向开仓。

    正资金费率 > 阈值时拒绝开多（做多需付费）。
    负资金费率 < -阈值时拒绝开空（做空需付费）。
    """

    def __init__(self, max_funding_rate: Decimal = Decimal("0.003")) -> None:
        self._max_rate = max_funding_rate
        self._funding_rates: dict[str, Decimal] = {}

    @property
    def name(self) -> str:
        return "BTC_FundingRateLimit"

    def update_funding_rate(self, instrument: str, rate: Decimal) -> None:
        self._funding_rates[instrument] = rate

    def check(self, request: OrderRequest, context: RiskContext) -> tuple[bool, str]:
        if request.offset != Offset.OPEN:
            return True, ""

        rate = self._funding_rates.get(request.symbol)
        if rate is None:
            return True, ""

        if rate > self._max_rate and request.direction == Direction.LONG:
            return False, (
                f"资金费率 {float(rate):.4%} 过高，拒绝开多"
            )
        if rate < -self._max_rate and request.direction == Direction.SHORT:
            return False, (
                f"资金费率 {float(rate):.4%} 过低(负)，拒绝开空"
            )
        return True, ""


class CryptoPositionValueLimit(RiskLimit):
    """加密货币持仓市值限制: 单品种持仓市值不超过账户总值的指定比例。"""

    def __init__(self, max_position_value_pct: Decimal = Decimal("0.25")) -> None:
        self._max_pct = max_position_value_pct

    @property
    def name(self) -> str:
        return "BTC_PositionValueLimit"

    def check(self, request: OrderRequest, context: RiskContext) -> tuple[bool, str]:
        if request.offset != Offset.OPEN:
            return True, ""

        if context.account_balance <= 0:
            return True, ""

        last_price = context.last_prices.get(request.symbol, Decimal("0"))
        if last_price <= 0:
            return True, ""

        current_position_value = Decimal("0")
        for key, pos in context.positions.items():
            if key.startswith(request.symbol + ":"):
                current_position_value += Decimal(str(pos.volume)) * last_price

        new_value = current_position_value + Decimal(str(request.volume)) * request.price
        ratio = new_value / context.account_balance

        if ratio > self._max_pct:
            return False, (
                f"持仓市值占比 {float(ratio):.2%} 将超过限制 {float(self._max_pct):.2%}"
            )
        return True, ""


class LeverageLimit(RiskLimit):
    """杠杆倍数限制: 拒绝超过最大允许杠杆的开仓请求。

    加密货币交易所允许高达 125x 杠杆，但高杠杆极大放大了爆仓风险。
    本规则在下单前验证实际杠杆是否在安全范围内。
    """

    def __init__(self, max_leverage: Decimal = Decimal("10")) -> None:
        self._max_leverage = max_leverage

    @property
    def name(self) -> str:
        return "BTC_LeverageLimit"

    def check(self, request: OrderRequest, context: RiskContext) -> tuple[bool, str]:
        if request.offset != Offset.OPEN:
            return True, ""

        if context.account_balance <= 0:
            return True, ""

        last_price = context.last_prices.get(request.symbol, request.price)
        if last_price <= 0:
            return True, ""

        total_notional = Decimal("0")
        for key, pos in context.positions.items():
            sym = key.split(":")[0] if ":" in key else key
            p = context.last_prices.get(sym, Decimal("0"))
            total_notional += Decimal(str(pos.volume)) * p

        new_notional = total_notional + Decimal(str(request.volume)) * last_price
        effective_leverage = new_notional / context.account_balance

        if effective_leverage > self._max_leverage:
            return False, (
                f"有效杠杆 {float(effective_leverage):.1f}x "
                f"将超过最大允许 {float(self._max_leverage):.0f}x"
            )
        return True, ""


class LiquidationGuard(RiskLimit):
    """强平预警: 当保证金率接近强平线时拒绝新开仓。

    maintenance_margin_pct: 维持保证金率（交易所通常 0.5%~2%）
    safety_buffer: 在维持保证金率之上额外保留的安全缓冲（默认 5%）
    """

    def __init__(
        self,
        maintenance_margin_pct: Decimal = Decimal("0.01"),
        safety_buffer: Decimal = Decimal("0.05"),
    ) -> None:
        self._maint_pct = maintenance_margin_pct
        self._buffer = safety_buffer

    @property
    def name(self) -> str:
        return "BTC_LiquidationGuard"

    def check(self, request: OrderRequest, context: RiskContext) -> tuple[bool, str]:
        if request.offset != Offset.OPEN:
            return True, ""

        if context.account_balance <= 0:
            return True, ""

        total_notional = Decimal("0")
        total_unrealized_loss = Decimal("0")
        for key, pos in context.positions.items():
            sym = key.split(":")[0] if ":" in key else key
            p = context.last_prices.get(sym, Decimal("0"))
            notional = Decimal(str(pos.volume)) * p
            total_notional += notional
            if hasattr(pos, "unrealized_pnl") and pos.unrealized_pnl < 0:
                total_unrealized_loss += Decimal(str(abs(pos.unrealized_pnl)))

        last_price = context.last_prices.get(request.symbol, request.price)
        new_notional = total_notional + Decimal(str(request.volume)) * last_price
        required_maintenance = new_notional * self._maint_pct
        threshold = required_maintenance * (1 + self._buffer / self._maint_pct)

        available = context.account_balance - total_unrealized_loss
        if available < threshold:
            margin_ratio = (
                float(available / new_notional * 100) if new_notional > 0 else 0
            )
            return False, (
                f"保证金率 {margin_ratio:.2f}% 过低，接近强平线 "
                f"(维持={float(self._maint_pct)*100:.1f}%+缓冲={float(self._buffer)*100:.1f}%)，"
                f"拒绝新开仓"
            )
        return True, ""
