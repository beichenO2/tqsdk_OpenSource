"""回测报告生成器 - 计算各项绩效指标。"""

from __future__ import annotations

import math
from decimal import Decimal
from datetime import datetime

from .models import (
    BacktestConfig,
    BacktestResult,
    EquityCurvePoint,
    Trade,
    OrderSide,
)


TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = Decimal("0.03")


class ReportGenerator:
    """根据权益曲线和交易记录生成完整回测报告。"""

    def generate(
        self,
        config: BacktestConfig,
        trades: list[Trade],
        equity_curve: list[EquityCurvePoint],
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> BacktestResult:
        result = BacktestResult(config=config, trades=trades, equity_curve=equity_curve)
        result.start_time = start_time
        result.end_time = end_time
        if start_time and end_time:
            result.elapsed_seconds = (end_time - start_time).total_seconds()

        result.total_trades = len(trades)

        if not equity_curve:
            return result

        result.final_equity = equity_curve[-1].equity
        result.total_return = (
            (result.final_equity - config.initial_capital) / config.initial_capital
            if config.initial_capital
            else Decimal(0)
        )

        self._calc_drawdown(equity_curve, result)
        self._calc_annual_return(config, equity_curve, result)
        self._calc_sharpe(equity_curve, result)
        self._calc_sortino(equity_curve, result)
        self._calc_trade_stats(trades, config, result)
        self._calc_calmar(result)

        return result

    def _calc_drawdown(self, curve: list[EquityCurvePoint], result: BacktestResult) -> None:
        peak = Decimal(0)
        max_dd = Decimal(0)
        max_dd_pct = Decimal(0)

        for pt in curve:
            if pt.equity > peak:
                peak = pt.equity
            dd = peak - pt.equity
            dd_pct = dd / peak if peak else Decimal(0)
            pt.drawdown = dd
            pt.drawdown_pct = dd_pct
            if dd > max_dd:
                max_dd = dd
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

        result.max_drawdown = max_dd
        result.max_drawdown_pct = max_dd_pct

    def _calc_annual_return(
        self,
        config: BacktestConfig,
        curve: list[EquityCurvePoint],
        result: BacktestResult,
    ) -> None:
        if len(curve) < 2 or not config.initial_capital:
            return
        days = (curve[-1].dt - curve[0].dt).days
        if days <= 0:
            return
        total_return_float = float(result.total_return)
        annual = (1 + total_return_float) ** (365.0 / days) - 1
        result.annual_return = Decimal(str(round(annual, 6)))

    def _calc_sharpe(self, curve: list[EquityCurvePoint], result: BacktestResult) -> None:
        if len(curve) < 2:
            return
        returns = self._daily_returns(curve)
        if len(returns) < 2:
            return
        rf_daily = float(RISK_FREE_RATE) / TRADING_DAYS_PER_YEAR
        excess = [r - rf_daily for r in returns]
        avg_excess = sum(excess) / len(excess)
        variance = sum((e - avg_excess) ** 2 for e in excess) / (len(excess) - 1)
        std = math.sqrt(variance) if variance > 0 else 0
        if std > 0:
            result.sharpe_ratio = Decimal(str(round(avg_excess / std * math.sqrt(TRADING_DAYS_PER_YEAR), 4)))

    def _calc_sortino(self, curve: list[EquityCurvePoint], result: BacktestResult) -> None:
        if len(curve) < 2:
            return
        returns = self._daily_returns(curve)
        n = len(returns)
        if n < 2:
            return
        rf_daily = float(RISK_FREE_RATE) / TRADING_DAYS_PER_YEAR
        excess = [r - rf_daily for r in returns]
        avg_excess = sum(excess) / n
        downside_sq = sum(min(e, 0.0) ** 2 for e in excess)
        if downside_sq == 0:
            return
        downside_std = math.sqrt(downside_sq / (n - 1))
        if downside_std > 0:
            result.sortino_ratio = Decimal(str(round(avg_excess / downside_std * math.sqrt(TRADING_DAYS_PER_YEAR), 4)))

    def _calc_trade_stats(
        self,
        trades: list[Trade],
        config: BacktestConfig,
        result: BacktestResult,
    ) -> None:
        if not trades:
            return

        trade_pairs = self._pair_trades(trades, multiplier=config.contract_multiplier if hasattr(config, 'contract_multiplier') and config.contract_multiplier else 1)
        if not trade_pairs:
            return

        pnls = [p["pnl"] for p in trade_pairs]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        result.win_rate = Decimal(str(round(len(wins) / len(pnls), 4))) if pnls else Decimal(0)
        result.avg_trade_pnl = Decimal(str(round(sum(pnls) / len(pnls), 2))) if pnls else Decimal(0)

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        result.profit_factor = (
            Decimal(str(round(gross_profit / gross_loss, 4))) if gross_loss > 0 else Decimal("999")
        )

        holding_periods = [p["holding_seconds"] for p in trade_pairs if p["holding_seconds"] > 0]
        if holding_periods:
            result.avg_holding_period = sum(holding_periods) / len(holding_periods)

    def _calc_calmar(self, result: BacktestResult) -> None:
        if result.max_drawdown_pct and result.max_drawdown_pct > 0:
            result.calmar_ratio = Decimal(
                str(round(float(result.annual_return) / float(result.max_drawdown_pct), 4))
            )

    @staticmethod
    def _daily_returns(curve: list[EquityCurvePoint]) -> list[float]:
        equities = [float(pt.equity) for pt in curve]
        returns: list[float] = []
        for i in range(1, len(equities)):
            prev = equities[i - 1]
            if prev > 0:
                returns.append((equities[i] - prev) / prev)
        return returns

    def _pair_trades(self, trades: list[Trade], multiplier: int = 1) -> list[dict]:
        """将成交配对为完整的交易回合（开仓→平仓），支持部分成交。"""
        from collections import deque
        open_trades: dict[str, deque[list]] = {}
        pairs: list[dict] = []

        for t in sorted(trades, key=lambda x: x.dt):
            key = t.symbol
            stack = open_trades.setdefault(key, deque())
            remaining = t.volume

            if not stack or stack[0][0].side == t.side:
                stack.append([t, t.volume])
                continue

            while remaining > 0 and stack and stack[0][0].side != t.side:
                opener, open_rem = stack[0]
                qty = min(open_rem, remaining)

                if opener.side == OrderSide.BUY:
                    pnl = float((t.price - opener.price) * qty * multiplier)
                else:
                    pnl = float((opener.price - t.price) * qty * multiplier)

                open_comm_share = float(opener.commission) * qty / opener.volume if opener.volume else 0
                close_comm_share = float(t.commission) * qty / t.volume if t.volume else 0
                pnl -= (open_comm_share + close_comm_share)
                holding = (t.dt - opener.dt).total_seconds()
                pairs.append({"pnl": pnl, "holding_seconds": holding})

                open_rem -= qty
                remaining -= qty
                if open_rem <= 0:
                    stack.popleft()
                else:
                    stack[0] = [opener, open_rem]

            if remaining > 0:
                stack.append([t, remaining])

        return pairs
