"""Post-backtest performance analysis.

Computes risk-adjusted metrics, drawdown analysis, and trade statistics
from raw equity curves and fill records.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Sequence

from ..models.types import (
    BacktestConfig,
    BacktestResult,
    Fill,
    OrderSide,
    PerformanceMetrics,
)

_ZERO = Decimal(0)
_ANNUAL_HOURS = 365.25 * 24
_RISK_FREE_ANNUAL = 0.03


class BacktestAnalyzer:
    """Compute performance metrics from backtest output."""

    def __init__(
        self,
        config: BacktestConfig,
        equity_curve: list[tuple[datetime, Decimal]],
        fills: list[Fill],
        started_at: datetime,
        finished_at: datetime,
    ) -> None:
        self._config = config
        self._equity_curve = equity_curve
        self._fills = fills
        self._started_at = started_at
        self._finished_at = finished_at

    def compute(self) -> BacktestResult:
        """Build full BacktestResult with computed metrics."""
        metrics = self._compute_metrics()
        daily_returns = self._compute_daily_returns()
        return BacktestResult(
            config=self._config,
            metrics=metrics,
            equity_curve=self._equity_curve,
            fills=self._fills,
            daily_returns=daily_returns,
            started_at=self._started_at,
            finished_at=self._finished_at,
        )

    def _compute_metrics(self) -> PerformanceMetrics:
        if len(self._equity_curve) < 2:
            return PerformanceMetrics()

        initial_equity = self._equity_curve[0][1]
        final_equity = self._equity_curve[-1][1]
        total_return = (final_equity - initial_equity) / initial_equity if initial_equity else _ZERO

        bt_start = self._equity_curve[0][0]
        bt_end = self._equity_curve[-1][0]
        bt_duration = bt_end - bt_start
        hours = max(bt_duration.total_seconds() / 3600, 1)
        annualized = self._annualize(total_return, hours)

        returns = self._returns_series()
        sharpe = self._sharpe_ratio(returns)
        sortino = self._sortino_ratio(returns)
        vol = self._volatility(returns)

        dd, dd_dur = self._max_drawdown()
        trades = self._trade_stats()

        total_commission = sum(f.commission for f in self._fills)
        total_slippage = sum(f.slippage for f in self._fills)

        calmar = float(annualized) / float(dd) if dd > 0 else 0.0

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=dd,
            max_drawdown_duration_days=dd_dur,
            win_rate=trades["win_rate"],
            profit_factor=trades["profit_factor"],
            total_trades=trades["total"],
            avg_trade_pnl=trades["avg_pnl"],
            avg_win=trades["avg_win"],
            avg_loss=trades["avg_loss"],
            calmar_ratio=calmar,
            volatility=vol,
            avg_holding_period_hours=trades["avg_hold_hours"],
            total_commission=total_commission,
            total_slippage=total_slippage,
        )

    def _returns_series(self) -> list[float]:
        returns: list[float] = []
        for i in range(1, len(self._equity_curve)):
            prev = self._equity_curve[i - 1][1]
            curr = self._equity_curve[i][1]
            if prev != 0:
                returns.append(float((curr - prev) / prev))
        return returns

    def _compute_daily_returns(self) -> list[tuple[datetime, float]]:
        if not self._equity_curve:
            return []

        daily: dict[str, tuple[datetime, Decimal, Decimal]] = {}
        for ts, eq in self._equity_curve:
            day_key = ts.strftime("%Y-%m-%d")
            if day_key not in daily:
                daily[day_key] = (ts, eq, eq)
            else:
                daily[day_key] = (daily[day_key][0], daily[day_key][1], eq)

        result: list[tuple[datetime, float]] = []
        sorted_days = sorted(daily.items())
        for i in range(len(sorted_days)):
            day_ts = sorted_days[i][1][0]
            day_close = sorted_days[i][1][2]
            if i == 0:
                day_open = sorted_days[i][1][1]
            else:
                day_open = sorted_days[i - 1][1][2]
            ret = float((day_close - day_open) / day_open) if day_open else 0.0
            result.append((day_ts, ret))
        return result

    @staticmethod
    def _annualize(total_return: Decimal, hours: float) -> Decimal:
        if hours <= 0:
            return Decimal(0)
        years = Decimal(str(hours / _ANNUAL_HOURS))
        if years <= 0:
            return Decimal(0)
        total_float = float(total_return)
        base = 1 + total_float
        if base <= 0:
            return Decimal("-1")
        import math
        try:
            cagr = math.exp(math.log(base) / float(years)) - 1
        except (OverflowError, ValueError):
            return Decimal(str(round(total_float, 6)))
        return Decimal(str(round(cagr, 6)))

    def _sharpe_ratio(self, returns: Sequence[float], risk_free: float = _RISK_FREE_ANNUAL) -> float:
        """Annualized Sharpe; annualization adapts to actual bar frequency."""
        if len(returns) < 2:
            return 0.0
        bars_per_year = self._bars_per_year()
        rf_per_bar = risk_free / bars_per_year if bars_per_year > 0 else 0.0
        excess = [r - rf_per_bar for r in returns]
        mean_excess = sum(excess) / len(excess)
        std = _std(returns)
        if std == 0:
            return 0.0
        return (mean_excess / std) * math.sqrt(bars_per_year)

    def _sortino_ratio(self, returns: Sequence[float], risk_free: float = _RISK_FREE_ANNUAL) -> float:
        """Annualized Sortino; annualization adapts to actual bar frequency."""
        n = len(returns)
        if n < 2:
            return 0.0
        bars_per_year = self._bars_per_year()
        rf_per_bar = risk_free / bars_per_year if bars_per_year > 0 else 0.0
        excess = [r - rf_per_bar for r in returns]
        mean_excess = sum(excess) / n
        downside_sq = sum(min(e, 0.0) ** 2 for e in excess)
        if downside_sq == 0:
            return float("inf") if mean_excess > 0 else 0.0
        down_dev = math.sqrt(downside_sq / (n - 1))
        if down_dev == 0:
            return 0.0
        return (mean_excess / down_dev) * math.sqrt(bars_per_year)

    def _volatility(self, returns: Sequence[float]) -> float:
        """Annualized volatility; adapts to actual bar frequency."""
        if len(returns) < 2:
            return 0.0
        return _std(returns) * math.sqrt(self._bars_per_year())

    def _bars_per_year(self) -> float:
        """Estimate bars-per-year from the equity curve timestamps."""
        if len(self._equity_curve) < 2:
            return 252.0
        dt_start = self._equity_curve[0][0]
        dt_end = self._equity_curve[-1][0]
        n_bars = len(self._equity_curve)
        duration_hours = max((dt_end - dt_start).total_seconds() / 3600, 1)
        bars_per_hour = n_bars / duration_hours
        return bars_per_hour * _ANNUAL_HOURS

    def _max_drawdown(self) -> tuple[Decimal, int]:
        if not self._equity_curve:
            return _ZERO, 0

        peak = self._equity_curve[0][1]
        max_dd = _ZERO
        self._equity_curve[0][0]
        max_dd_duration = timedelta(0)
        current_dd_start = self._equity_curve[0][0]

        for ts, eq in self._equity_curve:
            if eq > peak:
                peak = eq
                current_dd_start = ts
            dd = (peak - eq) / peak if peak > 0 else _ZERO
            if dd > max_dd:
                max_dd = dd
                max_dd_duration = ts - current_dd_start

        return max_dd, max(max_dd_duration.days, 0)

    def _trade_stats(self) -> dict:
        round_trips = self._build_round_trips()
        if not round_trips:
            return {
                "total": 0,
                "win_rate": 0.0,
                "profit_factor": _ZERO,
                "avg_pnl": _ZERO,
                "avg_win": _ZERO,
                "avg_loss": _ZERO,
                "avg_hold_hours": 0.0,
            }

        pnls = [rt["pnl"] for rt in round_trips]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        total_win = sum(wins) if wins else _ZERO
        total_loss = abs(sum(losses)) if losses else _ZERO

        hold_hours = [rt["hold_hours"] for rt in round_trips]

        return {
            "total": len(round_trips),
            "win_rate": len(wins) / len(round_trips) if round_trips else 0.0,
            "profit_factor": total_win / total_loss if total_loss > 0 else _ZERO,
            "avg_pnl": sum(pnls) / len(pnls),
            "avg_win": total_win / len(wins) if wins else _ZERO,
            "avg_loss": (sum(losses) / len(losses)) if losses else _ZERO,
            "avg_hold_hours": sum(hold_hours) / len(hold_hours) if hold_hours else 0.0,
        }

    def _build_round_trips(self) -> list[dict]:
        """Pair entry/exit fills into round-trip trades (FIFO, handles partial fills)."""
        open_positions: dict[str, list[tuple[Fill, Decimal]]] = {}
        trips: list[dict] = []

        for fill in sorted(self._fills, key=lambda f: f.timestamp):
            key = fill.symbol
            stack = open_positions.setdefault(key, [])
            remaining = fill.quantity

            if not stack or stack[0][0].side == fill.side:
                stack.append((fill, fill.quantity))
                continue

            while remaining > 0 and stack and stack[0][0].side != fill.side:
                entry_fill, entry_remaining = stack[0]
                qty = min(entry_remaining, remaining)

                if entry_fill.side == OrderSide.BUY:
                    pnl = (fill.price - entry_fill.price) * qty
                else:
                    pnl = (entry_fill.price - fill.price) * qty

                entry_comm_share = entry_fill.commission * qty / entry_fill.quantity if entry_fill.quantity else _ZERO
                exit_comm_share = fill.commission * qty / fill.quantity if fill.quantity else _ZERO
                pnl -= (entry_comm_share + exit_comm_share)
                hold = (fill.timestamp - entry_fill.timestamp).total_seconds() / 3600
                trips.append({"pnl": pnl, "hold_hours": hold})

                entry_remaining -= qty
                remaining -= qty
                if entry_remaining <= 0:
                    stack.pop(0)
                else:
                    stack[0] = (entry_fill, entry_remaining)

            if remaining > 0:
                stack.append((fill, remaining))

        return trips


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)
