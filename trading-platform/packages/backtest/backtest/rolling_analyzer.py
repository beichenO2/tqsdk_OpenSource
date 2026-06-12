"""Rolling-window metrics from an equity curve (daily aggregation)."""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

import numpy as np

from .models import BacktestResult, EquityCurvePoint
from .report import RISK_FREE_RATE, TRADING_DAYS_PER_YEAR


class RollingAnalyzer:
    """Compute N-day rolling Sharpe, max drawdown, return, and volatility.

    Intraday points are collapsed to the **last** equity observation per calendar day.
    Multiple ``window_days`` values produce columns ``sharpe_{N}``, ``max_drawdown_pct_{N}``, etc.
    """

    def __init__(self, window_days: int | list[int]) -> None:
        if isinstance(window_days, int):
            self.window_days = [window_days]
        else:
            self.window_days = sorted({int(w) for w in window_days if int(w) > 1})

    def analyze(self, result: BacktestResult) -> list[dict[str, Any]]:
        if not result.equity_curve or not self.window_days:
            return []

        daily_dates, daily_eq = _daily_last_equity(result.equity_curve)
        if len(daily_dates) < 2:
            return []

        eq = np.asarray(daily_eq, dtype=np.float64)
        rets = np.diff(eq) / np.where(eq[:-1] > 0, eq[:-1], 1.0)
        rf_daily = float(RISK_FREE_RATE) / TRADING_DAYS_PER_YEAR
        excess = rets - rf_daily

        max_w = max(self.window_days)
        out: list[dict[str, Any]] = []

        # Need at least max_w points in window → max_w-1 returns minimum for vol
        for end in range(max_w - 1, len(daily_dates)):
            row: dict[str, Any] = {"date": daily_dates[end]}
            for w in self.window_days:
                if end + 1 < w:
                    continue
                start = end - w + 1
                window_eq = eq[start : end + 1]
                window_rets = excess[start:end]  # length w-1, aligned with rets indices

                total_ret = float(window_eq[-1] / window_eq[0] - 1.0) if window_eq[0] > 0 else 0.0
                vol_ann = 0.0
                sharpe = 0.0
                if window_rets.size > 1:
                    std = float(np.std(window_rets, ddof=1))
                    if std > 0:
                        vol_ann = std * math.sqrt(TRADING_DAYS_PER_YEAR)
                        sharpe = float(
                            np.mean(window_rets) / std * math.sqrt(TRADING_DAYS_PER_YEAR)
                        )
                elif window_rets.size == 1 and float(window_rets[0]) != 0:
                    std = abs(float(window_rets[0]))
                    sharpe = float(window_rets[0]) / std * math.sqrt(TRADING_DAYS_PER_YEAR)

                mdd_pct = _max_drawdown_pct(window_eq)

                row[f"return_{w}"] = total_ret
                row[f"volatility_{w}"] = vol_ann
                row[f"sharpe_{w}"] = sharpe
                row[f"max_drawdown_pct_{w}"] = mdd_pct
            out.append(row)
        return out


def _daily_last_equity(curve: list[EquityCurvePoint]) -> tuple[list[date], list[float]]:
    by_day: dict[date, float] = {}
    for pt in sorted(curve, key=lambda p: p.dt):
        d = pt.dt.date() if isinstance(pt.dt, datetime) else pt.dt
        by_day[d] = float(pt.equity)
    dates = sorted(by_day.keys())
    return dates, [by_day[d] for d in dates]


def _max_drawdown_pct(series: np.ndarray) -> float:
    peak = np.maximum.accumulate(series)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, (peak - series) / peak, 0.0)
    return float(dd.max()) if dd.size else 0.0
