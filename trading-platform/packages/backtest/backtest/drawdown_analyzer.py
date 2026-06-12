"""Drawdown segmentation, underwater curve, and recovery statistics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from .models import BacktestResult


@dataclass(slots=True)
class DrawdownEvent:
    """A single peak → trough → (optional) recovery episode."""

    peak_dt: datetime
    peak_equity: float
    trough_dt: datetime
    trough_equity: float
    recovery_dt: datetime | None
    drawdown_abs: float
    drawdown_pct: float
    days_peak_to_trough: float
    days_trough_to_recovery: float | None
    days_peak_to_recovery: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "peak_dt": self.peak_dt.isoformat(),
            "peak_equity": self.peak_equity,
            "trough_dt": self.trough_dt.isoformat(),
            "trough_equity": self.trough_equity,
            "recovery_dt": self.recovery_dt.isoformat() if self.recovery_dt else None,
            "drawdown_abs": self.drawdown_abs,
            "drawdown_pct": self.drawdown_pct,
            "days_peak_to_trough": self.days_peak_to_trough,
            "days_trough_to_recovery": self.days_trough_to_recovery,
            "days_peak_to_recovery": self.days_peak_to_recovery,
        }


@dataclass(slots=True)
class DrawdownAnalysis:
    """Full output of :class:`DrawdownAnalyzer`."""

    events: list[DrawdownEvent]
    underwater_curve: list[dict[str, Any]]
    recovery_days: dict[str, float]
    drawdown_distribution: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": [e.to_dict() for e in self.events],
            "underwater_curve": self.underwater_curve,
            "recovery_days": self.recovery_days,
            "drawdown_distribution": self.drawdown_distribution,
        }


class DrawdownAnalyzer:
    """Detailed drawdown analysis from an equity curve."""

    def __init__(self, top_n: int = 10) -> None:
        self.top_n = max(1, int(top_n))

    def analyze(self, result: BacktestResult) -> DrawdownAnalysis:
        curve = result.equity_curve
        if not curve:
            return DrawdownAnalysis(
                events=[],
                underwater_curve=[],
                recovery_days=_empty_recovery(),
                drawdown_distribution=_empty_dist(),
            )

        dts = [p.dt for p in curve]
        eq = np.array([float(p.equity) for p in curve], dtype=np.float64)

        underwater = _underwater_series(dts, eq)
        events = _find_drawdown_events(dts, eq)
        events_sorted = sorted(events, key=lambda e: e.drawdown_pct, reverse=True)[: self.top_n]

        rec_days = [
            e.days_trough_to_recovery
            for e in events
            if e.days_trough_to_recovery is not None
        ]
        recovery_stats = {
            "count_recovered": float(len(rec_days)),
            "mean_days_trough_to_recovery": float(np.mean(rec_days)) if rec_days else 0.0,
            "median_days_trough_to_recovery": float(np.median(rec_days)) if rec_days else 0.0,
            "p90_days_trough_to_recovery": float(np.percentile(rec_days, 90)) if rec_days else 0.0,
        }

        dd_pcts = np.array([e.drawdown_pct for e in events], dtype=np.float64)
        dist = _histogram(dd_pcts)

        return DrawdownAnalysis(
            events=events_sorted,
            underwater_curve=underwater,
            recovery_days=recovery_stats,
            drawdown_distribution=dist,
        )


def _empty_recovery() -> dict[str, float]:
    return {
        "count_recovered": 0.0,
        "mean_days_trough_to_recovery": 0.0,
        "median_days_trough_to_recovery": 0.0,
        "p90_days_trough_to_recovery": 0.0,
    }


def _empty_dist() -> dict[str, Any]:
    return {"bins": [], "counts": [], "n": 0}


def _underwater_series(dts: list[datetime], eq: np.ndarray) -> list[dict[str, Any]]:
    peak = np.maximum.accumulate(eq)
    with np.errstate(divide="ignore", invalid="ignore"):
        uw = np.where(peak > 0, (peak - eq) / peak, 0.0)
    return [
        {
            "dt": dts[i].isoformat(),
            "equity": float(eq[i]),
            "peak_equity": float(peak[i]),
            "underwater_pct": float(uw[i]),
        }
        for i in range(len(dts))
    ]


def _find_drawdown_events(dts: list[datetime], eq: np.ndarray) -> list[DrawdownEvent]:
    events: list[DrawdownEvent] = []
    running_peak = float(eq[0])
    peak_dt = dts[0]
    in_dd = False
    dd_peak_equity = running_peak
    dd_peak_dt = peak_dt
    trough_idx = 0

    def days_between(a: datetime, b: datetime) -> float:
        return max(0.0, (b - a).total_seconds() / 86400.0)

    for i in range(1, len(eq)):
        e = float(eq[i])
        if not in_dd:
            if e >= running_peak:
                running_peak = e
                peak_dt = dts[i]
            else:
                in_dd = True
                dd_peak_equity = running_peak
                dd_peak_dt = peak_dt
                trough_idx = i
        else:
            if e < float(eq[trough_idx]):
                trough_idx = i
            if e >= dd_peak_equity:
                tr = float(eq[trough_idx])
                dd_abs = dd_peak_equity - tr
                dd_pct = dd_abs / dd_peak_equity if dd_peak_equity > 0 else 0.0
                events.append(
                    DrawdownEvent(
                        peak_dt=dd_peak_dt,
                        peak_equity=dd_peak_equity,
                        trough_dt=dts[trough_idx],
                        trough_equity=tr,
                        recovery_dt=dts[i],
                        drawdown_abs=dd_abs,
                        drawdown_pct=dd_pct,
                        days_peak_to_trough=days_between(dd_peak_dt, dts[trough_idx]),
                        days_trough_to_recovery=days_between(dts[trough_idx], dts[i]),
                        days_peak_to_recovery=days_between(dd_peak_dt, dts[i]),
                    )
                )
                in_dd = False
                running_peak = e
                peak_dt = dts[i]

    if in_dd:
        tr = float(eq[trough_idx])
        dd_abs = dd_peak_equity - tr
        dd_pct = dd_abs / dd_peak_equity if dd_peak_equity > 0 else 0.0
        events.append(
            DrawdownEvent(
                peak_dt=dd_peak_dt,
                peak_equity=dd_peak_equity,
                trough_dt=dts[trough_idx],
                trough_equity=tr,
                recovery_dt=None,
                drawdown_abs=dd_abs,
                drawdown_pct=dd_pct,
                days_peak_to_trough=days_between(dd_peak_dt, dts[trough_idx]),
                days_trough_to_recovery=None,
                days_peak_to_recovery=None,
            )
        )
    return events


def _histogram(values: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    if values.size == 0:
        return _empty_dist()
    counts, bin_edges = np.histogram(values, bins=n_bins)
    return {
        "bins": [float(x) for x in bin_edges],
        "counts": [int(x) for x in counts],
        "n": int(values.size),
    }
