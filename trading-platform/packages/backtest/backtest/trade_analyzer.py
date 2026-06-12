"""Round-trip trade statistics: MAE/MFE bounds, streaks, time-of-day/week."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from .models import OrderSide, Trade


@dataclass(slots=True)
class RoundTripAnalysis:
    """Per closed position (open → close) analytics."""

    symbol: str
    entry_dt: datetime
    exit_dt: datetime
    side: str
    entry_price: float
    exit_price: float
    volume: int
    pnl: float
    commission: float
    holding_seconds: float
    mae_price: float
    mfe_price: float
    mae_pnl: float
    mfe_pnl: float
    entry_quality: float
    exit_quality: float


@dataclass(slots=True)
class TradeAnalysisSummary:
    """Aggregate outputs from :class:`TradeAnalyzer`."""

    round_trips: list[RoundTripAnalysis]
    win_streak_max: int
    loss_streak_max: int
    holding_period_seconds: dict[str, float]
    by_hour: dict[int, dict[str, float]]
    by_weekday: dict[int, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        rts: list[dict[str, Any]] = []
        for rt in self.round_trips:
            d = asdict(rt)
            d["entry_dt"] = rt.entry_dt.isoformat()
            d["exit_dt"] = rt.exit_dt.isoformat()
            rts.append(d)
        return {
            "round_trips": rts,
            "win_streak_max": self.win_streak_max,
            "loss_streak_max": self.loss_streak_max,
            "holding_period_seconds": self.holding_period_seconds,
            "by_hour": {str(k): v for k, v in self.by_hour.items()},
            "by_weekday": {str(k): v for k, v in self.by_weekday.items()},
        }


class TradeAnalyzer:
    """Analyze paired trades (same heuristic as :class:`ReportGenerator`)."""

    @staticmethod
    def pair_round_trips(trades: list[Trade]) -> list[dict[str, Any]]:
        """Pair opening and closing legs; same stack logic as ``ReportGenerator._pair_trades``."""
        open_trades: dict[str, list[Trade]] = {}
        pairs: list[dict[str, Any]] = []

        for t in sorted(trades, key=lambda x: x.dt):
            key = t.symbol
            if key not in open_trades:
                open_trades[key] = []
            stack = open_trades[key]
            if stack and stack[-1].side != t.side:
                opener = stack.pop(0)
                mult = 1
                if opener.side == OrderSide.BUY:
                    pnl = float((t.price - opener.price) * min(opener.volume, t.volume) * mult)
                else:
                    pnl = float((opener.price - t.price) * min(opener.volume, t.volume) * mult)
                pnl -= float(opener.commission + t.commission)
                holding = (t.dt - opener.dt).total_seconds()
                pairs.append(
                    {
                        "opener": opener,
                        "closer": t,
                        "pnl": pnl,
                        "holding_seconds": holding,
                    }
                )
            else:
                stack.append(t)
        return pairs

    def analyze(self, trades: list[Trade]) -> TradeAnalysisSummary:
        raw = self.pair_round_trips(trades)
        round_trips: list[RoundTripAnalysis] = []
        pnls: list[float] = []

        for p in raw:
            o = p["opener"]
            c = p["closer"]
            pnl = float(p["pnl"])
            pnls.append(pnl)
            ep = float(o.price)
            xp = float(c.price)
            vol = int(min(o.volume, c.volume))
            is_long = o.side == OrderSide.BUY
            if is_long:
                mfe_price = max(0.0, xp - ep)
                mae_price = max(0.0, ep - xp)
            else:
                mfe_price = max(0.0, ep - xp)
                mae_price = max(0.0, xp - ep)
            mfe_pnl = mfe_price * vol
            mae_pnl = mae_price * vol
            entry_q, exit_q = _quality_scores(pnl, mae_pnl, mfe_pnl)
            round_trips.append(
                RoundTripAnalysis(
                    symbol=o.symbol,
                    entry_dt=o.dt,
                    exit_dt=c.dt,
                    side=o.side.value,
                    entry_price=ep,
                    exit_price=xp,
                    volume=vol,
                    pnl=pnl,
                    commission=float(o.commission + c.commission),
                    holding_seconds=float(p["holding_seconds"]),
                    mae_price=mae_price,
                    mfe_price=mfe_price,
                    mae_pnl=mae_pnl,
                    mfe_pnl=mfe_pnl,
                    entry_quality=entry_q,
                    exit_quality=exit_q,
                )
            )

        win_streak_max, loss_streak_max = _streaks(pnls)
        hp = _holding_stats([rt.holding_seconds for rt in round_trips])
        by_hour, by_weekday = _time_buckets(raw)

        return TradeAnalysisSummary(
            round_trips=round_trips,
            win_streak_max=win_streak_max,
            loss_streak_max=loss_streak_max,
            holding_period_seconds=hp,
            by_hour=by_hour,
            by_weekday=by_weekday,
        )


def _quality_scores(pnl: float, mae_pnl: float, mfe_pnl: float) -> tuple[float, float]:
    """Heuristic 0–100 scores from open/close only (path not observed)."""
    if mfe_pnl > 0:
        exit_capture = max(0.0, min(1.0, pnl / mfe_pnl))
    else:
        exit_capture = 0.5 if pnl == 0 else (1.0 if pnl > 0 else 0.0)

    if mae_pnl > 0:
        base = mae_pnl + max(pnl, 0.0)
        entry_def = max(0.0, min(1.0, max(pnl, 0.0) / base))
    else:
        entry_def = 1.0

    return round(entry_def * 100.0, 2), round(exit_capture * 100.0, 2)


def _streaks(pnls: list[float]) -> tuple[int, int]:
    wm = lm = cw = cl = 0
    for x in pnls:
        if x > 0:
            cw += 1
            cl = 0
            wm = max(wm, cw)
        else:
            cl += 1
            cw = 0
            lm = max(lm, cl)
    return wm, lm


def _holding_stats(seconds: list[float]) -> dict[str, float]:
    if not seconds:
        return {"count": 0.0, "mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    arr = sorted(seconds)
    n = len(arr)
    mid = arr[n // 2] if n % 2 else 0.5 * (arr[n // 2 - 1] + arr[n // 2])
    p90_idx = min(n - 1, max(0, int(round(0.9 * (n - 1)))))
    return {
        "count": float(n),
        "mean": float(sum(arr) / n),
        "median": float(mid),
        "p90": float(arr[p90_idx]),
        "max": float(arr[-1]),
    }


def _time_buckets(
    pairs: list[dict[str, Any]],
) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, float]]]:
    by_hour: dict[int, list[float]] = defaultdict(list)
    by_wd: dict[int, list[float]] = defaultdict(list)
    for p in pairs:
        c: Trade = p["closer"]
        pnl = float(p["pnl"])
        by_hour[c.dt.hour].append(pnl)
        by_wd[c.dt.weekday()].append(pnl)

    def agg(buckets: dict[int, list[float]]) -> dict[int, dict[str, float]]:
        out: dict[int, dict[str, float]] = {}
        for k, vals in buckets.items():
            wins = sum(1 for v in vals if v > 0)
            out[k] = {
                "trades": float(len(vals)),
                "total_pnl": float(sum(vals)),
                "mean_pnl": float(sum(vals) / len(vals)) if vals else 0.0,
                "win_rate": float(wins / len(vals)) if vals else 0.0,
            }
        return out

    return agg(by_hour), agg(by_wd)
