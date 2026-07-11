"""AlphaGate — five-gate acceptance framework for candidate alpha signals.

All PnL / trade metrics are vectorized (pandas/numpy). Positions must already
be causal (no look-ahead); returns use ``position.shift(1)`` at execution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_BARS_PER_YEAR_4H = 6 * 365  # 2190


@dataclass
class GateResult:
    name: str
    passed: bool
    value: float | None
    threshold: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class GateReport:
    gates: dict[str, GateResult]
    metrics: dict[str, float]
    verdict: str
    one_way_cost_bps: float

    def to_markdown(self) -> str:
        lines = [
            "## AlphaGate Report",
            "",
            f"- **Verdict:** {self.verdict}",
            f"- **One-way cost:** {self.one_way_cost_bps:.1f} bp",
            "",
            "| Gate | Pass | Value | Threshold |",
            "|------|------|-------|-----------|",
        ]
        for key in ("G1_walk_forward", "G2_cost_sensitivity", "G3_trade_expectancy",
                    "G4_turnover_cost", "G5_benchmark"):
            g = self.gates[key]
            mark = "✓" if g.passed else "✗"
            val = "—" if g.value is None or (isinstance(g.value, float) and np.isnan(g.value)) else f"{g.value:.4f}"
            lines.append(f"| {g.name} | {mark} | {val} | {g.threshold} |")

        lines.extend([
            "",
            "### Summary metrics",
            "",
            f"- Total net return: {self.metrics.get('total_net_return', 0):.4f}",
            f"- Net Sharpe (ann.): {self.metrics.get('net_sharpe', 0):.4f}",
            f"- Annual turnover: {self.metrics.get('annual_turnover', 0):.2f}",
            f"- Cost ratio: {self.metrics.get('cost_ratio', 0):.4f}",
            f"- Per-trade expectancy (bp): {self.metrics.get('trade_expectancy_bp', 0):.2f}",
        ])

        g5 = self.gates.get("G5_benchmark")
        if g5 is not None and g5.details:
            d = g5.details
            lines.extend([
                "",
                "### G5 benchmark details",
                "",
                "| Metric | Strategy | Buy&Hold | Supertrend |",
                "|--------|----------|----------|------------|",
                f"| Total return | {d.get('strategy_return', 0):.4f} | "
                f"{d.get('buy_hold_return', 0):.4f} | {d.get('supertrend_return', 0):.4f} |",
                f"| Net Sharpe | {d.get('strategy_sharpe', 0):.4f} | "
                f"{d.get('buy_hold_sharpe', 0):.4f} | {d.get('supertrend_sharpe', 0):.4f} |",
                "",
                f"- **G5a** (risk-adjusted): "
                f"{'✓' if d.get('g5a_passed') else '✗'} — "
                f"strategy Sharpe > B&H and > supertrend",
                f"- **G5b** (walk-forward vs B&H): "
                f"{'✓' if d.get('g5b_passed') else '✗'} — "
                f"{d.get('segments_beat_bh', 0)}/4 segments beat buy&hold",
            ])
            seg_detail = d.get("segment_bh_comparison", [])
            if seg_detail:
                lines.append("- Segment wins vs B&H:")
                for i, item in enumerate(seg_detail, start=1):
                    mark = "✓" if item.get("beat_bh") else "✗"
                    lines.append(
                        f"  - Seg {i}: {mark} "
                        f"(strat {item.get('strategy_return', 0):.4f} vs "
                        f"B&H {item.get('buy_hold_return', 0):.4f})"
                    )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "one_way_cost_bps": self.one_way_cost_bps,
            "gates": {k: asdict(v) for k, v in self.gates.items()},
            "metrics": self.metrics,
        }


class AlphaGate:
    """Evaluate a causal position series against five acceptance gates."""

    def __init__(
        self,
        one_way_cost_bps: float = 5.0,
        bars_per_year: float = DEFAULT_BARS_PER_YEAR_4H,
        supertrend_period: int = 10,
        supertrend_multiplier: float = 3.0,
    ) -> None:
        self.one_way_cost_bps = one_way_cost_bps
        self.bars_per_year = bars_per_year
        self.supertrend_period = supertrend_period
        self.supertrend_multiplier = supertrend_multiplier

    def evaluate(self, position: pd.Series, ohlcv: pd.DataFrame) -> GateReport:
        pos, close, high, low = self._validate_inputs(position, ohlcv)
        one_way = self.one_way_cost_bps / 10_000.0

        ret_net, metrics_base = self._compute_returns(pos, close, one_way)
        gates: dict[str, GateResult] = {}

        g1 = self._gate_walk_forward(ret_net)
        gates["G1_walk_forward"] = g1

        g2, cost_details = self._gate_cost_sensitivity(pos, close)
        gates["G2_cost_sensitivity"] = g2

        g3 = self._gate_trade_expectancy(pos, ret_net, one_way)
        gates["G3_trade_expectancy"] = g3

        g4 = self._gate_turnover_cost(pos, one_way)
        gates["G4_turnover_cost"] = g4

        g5 = self._gate_benchmark(pos, close, one_way, high, low)
        gates["G5_benchmark"] = g5

        n_pass = sum(g.passed for g in gates.values())
        if n_pass == 5:
            verdict = "PASS"
        elif n_pass == 4:
            verdict = "MARGINAL"
        else:
            verdict = "REJECT"

        metrics = {
            **metrics_base,
            **cost_details,
            "annual_turnover": g4.details.get("annual_turnover", float("nan")),
            "cost_ratio": g4.details.get("cost_ratio", float("nan")),
            "trade_expectancy_bp": g3.value if g3.value is not None else float("nan"),
            "buy_hold_return": g5.details.get("buy_hold_return", float("nan")),
            "supertrend_return": g5.details.get("supertrend_return", float("nan")),
            "buy_hold_sharpe": g5.details.get("buy_hold_sharpe", float("nan")),
            "supertrend_sharpe": g5.details.get("supertrend_sharpe", float("nan")),
            "g5_segments_beat_bh": g5.details.get("segments_beat_bh", float("nan")),
        }

        return GateReport(
            gates=gates,
            metrics=metrics,
            verdict=verdict,
            one_way_cost_bps=self.one_way_cost_bps,
        )

    def _validate_inputs(
        self, position: pd.Series, ohlcv: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        if "close" not in ohlcv.columns:
            raise ValueError("ohlcv must contain 'close' column")
        if len(position) != len(ohlcv):
            raise ValueError(
                f"position length {len(position)} != ohlcv length {len(ohlcv)}"
            )
        pos = pd.Series(position.values, index=ohlcv.index, dtype=float).fillna(0.0)
        if (pos.abs() > 1.0 + 1e-9).any():
            raise ValueError("position values must be in [-1, 1]")
        close = ohlcv["close"].astype(float)
        high = ohlcv["high"].astype(float) if "high" in ohlcv.columns else close
        low = ohlcv["low"].astype(float) if "low" in ohlcv.columns else close
        return pos, close, high, low

    def _compute_returns(
        self, position: pd.Series, close: pd.Series, one_way_cost: float
    ) -> tuple[pd.Series, dict[str, float]]:
        price_ret = close.pct_change().fillna(0.0)
        pos_lag = position.shift(1).fillna(0.0)
        turnover = position.diff().abs().fillna(0.0)
        ret_gross = pos_lag * price_ret
        ret_cost = turnover * one_way_cost
        ret_net = ret_gross - ret_cost

        total_net = float((1.0 + ret_net).prod() - 1.0)
        sharpe = self._sharpe(ret_net)
        total_cost = float(ret_cost.sum())

        return ret_net, {
            "total_net_return": total_net,
            "net_sharpe": sharpe,
            "total_cost": total_cost,
        }

    def _sharpe(self, ret_net: pd.Series) -> float:
        r = ret_net.dropna()
        if len(r) < 2 or r.std(ddof=0) < 1e-12:
            return 0.0
        return float(r.mean() / r.std(ddof=0) * np.sqrt(self.bars_per_year))

    def _walk_forward_slices(self, n: int) -> list[tuple[int, int]]:
        seg_len = n // 4
        slices: list[tuple[int, int]] = []
        for i in range(4):
            start = i * seg_len
            end = (i + 1) * seg_len if i < 3 else n
            slices.append((start, end))
        return slices

    def _segment_total_return(self, ret: pd.Series, start: int, end: int) -> float:
        seg = ret.iloc[start:end]
        if seg.empty:
            return 0.0
        return float((1.0 + seg).prod() - 1.0)

    def _gate_walk_forward(self, ret_net: pd.Series) -> GateResult:
        n = len(ret_net)
        seg_len = n // 4
        if seg_len < 10:
            return GateResult(
                name="G1 Walk-forward",
                passed=False,
                value=0.0,
                threshold="≥3/4 segments net return > 0",
                details={"segments_positive": 0},
            )
        seg_rets: list[float] = []
        seg_sharpes: list[float] = []
        for start, end in self._walk_forward_slices(n):
            seg = ret_net.iloc[start:end]
            seg_rets.append(self._segment_total_return(ret_net, start, end))
            seg_sharpes.append(self._sharpe(seg))
        n_pos = sum(r > 0 for r in seg_rets)
        return GateResult(
            name="G1 Walk-forward",
            passed=n_pos >= 3,
            value=float(n_pos),
            threshold="≥3/4 segments net return > 0",
            details={
                "segments_positive": n_pos,
                "segment_returns": seg_rets,
                "segment_sharpes": seg_sharpes,
            },
        )

    def _gate_cost_sensitivity(
        self, position: pd.Series, close: pd.Series
    ) -> tuple[GateResult, dict[str, float]]:
        details: dict[str, float] = {}
        for bps in (2.0, 5.0, 10.0):
            _, m = self._compute_returns(position, close, bps / 10_000.0)
            details[f"net_return_{int(bps)}bp"] = m["total_net_return"]
        passed = details["net_return_10bp"] > 0
        return GateResult(
            name="G2 Cost sensitivity",
            passed=passed,
            value=details["net_return_10bp"],
            threshold="10bp one-way total net return > 0",
            details=details,
        ), details

    def _gate_trade_expectancy(
        self, position: pd.Series, ret_net: pd.Series, one_way_cost: float
    ) -> GateResult:
        changed = position.diff().abs().fillna(0.0) > 1e-9
        trade_id = changed.cumsum()
        # Exclude pre-first-trade idle bars from averaging
        active = trade_id > 0
        if not active.any():
            return GateResult(
                name="G3 Trade expectancy",
                passed=False,
                value=0.0,
                threshold=f"≥ {2 * 2 * self.one_way_cost_bps:.0f}bp round-trip",
            )
        trade_pnl = ret_net.groupby(trade_id).sum()
        trade_pnl = trade_pnl[trade_pnl.index > 0]
        if trade_pnl.empty:
            return GateResult(
                name="G3 Trade expectancy",
                passed=False,
                value=0.0,
                threshold=f"≥ {2 * 2 * self.one_way_cost_bps:.0f}bp round-trip",
            )
        expectancy_bp = float(trade_pnl.mean() * 10_000)
        round_trip_bp = 2.0 * 2.0 * self.one_way_cost_bps
        return GateResult(
            name="G3 Trade expectancy",
            passed=expectancy_bp >= round_trip_bp,
            value=expectancy_bp,
            threshold=f"≥ {round_trip_bp:.0f}bp per trade (2× round-trip)",
            details={"n_trades": int(len(trade_pnl)), "round_trip_bp": round_trip_bp},
        )

    def _gate_turnover_cost(
        self, position: pd.Series, one_way_cost: float
    ) -> GateResult:
        n = len(position)
        total_turnover = float(position.diff().abs().fillna(0.0).sum())
        annual_turnover = total_turnover * (self.bars_per_year / max(n, 1))
        total_cost = total_turnover * one_way_cost
        cost_ratio = total_cost  # initial notional = 1
        passed = annual_turnover <= 100.0 and cost_ratio <= 0.02
        return GateResult(
            name="G4 Turnover & cost",
            passed=passed,
            value=annual_turnover,
            threshold="turnover ≤ 100, cost ratio ≤ 2%",
            details={
                "annual_turnover": annual_turnover,
                "cost_ratio": cost_ratio,
                "turnover_pass": annual_turnover <= 100.0,
                "cost_pass": cost_ratio <= 0.02,
            },
        )

    def _gate_benchmark(
        self,
        position: pd.Series,
        close: pd.Series,
        one_way_cost: float,
        high: pd.Series,
        low: pd.Series,
    ) -> GateResult:
        strat_ret, strat_m = self._compute_returns(position, close, one_way_cost)
        bh_ret = close.pct_change().fillna(0.0)
        bh_r = float((1.0 + bh_ret).prod() - 1.0)
        bh_sharpe = self._sharpe(bh_ret)
        st_pos = supertrend_position(
            high, low, close,
            period=self.supertrend_period,
            multiplier=self.supertrend_multiplier,
        )
        _, st_m = self._compute_returns(st_pos, close, one_way_cost)
        strat_r = strat_m["total_net_return"]
        strat_sharpe = strat_m["net_sharpe"]
        st_r = st_m["total_net_return"]
        st_sharpe = st_m["net_sharpe"]

        g5a_passed = strat_sharpe > bh_sharpe and strat_sharpe > st_sharpe

        n = len(strat_ret)
        seg_len = n // 4
        segment_bh_comparison: list[dict[str, float | bool]] = []
        segments_beat_bh = 0
        if seg_len >= 10:
            for start, end in self._walk_forward_slices(n):
                strat_seg_r = self._segment_total_return(strat_ret, start, end)
                bh_seg_r = self._segment_total_return(bh_ret, start, end)
                beat_bh = strat_seg_r > bh_seg_r
                if beat_bh:
                    segments_beat_bh += 1
                segment_bh_comparison.append({
                    "strategy_return": strat_seg_r,
                    "buy_hold_return": bh_seg_r,
                    "beat_bh": beat_bh,
                })
        g5b_passed = segments_beat_bh >= 2
        passed = g5a_passed and g5b_passed

        return GateResult(
            name="G5 Benchmark",
            passed=passed,
            value=strat_sharpe,
            threshold="G5a: Sharpe > B&H & ST; G5b: ≥2/4 WF segments beat B&H",
            details={
                "strategy_return": strat_r,
                "strategy_sharpe": strat_sharpe,
                "buy_hold_return": bh_r,
                "buy_hold_sharpe": bh_sharpe,
                "supertrend_return": st_r,
                "supertrend_sharpe": st_sharpe,
                "g5a_passed": g5a_passed,
                "g5b_passed": g5b_passed,
                "segments_beat_bh": segments_beat_bh,
                "segment_bh_comparison": segment_bh_comparison,
            },
        )


def _wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def supertrend_position(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0,
) -> pd.Series:
    """Lightweight vectorized Supertrend → position in {-1, 0, 1}.

    Band recursion is O(n) on numpy arrays (indicator state machine, not bar-PnL loop).
    """
    h = high.to_numpy(dtype=float)
    lo = low.to_numpy(dtype=float)
    c = close.to_numpy(dtype=float)
    n = len(c)
    atr = _wilder_atr(high, low, close, period).to_numpy(dtype=float)
    hl2 = (h + lo) / 2.0
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    trend = np.ones(n, dtype=int)

    for i in range(1, n):
        if np.isnan(atr[i]):
            final_upper[i] = basic_upper[i]
            final_lower[i] = basic_lower[i]
            trend[i] = trend[i - 1]
            continue
        fu_prev = final_upper[i - 1] if not np.isnan(final_upper[i - 1]) else basic_upper[i]
        fl_prev = final_lower[i - 1] if not np.isnan(final_lower[i - 1]) else basic_lower[i]
        if c[i - 1] <= fu_prev:
            final_upper[i] = min(basic_upper[i], fu_prev)
        else:
            final_upper[i] = basic_upper[i]
        if c[i - 1] >= fl_prev:
            final_lower[i] = max(basic_lower[i], fl_prev)
        else:
            final_lower[i] = basic_lower[i]
        if trend[i - 1] == 1:
            trend[i] = 1 if c[i] >= final_lower[i] else -1
        else:
            trend[i] = -1 if c[i] <= final_upper[i] else 1

    pos = pd.Series(trend.astype(float), index=close.index)
    pos = pos.where(~np.isnan(atr), 0.0).clip(-1, 1)
    return pos


def causal_rolling_zscore(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Rolling z-score shifted by 1 bar (no look-ahead)."""
    mp = min_periods if min_periods is not None else max(window // 2, 2)
    mu = series.rolling(window, min_periods=mp).mean().shift(1)
    sig = series.rolling(window, min_periods=mp).std(ddof=0).shift(1)
    return (series - mu) / sig.replace(0.0, np.nan)
