"""Monte Carlo analysis by shuffling round-trip trade PnLs."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import numpy as np

from .models import BacktestResult
from .report import ReportGenerator


@dataclass(slots=True)
class MonteCarloResult:
    """Aggregated outputs from :class:`MonteCarloAnalyzer`."""

    n_simulations: int
    n_trades_sampled: int
    var_return_95: float
    var_return_99: float
    expected_max_drawdown_pct: float
    expected_max_drawdown_abs: float
    final_return_mean: float
    final_return_std: float
    final_return_ci_95: tuple[float, float]
    final_return_ci_99: tuple[float, float]
    final_equity_mean: float
    paths_max_drawdown_pct: np.ndarray
    paths_final_return: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_simulations": self.n_simulations,
            "n_trades_sampled": self.n_trades_sampled,
            "var_return_95": self.var_return_95,
            "var_return_99": self.var_return_99,
            "expected_max_drawdown_pct": self.expected_max_drawdown_pct,
            "expected_max_drawdown_abs": self.expected_max_drawdown_abs,
            "final_return_mean": self.final_return_mean,
            "final_return_std": self.final_return_std,
            "final_return_ci_95": list(self.final_return_ci_95),
            "final_return_ci_99": list(self.final_return_ci_99),
            "final_equity_mean": self.final_equity_mean,
        }


class MonteCarloAnalyzer:
    """Shuffle round-trip trade PnLs to estimate tail risk and path variability.

    Uses the same long/short pairing heuristic as :class:`ReportGenerator`.
    """

    def __init__(self, n_simulations: int = 1000, *, random_seed: int | None = None) -> None:
        self.n_simulations = int(n_simulations)
        self._rng = np.random.default_rng(random_seed)

    def analyze(self, result: BacktestResult) -> MonteCarloResult:
        pairs = ReportGenerator._pair_trades(result.trades)
        pnls = np.array([float(p["pnl"]) for p in pairs], dtype=np.float64)
        initial = float(result.config.initial_capital or Decimal(0))
        if initial <= 0:
            initial = float(result.equity_curve[0].equity) if result.equity_curve else 1.0

        if pnls.size == 0:
            zeros = np.zeros(max(self.n_simulations, 1), dtype=np.float64)
            return MonteCarloResult(
                n_simulations=self.n_simulations,
                n_trades_sampled=0,
                var_return_95=0.0,
                var_return_99=0.0,
                expected_max_drawdown_pct=0.0,
                expected_max_drawdown_abs=0.0,
                final_return_mean=0.0,
                final_return_std=0.0,
                final_return_ci_95=(0.0, 0.0),
                final_return_ci_99=(0.0, 0.0),
                final_equity_mean=initial,
                paths_max_drawdown_pct=zeros,
                paths_final_return=zeros,
            )

        n_sims = max(1, self.n_simulations)
        rand = self._rng.random((n_sims, pnls.size))
        order = np.argsort(rand, axis=1)
        shuffled = pnls[order]
        cumulative = np.cumsum(shuffled, axis=1)
        equity = initial + np.hstack([np.zeros((n_sims, 1)), cumulative])
        peaks = np.maximum.accumulate(equity, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            dd_pct = np.where(peaks > 0, (peaks - equity) / peaks, 0.0)
        max_dd_pct = dd_pct.max(axis=1)
        max_dd_abs = (peaks - equity).max(axis=1)
        final_eq = equity[:, -1]
        final_ret = (final_eq - initial) / initial

        var_95 = float(np.percentile(final_ret, 5))
        var_99 = float(np.percentile(final_ret, 1))
        ci95 = (
            float(np.percentile(final_ret, 2.5)),
            float(np.percentile(final_ret, 97.5)),
        )
        ci99 = (
            float(np.percentile(final_ret, 0.5)),
            float(np.percentile(final_ret, 99.5)),
        )

        return MonteCarloResult(
            n_simulations=n_sims,
            n_trades_sampled=int(pnls.size),
            var_return_95=var_95,
            var_return_99=var_99,
            expected_max_drawdown_pct=float(np.mean(max_dd_pct)),
            expected_max_drawdown_abs=float(np.mean(max_dd_abs)),
            final_return_mean=float(np.mean(final_ret)),
            final_return_std=float(np.std(final_ret, ddof=1)) if n_sims > 1 else 0.0,
            final_return_ci_95=ci95,
            final_return_ci_99=ci99,
            final_equity_mean=float(np.mean(final_eq)),
            paths_max_drawdown_pct=max_dd_pct,
            paths_final_return=final_ret,
        )
