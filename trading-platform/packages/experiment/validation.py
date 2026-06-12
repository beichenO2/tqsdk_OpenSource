"""方法验证管线 — 先验证再使用。

策略/模型的超参优化完成后，必须通过此管线验证才能部署。
验证包括：OOS 测试、Walk-Forward 分析、Monte Carlo 模拟、
多品种一致性检查。返回 PASS/FAIL 判定及详细报告。

用法:
    validator = MethodValidator(min_sharpe=0.3, min_oos_return=0.0)
    report = validator.validate(strategy_fn, best_params, data)
    if report.passed:
        # 可以部署
    else:
        print(report.failures)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationCheck:
    name: str
    passed: bool
    metric_value: float
    threshold: float
    detail: str = ""


@dataclass
class ValidationReport:
    strategy_id: str
    params: dict[str, Any]
    checks: list[ValidationCheck] = field(default_factory=list)
    passed: bool = False
    summary: str = ""

    @property
    def failures(self) -> list[ValidationCheck]:
        return [c for c in self.checks if not c.passed]

    @property
    def pass_rate(self) -> float:
        if not self.checks:
            return 0.0
        return sum(1 for c in self.checks if c.passed) / len(self.checks)


@dataclass
class MethodValidator:
    """验证管线 — 策略/模型上线前的 gate check。

    Parameters
    ----------
    min_sharpe : float
        OOS 最低 Sharpe ratio。
    min_oos_return : float
        OOS 最低累计收益（0.0 = 不亏即可）。
    max_drawdown : float
        OOS 最大回撤上限（绝对值，例如 0.3 = 30%）。
    min_trades : int
        OOS 最少交易次数（太少则统计无意义）。
    min_profitable_assets : float
        多品种测试时至少多大比例的品种盈利（0.67 = 2/3）。
    wf_min_profitable_windows : float
        Walk-Forward 至少多大比例的窗口盈利。
    mc_confidence_level : float
        Monte Carlo 置信区间水平（默认 0.95）。
    mc_min_median_return : float
        MC 中位数收益下限。
    """

    min_sharpe: float = 0.3
    min_oos_return: float = 0.0
    max_drawdown: float = 0.30
    min_trades: int = 30
    min_profitable_assets: float = 0.67
    wf_min_profitable_windows: float = 0.5
    mc_confidence_level: float = 0.95
    mc_min_median_return: float = 0.0

    def validate(
        self,
        backtest_fn: Callable[[dict[str, Any], Any], dict[str, float]],
        params: dict[str, Any],
        oos_datasets: dict[str, Any],
        strategy_id: str = "unknown",
        wf_results: Optional[list[dict[str, float]]] = None,
        mc_returns: Optional[np.ndarray] = None,
    ) -> ValidationReport:
        """运行全套验证。

        Parameters
        ----------
        backtest_fn :
            接收 (params, dataset) 返回 {"sharpe", "total_return", "max_dd", "trades", ...}
        params :
            待验证的最优参数。
        oos_datasets :
            {symbol_name: dataset} — 每个品种的 OOS 数据。
        strategy_id :
            策略标识符。
        wf_results :
            Walk-Forward 各窗口结果列表（可选，有则验证）。
        mc_returns :
            Monte Carlo 模拟收益数组（可选，有则验证）。
        """
        report = ValidationReport(strategy_id=strategy_id, params=params)

        oos_results = {}
        for symbol, dataset in oos_datasets.items():
            try:
                result = backtest_fn(params, dataset)
                oos_results[symbol] = result
            except Exception as exc:
                logger.warning("OOS backtest failed for %s: %s", symbol, exc)
                oos_results[symbol] = {
                    "sharpe": -999, "total_return": -1.0,
                    "max_dd": 1.0, "trades": 0,
                }

        self._check_oos(report, oos_results)
        self._check_multi_asset(report, oos_results)

        if wf_results:
            self._check_walk_forward(report, wf_results)

        if mc_returns is not None and len(mc_returns) > 0:
            self._check_monte_carlo(report, mc_returns)

        n_passed = sum(1 for c in report.checks if c.passed)
        n_total = len(report.checks)
        report.passed = n_total > 0 and all(c.passed for c in report.checks)
        report.summary = (
            f"{'PASS' if report.passed else 'FAIL'}: "
            f"{n_passed}/{n_total} checks passed"
        )
        logger.info("Validation %s: %s", strategy_id, report.summary)
        return report

    def _check_oos(
        self, report: ValidationReport, results: dict[str, dict[str, float]]
    ) -> None:
        sharpes = [r.get("sharpe", -999) for r in results.values()]
        returns = [r.get("total_return", -1.0) for r in results.values()]
        drawdowns = [r.get("max_dd", 1.0) for r in results.values()]
        trades = [r.get("trades", 0) for r in results.values()]

        avg_sharpe = float(np.mean(sharpes)) if sharpes else -999
        report.checks.append(ValidationCheck(
            name="OOS Sharpe",
            passed=avg_sharpe >= self.min_sharpe,
            metric_value=avg_sharpe,
            threshold=self.min_sharpe,
            detail=f"Per-asset: {dict(zip(results.keys(), sharpes))}",
        ))

        avg_return = float(np.mean(returns)) if returns else -1.0
        report.checks.append(ValidationCheck(
            name="OOS Return",
            passed=avg_return >= self.min_oos_return,
            metric_value=avg_return,
            threshold=self.min_oos_return,
            detail=f"Per-asset: {dict(zip(results.keys(), returns))}",
        ))

        max_dd = float(np.max(drawdowns)) if drawdowns else 1.0
        report.checks.append(ValidationCheck(
            name="OOS MaxDD",
            passed=max_dd <= self.max_drawdown,
            metric_value=max_dd,
            threshold=self.max_drawdown,
            detail=f"Worst drawdown across assets",
        ))

        min_trade_count = int(np.min(trades)) if trades else 0
        report.checks.append(ValidationCheck(
            name="OOS MinTrades",
            passed=min_trade_count >= self.min_trades,
            metric_value=float(min_trade_count),
            threshold=float(self.min_trades),
            detail=f"Per-asset: {dict(zip(results.keys(), trades))}",
        ))

    def _check_multi_asset(
        self, report: ValidationReport, results: dict[str, dict[str, float]]
    ) -> None:
        if len(results) < 2:
            return

        profitable = sum(
            1 for r in results.values() if r.get("total_return", -1) > 0
        )
        ratio = profitable / len(results)
        report.checks.append(ValidationCheck(
            name="MultiAsset Consistency",
            passed=ratio >= self.min_profitable_assets,
            metric_value=ratio,
            threshold=self.min_profitable_assets,
            detail=f"{profitable}/{len(results)} assets profitable",
        ))

    def _check_walk_forward(
        self, report: ValidationReport, wf_results: list[dict[str, float]]
    ) -> None:
        if not wf_results:
            return
        profitable_windows = sum(
            1 for w in wf_results if w.get("total_return", -1) > 0
        )
        ratio = profitable_windows / len(wf_results)
        report.checks.append(ValidationCheck(
            name="WalkForward Stability",
            passed=ratio >= self.wf_min_profitable_windows,
            metric_value=ratio,
            threshold=self.wf_min_profitable_windows,
            detail=f"{profitable_windows}/{len(wf_results)} windows profitable",
        ))

    def _check_monte_carlo(
        self, report: ValidationReport, mc_returns: np.ndarray
    ) -> None:
        alpha = 1.0 - self.mc_confidence_level
        lower_bound = float(np.percentile(mc_returns, alpha * 100))
        median = float(np.median(mc_returns))

        report.checks.append(ValidationCheck(
            name="MonteCarlo CI Lower",
            passed=lower_bound > -self.max_drawdown,
            metric_value=lower_bound,
            threshold=-self.max_drawdown,
            detail=f"{self.mc_confidence_level*100:.0f}% CI lower bound",
        ))
        report.checks.append(ValidationCheck(
            name="MonteCarlo Median",
            passed=median >= self.mc_min_median_return,
            metric_value=median,
            threshold=self.mc_min_median_return,
            detail=f"Median of {len(mc_returns)} simulations",
        ))
