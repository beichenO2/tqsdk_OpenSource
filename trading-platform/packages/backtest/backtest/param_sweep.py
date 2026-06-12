"""参数网格搜索统一入口，支持并行执行。"""

from __future__ import annotations

import concurrent.futures
import itertools
import logging
import time
from typing import TYPE_CHECKING, Any, Literal

from .datafeed import BarDataFeed
from .engine import BacktestEngine
from .strategy_adapter import StrategyAdapter

if TYPE_CHECKING:
    from .models import BacktestConfig, BacktestResult, Bar

logger = logging.getLogger(__name__)

MetricName = Literal["total_return", "sharpe_ratio", "profit_factor", "win_rate"]

_METRIC_ATTRS: dict[str, str] = {
    "total_return": "total_return",
    "sharpe_ratio": "sharpe_ratio",
    "profit_factor": "profit_factor",
    "win_rate": "win_rate",
}


def _bars_snapshot(data: BarDataFeed | list[Bar]) -> list[Bar]:
    if isinstance(data, list):
        return list(data)
    return list(data._bars)


def _backtest_metrics(result: BacktestResult) -> dict[str, float | int]:
    return {
        "final_equity": float(result.final_equity),
        "total_return": float(result.total_return),
        "annual_return": float(result.annual_return),
        "max_drawdown": float(result.max_drawdown),
        "max_drawdown_pct": float(result.max_drawdown_pct),
        "sharpe_ratio": float(result.sharpe_ratio),
        "sortino_ratio": float(result.sortino_ratio),
        "win_rate": float(result.win_rate),
        "profit_factor": float(result.profit_factor),
        "total_trades": result.total_trades,
        "avg_trade_pnl": float(result.avg_trade_pnl),
        "avg_holding_period": float(result.avg_holding_period),
        "calmar_ratio": float(result.calmar_ratio),
        "elapsed_seconds": float(result.elapsed_seconds),
    }


def _metric_value(result: BacktestResult, metric: str) -> float:
    attr = _METRIC_ATTRS[metric]
    return float(getattr(result, attr))


class ParameterSweep:
    """参数网格搜索：返回最优参数和完整回测结果。"""

    def __init__(
        self,
        config: BacktestConfig,
        strategy_name: str,
        fixed_params: dict[str, Any],
        param_grid: dict[str, list[Any]],
        data: BarDataFeed | list[Bar],
        *,
        metric: MetricName | str = "sharpe_ratio",
        default_volume: int = 1,
    ) -> None:
        if metric not in _METRIC_ATTRS:
            allowed = ", ".join(sorted(_METRIC_ATTRS))
            raise ValueError(f"未知优化指标 {metric!r}，可选: {allowed}")

        self._config = config
        self._strategy_name = strategy_name
        self._fixed_params = dict(fixed_params)
        self._param_grid = {k: list(v) for k, v in param_grid.items()}
        self._data = data
        self._metric = str(metric)
        self._default_volume = default_volume

    def run(
        self,
        *,
        max_workers: int | None = None,
        executor: concurrent.futures.Executor | None = None,
    ) -> dict[str, Any]:
        """兼容统一调用方式：run() 等价于 parameter_sweep()."""
        return self.parameter_sweep(max_workers=max_workers, executor=executor)

    def parameter_sweep(
        self,
        *,
        max_workers: int | None = None,
        executor: concurrent.futures.Executor | None = None,
    ) -> dict[str, Any]:
        """
        执行参数网格搜索。

        返回:
            - best_params: 最优参数组合
            - best_score: 最优评分（由 metric 决定）
            - all_results: 所有组合的完整结果（含 BacktestResult）
            - total_combinations: 总组合数
        """
        if max_workers is not None and max_workers < 1:
            raise ValueError("max_workers 必须 >= 1")
        if executor is not None and max_workers is not None:
            raise ValueError("executor 与 max_workers 不能同时指定")

        keys = list(self._param_grid.keys())
        value_lists = [self._param_grid[k] for k in keys]
        combinations = list(itertools.product(*value_lists)) if keys else [()]
        total_combinations = len(combinations)
        wall_start = time.monotonic()

        if total_combinations == 0:
            return {
                "best_params": {},
                "best_score": float("-inf"),
                "all_results": [],
                "total_combinations": 0,
                "metric_name": self._metric,
                "elapsed_seconds": time.monotonic() - wall_start,
            }

        bars_src = _bars_snapshot(self._data)
        if executor is None and max_workers == 1:
            rows = [self._run_single_combination(keys, combo, bars_src) for combo in combinations]
            rows.sort(key=lambda r: (-r["score"], str(sorted(r["params"].items()))))
            best = rows[0]
            return {
                "best_params": dict(best["params"]),
                "best_score": float(best["score"]),
                "all_results": rows,
                "total_combinations": total_combinations,
                "metric_name": self._metric,
                "elapsed_seconds": time.monotonic() - wall_start,
            }

        owned_executor = executor is None
        working_executor = executor or concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
        )

        try:
            futures = [
                working_executor.submit(self._run_single_combination, keys, combo, bars_src)
                for combo in combinations
            ]
            rows = [f.result() for f in concurrent.futures.as_completed(futures)]
        finally:
            if owned_executor:
                working_executor.shutdown(wait=True)

        rows.sort(key=lambda r: (-r["score"], str(sorted(r["params"].items()))))
        best = rows[0]
        elapsed = time.monotonic() - wall_start

        return {
            "best_params": dict(best["params"]),
            "best_score": float(best["score"]),
            "all_results": rows,
            "total_combinations": total_combinations,
            "metric_name": self._metric,
            "elapsed_seconds": elapsed,
        }

    def _run_single_combination(
        self,
        keys: list[str],
        combo: tuple[Any, ...],
        bars_src: list[Bar],
    ) -> dict[str, Any]:
        from strategy.base import StrategyConfig
        from strategy.registry import StrategyRegistry

        varied = dict(zip(keys, combo, strict=True))
        params = {**self._fixed_params, **varied}

        strategy_cfg = StrategyConfig(
            name=self._strategy_name,
            symbols=list(self._config.symbols),
            params=params,
        )
        base = StrategyRegistry.create(self._strategy_name, strategy_cfg)
        adapter = StrategyAdapter(base, default_volume=self._default_volume)

        engine = BacktestEngine(self._config)
        feed = BarDataFeed(engine.event_bus)
        feed.add_bars(bars_src)
        engine.set_datafeed(feed)
        engine.set_strategy(adapter)

        result = engine.run()
        score = _metric_value(result, self._metric)
        metrics = _backtest_metrics(result)
        row: dict[str, Any] = {
            "params": dict(params),
            "score": float(score),
            "metrics": metrics,
            "backtest_result": result,
        }
        logger.debug("ParameterSweep: params=%s %s=%.6f", params, self._metric, score)
        return row


def parameter_sweep(
    config: BacktestConfig,
    strategy_name: str,
    fixed_params: dict[str, Any],
    param_grid: dict[str, list[Any]],
    data: BarDataFeed | list[Bar],
    *,
    metric: MetricName | str = "sharpe_ratio",
    default_volume: int = 1,
    max_workers: int | None = None,
    executor: concurrent.futures.Executor | None = None,
) -> dict[str, Any]:
    """便捷函数：直接执行参数搜索并返回结果。"""
    sweep = ParameterSweep(
        config=config,
        strategy_name=strategy_name,
        fixed_params=fixed_params,
        param_grid=param_grid,
        data=data,
        metric=metric,
        default_volume=default_volume,
    )
    return sweep.parameter_sweep(max_workers=max_workers, executor=executor)
