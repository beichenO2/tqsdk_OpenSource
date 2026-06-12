"""策略参数网格优化器 — 遍历参数空间寻找最优策略配置。"""

from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

from strategy.base import StrategyConfig
from strategy.registry import StrategyRegistry

from .datafeed import BarDataFeed
from .engine import BacktestEngine
from .models import BacktestConfig, BacktestResult, Bar
from .strategy_adapter import StrategyAdapter

logger = logging.getLogger(__name__)

MetricName = Literal["total_return", "sharpe_ratio", "profit_factor", "win_rate"]

_METRIC_ATTRS: dict[str, str] = {
    "total_return": "total_return",
    "sharpe_ratio": "sharpe_ratio",
    "profit_factor": "profit_factor",
    "win_rate": "win_rate",
}


@dataclass(slots=True)
class OptimizationResult:
    """网格搜索的汇总结果。"""

    best_params: dict[str, Any]
    best_metric_value: float
    metric_name: str
    all_results: list[dict[str, Any]]
    total_combinations: int
    elapsed_seconds: float


def _bars_snapshot(data: BarDataFeed | list[Bar]) -> list[Bar]:
    if isinstance(data, list):
        return list(data)
    return list(data._bars)


def _backtest_metrics(result: BacktestResult) -> dict[str, float | int]:
    """将 BacktestResult 中的主要数值指标展平为可排序、可序列化的字典。"""
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


class GridOptimizer:
    """对注册策略做参数网格搜索，每次组合独立回测。"""

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

    def run(self) -> OptimizationResult:
        keys = list(self._param_grid.keys())
        value_lists = [self._param_grid[k] for k in keys]
        combinations = list(itertools.product(*value_lists)) if keys else [()]
        total = len(combinations)

        wall_start = time.monotonic()
        bars_src = _bars_snapshot(self._data)
        rows: list[dict[str, Any]] = []

        for combo in combinations:
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
            mval = _metric_value(result, self._metric)
            metrics = _backtest_metrics(result)
            row: dict[str, Any] = {"params": dict(params), **metrics}
            rows.append(row)
            logger.debug(
                "GridOptimizer: params=%s %s=%.6f",
                params,
                self._metric,
                mval,
            )

        elapsed = time.monotonic() - wall_start
        rows.sort(key=lambda r: r[self._metric], reverse=True)

        if not rows:
            return OptimizationResult(
                best_params={},
                best_metric_value=float("-inf"),
                metric_name=self._metric,
                all_results=[],
                total_combinations=0,
                elapsed_seconds=elapsed,
            )

        best = rows[0]
        return OptimizationResult(
            best_params=dict(best["params"]),
            best_metric_value=float(best[self._metric]),
            metric_name=self._metric,
            all_results=rows,
            total_combinations=total,
            elapsed_seconds=elapsed,
        )


def quick_optimize(
    strategy_name: str,
    bars: list[Bar],
    param_grid: dict[str, list[Any]],
    *,
    fixed_params: dict[str, Any] | None = None,
    metric: MetricName | str = "sharpe_ratio",
    initial_capital: Decimal = Decimal("1000000"),
    commission_rate: Decimal = Decimal("0.0001"),
    slippage_ticks: int = 1,
    tick_size: Decimal = Decimal("1"),
    contract_multiplier: int = 1,
    default_volume: int = 1,
) -> OptimizationResult:
    """便捷入口：从 K 线推导标的与区间，其余使用常见默认回测参数。"""
    if not bars:
        raise ValueError("bars 不能为空")
    symbols = sorted({b.symbol for b in bars})
    start_date = min(b.dt for b in bars)
    end_date = max(b.dt for b in bars)
    config = BacktestConfig(
        strategy_id=f"quick_opt_{strategy_name}",
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        slippage_ticks=slippage_ticks,
        tick_size=tick_size,
        contract_multiplier=contract_multiplier,
    )
    opt = GridOptimizer(
        config=config,
        strategy_name=strategy_name,
        fixed_params=fixed_params or {},
        param_grid=param_grid,
        data=bars,
        metric=metric,
        default_volume=default_volume,
    )
    return opt.run()
