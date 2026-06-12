"""Walk-forward analysis utilities for rolling train/test evaluation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any

from .datafeed import BarDataFeed
from .param_sweep import MetricName, ParameterSweep

if TYPE_CHECKING:
    from concurrent.futures import Executor

    from .models import BacktestConfig, Bar


def _bars_snapshot(data: BarDataFeed | list[Bar]) -> list[Bar]:
    if isinstance(data, list):
        bars = list(data)
    else:
        bars = list(data._bars)
    bars.sort(key=lambda b: b.dt)
    return bars


def _slice_bars(bars: list[Bar], start: datetime, end: datetime) -> list[Bar]:
    return [b for b in bars if start <= b.dt <= end]


@dataclass(slots=True)
class WalkForwardWindowResult:
    """Single rolling window output."""

    window_index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_best_params: dict[str, Any]
    train_best_score: float
    test_score: float
    test_metrics: dict[str, float | int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_index": self.window_index,
            "train_range": {
                "start": self.train_start.isoformat(),
                "end": self.train_end.isoformat(),
            },
            "test_range": {
                "start": self.test_start.isoformat(),
                "end": self.test_end.isoformat(),
            },
            "train_best_params": dict(self.train_best_params),
            "train_best_score": self.train_best_score,
            "test_score": self.test_score,
            "test_metrics": dict(self.test_metrics),
        }


class WalkForwardAnalyzer:
    """Rolling walk-forward analyzer using parameter sweep composition."""

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
        self._config = config
        self._strategy_name = strategy_name
        self._fixed_params = dict(fixed_params)
        self._param_grid = {k: list(v) for k, v in param_grid.items()}
        self._data = data
        self._metric = metric
        self._default_volume = default_volume

    def walk_forward_analysis(
        self,
        train_window: int,
        test_window: int,
        step_window: int | None = None,
        *,
        max_workers: int | None = None,
        executor: Executor | None = None,
    ) -> dict[str, Any]:
        if train_window < 1:
            raise ValueError("train_window must be >= 1")
        if test_window < 1:
            raise ValueError("test_window must be >= 1")

        step = test_window if step_window is None else step_window
        if step < 1:
            raise ValueError("step_window must be >= 1")

        bars = _bars_snapshot(self._data)
        if not bars:
            raise ValueError("data must include at least one bar")

        timeline = sorted({b.dt for b in bars})
        required_points = train_window + test_window
        if len(timeline) < required_points:
            raise ValueError(
                f"insufficient data points for one window: need {required_points}, got {len(timeline)}"
            )

        windows: list[WalkForwardWindowResult] = []
        cursor = 0
        window_idx = 0
        while cursor + required_points <= len(timeline):
            train_start = timeline[cursor]
            train_end = timeline[cursor + train_window - 1]
            test_start = timeline[cursor + train_window]
            test_end = timeline[cursor + required_points - 1]

            train_bars = _slice_bars(bars, train_start, train_end)
            test_bars = _slice_bars(bars, test_start, test_end)
            if not train_bars or not test_bars:
                raise ValueError(
                    f"empty train/test bars at window {window_idx}: "
                    f"train={len(train_bars)}, test={len(test_bars)}"
                )

            train_config = replace(
                self._config,
                strategy_id=f"{self._config.strategy_id}_wf_train_{window_idx}",
                start_date=train_start,
                end_date=train_end,
            )
            train_sweep = ParameterSweep(
                config=train_config,
                strategy_name=self._strategy_name,
                fixed_params=self._fixed_params,
                param_grid=self._param_grid,
                data=train_bars,
                metric=self._metric,
                default_volume=self._default_volume,
            )
            train_result = train_sweep.parameter_sweep(
                max_workers=max_workers,
                executor=executor,
            )
            best_params = dict(train_result["best_params"])

            test_config = replace(
                self._config,
                strategy_id=f"{self._config.strategy_id}_wf_test_{window_idx}",
                start_date=test_start,
                end_date=test_end,
            )
            test_sweep = ParameterSweep(
                config=test_config,
                strategy_name=self._strategy_name,
                fixed_params=best_params,
                param_grid={},
                data=test_bars,
                metric=self._metric,
                default_volume=self._default_volume,
            )
            test_result = test_sweep.parameter_sweep(max_workers=1)
            test_row = test_result["all_results"][0]

            windows.append(
                WalkForwardWindowResult(
                    window_index=window_idx,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    train_best_params=best_params,
                    train_best_score=float(train_result["best_score"]),
                    test_score=float(test_row["score"]),
                    test_metrics=dict(test_row["metrics"]),
                )
            )

            cursor += step
            window_idx += 1

        windows_dict = [w.to_dict() for w in windows]
        test_scores = [w.test_score for w in windows]
        avg_metrics: dict[str, float] = {}
        if windows:
            metric_keys = set().union(*(w.test_metrics.keys() for w in windows))
            for key in metric_keys:
                values = [float(w.test_metrics[key]) for w in windows if key in w.test_metrics]
                if values:
                    avg_metrics[key] = sum(values) / len(values)

        summary = {
            "metric_name": self._metric,
            "total_windows": len(windows),
            "average_test_score": (sum(test_scores) / len(test_scores)) if test_scores else 0.0,
            "best_test_score": max(test_scores) if test_scores else 0.0,
            "worst_test_score": min(test_scores) if test_scores else 0.0,
            "average_test_metrics": avg_metrics,
        }

        return {
            "metric_name": self._metric,
            "train_window": train_window,
            "test_window": test_window,
            "step_window": step,
            "windows": windows_dict,
            "summary": summary,
        }


def walk_forward_analysis(
    config: BacktestConfig,
    strategy_name: str,
    fixed_params: dict[str, Any],
    param_grid: dict[str, list[Any]],
    data: BarDataFeed | list[Bar],
    *,
    train_window: int,
    test_window: int,
    step_window: int | None = None,
    metric: MetricName | str = "sharpe_ratio",
    default_volume: int = 1,
    max_workers: int | None = None,
    executor: Executor | None = None,
) -> dict[str, Any]:
    """Convenience function around :class:`WalkForwardAnalyzer`."""
    analyzer = WalkForwardAnalyzer(
        config=config,
        strategy_name=strategy_name,
        fixed_params=fixed_params,
        param_grid=param_grid,
        data=data,
        metric=metric,
        default_volume=default_volume,
    )
    return analyzer.walk_forward_analysis(
        train_window=train_window,
        test_window=test_window,
        step_window=step_window,
        max_workers=max_workers,
        executor=executor,
    )


__all__ = ["WalkForwardAnalyzer", "walk_forward_analysis", "WalkForwardWindowResult"]
