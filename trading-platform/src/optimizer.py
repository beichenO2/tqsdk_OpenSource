"""Grid search over strategy parameter combinations."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable


@dataclass(slots=True)
class GridOptimizeResult:
    """Outcome of a full grid evaluation."""

    best_params: dict[str, Any]
    best_score: float
    results: list[dict[str, Any]] = field(default_factory=list)


class GridOptimizer:
    """Exhaustive grid search: cartesian product of `param_grid` values."""

    def __init__(self, param_grid: dict[str, list[Any]]) -> None:
        self.param_grid = param_grid

    def run(
        self,
        evaluate: Callable[[dict[str, Any]], float],
        *,
        maximize: bool = True,
    ) -> GridOptimizeResult:
        """Evaluate every combination; `evaluate(params)` returns a scalar score."""
        keys = list(self.param_grid)
        if not keys:
            return GridOptimizeResult(best_params={}, best_score=0.0, results=[])

        results: list[dict[str, Any]] = []
        best_score: float | None = None
        best_params: dict[str, Any] = {}

        for combo in product(*(self.param_grid[k] for k in keys)):
            params = dict(zip(keys, combo, strict=True))
            score = float(evaluate(params))
            results.append({"params": params, "score": score})
            if best_score is None:
                best_score = score
                best_params = params
            elif maximize and score > best_score:
                best_score = score
                best_params = params
            elif not maximize and score < best_score:
                best_score = score
                best_params = params

        return GridOptimizeResult(
            best_params=best_params,
            best_score=float(best_score) if best_score is not None else 0.0,
            results=results,
        )
