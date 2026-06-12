"""Optuna 超参数搜索 — 为 ML/DL/RL 模型自动调参。

用法:
    searcher = OptunaHyperSearch(
        objective_fn=my_objective,
        param_space={"learning_rate": (1e-5, 1e-2, "log"), "max_depth": (3, 12)},
    )
    best = searcher.run(n_trials=50)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

try:
    import optuna
    from optuna import Trial
except ImportError as _e:
    optuna = None  # type: ignore[assignment]
    _OPTUNA_ERR = _e
else:
    _OPTUNA_ERR = None


def _require_optuna():
    if optuna is None:
        raise ImportError(f"optuna required: {_OPTUNA_ERR!r}") from _OPTUNA_ERR


ParamSpec = dict[str, tuple[Any, ...]]


@dataclass
class SearchResult:
    best_params: dict[str, Any] = field(default_factory=dict)
    best_value: float = 0.0
    n_trials: int = 0
    all_trials: list[dict[str, Any]] = field(default_factory=list)


def _suggest_param(trial: Trial, name: str, spec: tuple[Any, ...]) -> Any:
    """Interpret a param spec tuple into an Optuna suggestion.

    Spec formats:
        (low, high)                → int if both int, else float
        (low, high, "log")         → float with log=True
        (low, high, "int")         → int
        ("cat", [choices])         → categorical
        ([choice1, choice2, ...])  → categorical (single-tuple shortcut)
    """
    if len(spec) == 1 and isinstance(spec[0], (list, tuple)):
        return trial.suggest_categorical(name, list(spec[0]))

    if len(spec) >= 2 and spec[0] == "cat":
        return trial.suggest_categorical(name, list(spec[1]))

    low, high = spec[0], spec[1]
    log = len(spec) > 2 and spec[2] == "log"
    as_int = len(spec) > 2 and spec[2] == "int"

    if as_int or (isinstance(low, int) and isinstance(high, int) and not log):
        return trial.suggest_int(name, int(low), int(high))

    return trial.suggest_float(name, float(low), float(high), log=log)


class OptunaHyperSearch:
    """Wraps Optuna for hyperparameter optimization."""

    def __init__(
        self,
        objective_fn: Callable[[dict[str, Any]], float],
        param_space: ParamSpec,
        direction: str = "maximize",
        study_name: str | None = None,
        storage: str | None = None,
    ):
        _require_optuna()
        self.objective_fn = objective_fn
        self.param_space = param_space
        self.direction = direction
        self.study_name = study_name or "trading_hp_search"
        self.storage = storage

    def run(self, n_trials: int = 50, timeout: int | None = None) -> SearchResult:
        _require_optuna()
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            storage=self.storage,
            load_if_exists=True,
        )

        def _objective(trial: Trial) -> float:
            params = {
                name: _suggest_param(trial, name, spec)
                for name, spec in self.param_space.items()
            }
            return self.objective_fn(params)

        study.optimize(_objective, n_trials=n_trials, timeout=timeout)

        all_trials = []
        for t in study.trials:
            all_trials.append({
                "number": t.number,
                "params": t.params,
                "value": t.value,
                "state": str(t.state),
            })

        return SearchResult(
            best_params=study.best_params,
            best_value=study.best_value,
            n_trials=len(study.trials),
            all_trials=all_trials,
        )


def ml_objective_factory(
    X_train, y_train, X_val, y_val, framework: str = "xgboost",
) -> Callable[[dict[str, Any]], float]:
    """Create an objective function for ML hyperparameter search."""
    import asyncio
    from ml.base import MLFramework, MLModelMeta

    def objective(params: dict[str, Any]) -> float:
        if framework == "xgboost":
            from ml.xgboost_model import XGBoostModel
            meta = MLModelMeta(
                model_id="hp_search",
                name="HP Search",
                framework=MLFramework.XGBOOST,
                hyperparams=params,
            )
            model = XGBoostModel(meta)
        else:
            from ml.lightgbm_model import LightGBMModel
            meta = MLModelMeta(
                model_id="hp_search",
                name="HP Search",
                framework=MLFramework.LIGHTGBM,
                hyperparams=params,
            )
            model = LightGBMModel(meta)

        result = asyncio.run(model.train(X_train, y_train, X_val, y_val))
        return result.val_score or result.train_score

    return objective


XGBOOST_PARAM_SPACE: ParamSpec = {
    "max_depth": (3, 10, "int"),
    "n_estimators": (50, 500, "int"),
    "learning_rate": (0.01, 0.3, "log"),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.5, 1.0),
    "min_child_weight": (1, 10, "int"),
}

LIGHTGBM_PARAM_SPACE: ParamSpec = {
    "max_depth": (3, 10, "int"),
    "n_estimators": (50, 500, "int"),
    "learning_rate": (0.01, 0.3, "log"),
    "num_leaves": (15, 63, "int"),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.5, 1.0),
    "min_child_samples": (5, 50, "int"),
}

PPO_PARAM_SPACE: ParamSpec = {
    "learning_rate": (1e-5, 1e-3, "log"),
    "gamma": (0.95, 0.999),
    "clip_range": (0.1, 0.3),
    "n_epochs": (3, 15, "int"),
    "batch_size": ("cat", [32, 64, 128, 256]),
}
