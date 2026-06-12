"""Walk-Forward ML validation for crypto.

Replaces random train/test split with temporal walk-forward cross-validation.
Each fold trains on a growing window and tests on the next unseen period,
eliminating temporal leakage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WFSplit:
    """One walk-forward train/test split."""
    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_size: int = 0
    test_size: int = 0


@dataclass
class WFResult:
    """Walk-forward validation result."""
    n_folds: int = 0
    test_accuracies: list[float] = field(default_factory=list)
    mean_accuracy: float = 0.0
    std_accuracy: float = 0.0
    min_accuracy: float = 0.0
    max_accuracy: float = 0.0
    per_fold: list[dict[str, Any]] = field(default_factory=list)


def walk_forward_splits(
    n_samples: int,
    n_folds: int = 5,
    min_train_ratio: float = 0.3,
    test_size_ratio: float = 0.1,
) -> list[WFSplit]:
    """Generate walk-forward splits.

    The first fold uses min_train_ratio of data for training.
    Each subsequent fold grows the training window by one test block.
    """
    test_size = max(int(n_samples * test_size_ratio), 50)
    min_train = max(int(n_samples * min_train_ratio), 200)

    splits = []
    for i in range(n_folds):
        train_end = min_train + i * test_size
        test_start = train_end
        test_end = min(test_start + test_size, n_samples)

        if test_end > n_samples or test_start >= n_samples:
            break

        splits.append(WFSplit(
            fold=i,
            train_start=0,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            train_size=train_end,
            test_size=test_end - test_start,
        ))

    return splits


def walk_forward_validate(
    X: np.ndarray,
    y: np.ndarray,
    model_factory,
    n_folds: int = 5,
    min_train_ratio: float = 0.3,
    test_size_ratio: float = 0.1,
) -> WFResult:
    """Run walk-forward validation.

    model_factory: callable that returns a model with .fit(X, y) and .predict(X) methods.
    """
    splits = walk_forward_splits(len(X), n_folds, min_train_ratio, test_size_ratio)
    result = WFResult(n_folds=len(splits))

    for split in splits:
        X_train = X[split.train_start:split.train_end]
        y_train = y[split.train_start:split.train_end]
        X_test = X[split.test_start:split.test_end]
        y_test = y[split.test_start:split.test_end]

        model = model_factory()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        accuracy = np.mean(preds == y_test)
        result.test_accuracies.append(float(accuracy))
        result.per_fold.append({
            "fold": split.fold,
            "train_size": split.train_size,
            "test_size": split.test_size,
            "accuracy": float(accuracy),
        })

        logger.info(
            "WF fold %d: train=%d test=%d acc=%.4f",
            split.fold, split.train_size, split.test_size, accuracy,
        )

    if result.test_accuracies:
        accs = result.test_accuracies
        result.mean_accuracy = float(np.mean(accs))
        result.std_accuracy = float(np.std(accs))
        result.min_accuracy = float(np.min(accs))
        result.max_accuracy = float(np.max(accs))

    return result
