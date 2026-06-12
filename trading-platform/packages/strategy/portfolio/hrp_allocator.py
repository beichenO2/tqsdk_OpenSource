"""Hierarchical Risk Parity (HRP) Portfolio Allocator (López de Prado, 2016).

Allocates capital across strategies (or assets) based on their correlation
structure and individual variance. Unlike Markowitz, HRP doesn't require
an invertible covariance matrix and is numerically stable.

Pipeline:
1. Compute correlation matrix from strategy returns
2. Hierarchical clustering (single linkage)
3. Quasi-diagonalization (reorder to cluster structure)
4. Recursive bisection to allocate weights inversely to variance
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _correlation_distance(corr: np.ndarray) -> np.ndarray:
    """Convert correlation matrix to distance matrix."""
    return np.sqrt(0.5 * (1 - corr))


def _single_linkage_cluster(dist: np.ndarray) -> list[tuple[int, int, float, int]]:
    """Simple single-linkage agglomerative clustering."""
    n = dist.shape[0]
    clusters: dict[int, list[int]] = {i: [i] for i in range(n)}
    linkage = []
    next_id = n

    active = set(range(n))
    d = dist.copy()
    np.fill_diagonal(d, np.inf)

    for _ in range(n - 1):
        min_dist = np.inf
        merge_i, merge_j = -1, -1
        active_list = sorted(active)
        for ii, ci in enumerate(active_list):
            for cj in active_list[ii + 1:]:
                if d[ci, cj] < min_dist:
                    min_dist = d[ci, cj]
                    merge_i, merge_j = ci, cj

        new_size = len(clusters[merge_i]) + len(clusters[merge_j])
        linkage.append((merge_i, merge_j, min_dist, new_size))
        clusters[next_id] = clusters[merge_i] + clusters[merge_j]

        new_row = np.full(d.shape[0], np.inf)
        for k in active:
            if k != merge_i and k != merge_j:
                new_row[k] = min(d[merge_i, k], d[merge_j, k])

        if next_id >= d.shape[0]:
            d = np.pad(d, ((0, 1), (0, 1)), constant_values=np.inf)
            new_row = np.pad(new_row, (0, 1), constant_values=np.inf)

        d[next_id, :len(new_row)] = new_row[:d.shape[1]]
        d[:d.shape[0], next_id] = new_row[:d.shape[0]]
        d[next_id, next_id] = np.inf

        active.discard(merge_i)
        active.discard(merge_j)
        active.add(next_id)
        del clusters[merge_i]
        del clusters[merge_j]
        next_id += 1

    return linkage


def _get_quasi_diag(linkage: list[tuple[int, int, float, int]], n: int) -> list[int]:
    """Reorder items to match hierarchical structure."""
    if n == 1:
        return [0]

    node_children: dict[int, tuple[int, int]] = {}
    for i, (left, right, _, _) in enumerate(linkage):
        node_children[n + i] = (int(left), int(right))

    root = n + len(linkage) - 1

    def _expand(node: int) -> list[int]:
        if node < n:
            return [node]
        left, right = node_children[node]
        return _expand(left) + _expand(right)

    return _expand(root)


def _recursive_bisection(
    cov: np.ndarray, sorted_idx: list[int],
) -> np.ndarray:
    """Allocate weights by recursive bisection of sorted clusters."""
    n = len(sorted_idx)
    weights = np.ones(n)

    clusters = [sorted_idx]
    while clusters:
        new_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            left = cluster[:mid]
            right = cluster[mid:]

            left_var = _cluster_var(cov, left)
            right_var = _cluster_var(cov, right)

            alloc = 1 - left_var / (left_var + right_var) if (left_var + right_var) > 0 else 0.5

            for i in left:
                idx = sorted_idx.index(i)
                weights[idx] *= alloc
            for i in right:
                idx = sorted_idx.index(i)
                weights[idx] *= (1 - alloc)

            new_clusters.extend([left, right])
        clusters = new_clusters

    return weights


def _cluster_var(cov: np.ndarray, items: list[int]) -> float:
    """Inverse-variance weighted portfolio variance of a cluster."""
    sub_cov = cov[np.ix_(items, items)]
    diag = np.diag(sub_cov)
    diag = np.where(diag > 0, diag, 1e-10)
    ivp = 1.0 / diag
    ivp_sum = ivp.sum()
    if ivp_sum <= 0:
        return 1.0
    w = ivp / ivp_sum
    return float(w @ sub_cov @ w)


class HRPAllocator:
    """Hierarchical Risk Parity allocator for strategy/asset returns."""

    def __init__(self, min_weight: float = 0.05, max_weight: float = 0.60):
        self.min_weight = min_weight
        self.max_weight = max_weight

    def allocate(
        self,
        returns_dict: dict[str, list[float]],
    ) -> dict[str, float]:
        """Compute HRP weights from strategy return histories.

        Args:
            returns_dict: {strategy_name: [return_1, return_2, ...]}

        Returns:
            {strategy_name: weight}  where weights sum to 1.0
        """
        names = list(returns_dict.keys())
        n = len(names)

        if n == 0:
            return {}
        if n == 1:
            return {names[0]: 1.0}

        min_len = min(len(v) for v in returns_dict.values())
        if min_len < 5:
            return {name: 1.0 / n for name in names}

        returns_matrix = np.array([returns_dict[name][-min_len:] for name in names])

        corr = np.corrcoef(returns_matrix)
        corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
        np.fill_diagonal(corr, 1.0)

        cov = np.cov(returns_matrix)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
        cov = np.nan_to_num(cov, nan=0.0)

        dist = _correlation_distance(corr)
        linkage = _single_linkage_cluster(dist)
        sorted_idx = _get_quasi_diag(linkage, n)
        raw_weights = _recursive_bisection(cov, sorted_idx)

        weights = {}
        for i, name in enumerate(names):
            idx = sorted_idx.index(i)
            w = float(raw_weights[idx])
            w = max(self.min_weight, min(self.max_weight, w))
            weights[name] = w

        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        logger.info("HRP allocation: %s", {k: f"{v:.3f}" for k, v in weights.items()})
        return weights


class RiskParityAllocator:
    """Equal Risk Contribution (Risk Parity) allocator.

    Each strategy contributes equally to total portfolio risk.
    """

    def allocate(self, returns_dict: dict[str, list[float]]) -> dict[str, float]:
        names = list(returns_dict.keys())
        n = len(names)
        if n == 0:
            return {}
        if n == 1:
            return {names[0]: 1.0}

        min_len = min(len(v) for v in returns_dict.values())
        if min_len < 5:
            return {name: 1.0 / n for name in names}

        vols = {}
        for name in names:
            rets = returns_dict[name][-min_len:]
            std = float(np.std(rets))
            vols[name] = max(std, 1e-10)

        inv_vol = {name: 1.0 / v for name, v in vols.items()}
        total = sum(inv_vol.values())
        weights = {name: v / total for name, v in inv_vol.items()}

        logger.info("Risk Parity allocation: %s", {k: f"{v:.3f}" for k, v in weights.items()})
        return weights
