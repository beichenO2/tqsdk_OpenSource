"""因子子系统 — IC/IR/去重/合成（对标 PRD §4.2）。

桥接现有 ``packages/features`` 注册表，新增分析与合成能力。
"""

from __future__ import annotations

from factor.analysis import (
    correlation_matrix,
    deduplicate_factors,
    factor_ic,
    ic_decay,
    summarize_ic,
)
from factor.alphalens_cs import (
    analyze_cross_section,
    cross_sectional_ic,
    quantile_returns,
)
from factor.combine import combine_equal_weight, combine_ic_weight, orthogonalize
from factor.evolution import FactorBandit, run_evolution_round
from factor.registry import ensure_features_loaded, list_factor_metas

__all__ = [
    "ensure_features_loaded",
    "list_factor_metas",
    "factor_ic",
    "ic_decay",
    "summarize_ic",
    "correlation_matrix",
    "deduplicate_factors",
    "combine_equal_weight",
    "combine_ic_weight",
    "orthogonalize",
    "cross_sectional_ic",
    "quantile_returns",
    "analyze_cross_section",
    "FactorBandit",
    "run_evolution_round",
]
