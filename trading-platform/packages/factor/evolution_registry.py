"""将 LLM 因子进化产物注册进 FactorRegistry。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]


def _eligible(candidate: dict[str, Any]) -> bool:
    if not candidate.get("expr"):
        return False
    if candidate.get("error"):
        return False
    if candidate.get("ic_mean") is None:
        return False
    if candidate.get("score") is None:
        return False
    return True


def register_evolved_factors(path: str | Path | None = None) -> list[str]:
    """读取进化产物 JSON，将达标因子注册进 FactorRegistry。

    文件不存在时返回空列表，不抛异常。
    """
    json_path = Path(path) if path else ROOT / "output" / "factor_evolution" / "latest.json"
    if not json_path.exists():
        return []

    try:
        data = json.loads(json_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Failed to read evolution artifact %s: %s", json_path, e)
        return []

    from factor.evolution import evaluate_expression
    from factor.registry import get_registry

    registry = get_registry()
    registered: list[str] = []
    seen_exprs: set[str] = set()

    candidates: list[dict[str, Any]] = list(data.get("candidates") or [])
    best = data.get("best")
    if isinstance(best, dict) and best not in candidates:
        candidates.insert(0, best)

    idx = 0
    for cand in candidates:
        if not _eligible(cand):
            continue
        expr = str(cand["expr"]).strip()
        if expr in seen_exprs:
            continue
        seen_exprs.add(expr)

        name = f"evolved_{idx}"

        def _make_compute(expression: str, col: str):
            def compute(df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
                series = evaluate_expression(expression, df)
                out = df.copy()
                out[col] = series
                return out

            return compute

        registry.register(
            name=name,
            category="evolved",
            description=f"Evolved: {expr}",
            output_columns=[name],
            compute_fn=_make_compute(expr, name),
            expr=expr,
            ic_mean=cand.get("ic_mean"),
            score=cand.get("score"),
        )
        registered.append(name)
        idx += 1

    return registered
