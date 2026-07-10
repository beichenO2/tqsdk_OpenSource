"""将 LLM 因子进化产物注册进 FactorRegistry（CogAlpha 式双阈值门控）。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class EvolutionGateConfig:
    """双阈值：qualified 作下轮种子；elite 才写入 FactorRegistry。"""

    qualified_ic_abs: float = 0.015
    qualified_ir_abs: float = 0.3
    elite_ic_abs: float = 0.03
    elite_ir_abs: float = 0.5


DEFAULT_GATE = EvolutionGateConfig()


def _collect_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = list(payload.get("candidates") or [])
    best = payload.get("best")
    if isinstance(best, dict):
        best_expr = str(best.get("expr") or "").strip()
        if best_expr and not any(str(c.get("expr", "")).strip() == best_expr for c in candidates):
            candidates.insert(0, best)
    return candidates


def _base_ok(candidate: dict[str, Any]) -> bool:
    if not candidate.get("expr"):
        return False
    if candidate.get("error"):
        return False
    if candidate.get("ic_mean") is None or candidate.get("ir") is None:
        return False
    return True


def _is_qualified(candidate: dict[str, Any], cfg: EvolutionGateConfig) -> bool:
    if not _base_ok(candidate):
        return False
    if not candidate.get("dedupe_ok", True):
        return False
    ic = float(candidate["ic_mean"])
    ir = float(candidate["ir"])
    return abs(ic) >= cfg.qualified_ic_abs and abs(ir) >= cfg.qualified_ir_abs


def _is_elite(candidate: dict[str, Any], cfg: EvolutionGateConfig) -> bool:
    if not _is_qualified(candidate, cfg):
        return False
    ic = float(candidate["ic_mean"])
    ir = float(candidate["ir"])
    return abs(ic) >= cfg.elite_ic_abs and abs(ir) >= cfg.elite_ir_abs


def classify_candidates(
    payload: dict[str, Any],
    cfg: EvolutionGateConfig | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """按 qualified / elite / rejected 分类进化候选。"""
    cfg = cfg or DEFAULT_GATE
    elite: list[dict[str, Any]] = []
    qualified: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for cand in _collect_candidates(payload):
        if _is_elite(cand, cfg):
            elite.append(cand)
        elif _is_qualified(cand, cfg):
            qualified.append(cand)
        else:
            rejected.append(cand)

    return {"elite": elite, "qualified": qualified, "rejected": rejected}


def _register_elite_list(
    elite: list[dict[str, Any]],
    *,
    registry: Any | None = None,
) -> list[str]:
    from factor.evolution import evaluate_expression
    from factor.registry import get_registry

    registry = registry or get_registry()
    registered: list[str] = []
    seen_exprs: set[str] = set()
    idx = 0

    for cand in elite:
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


def register_elite_from_payload(
    payload: dict[str, Any],
    cfg: EvolutionGateConfig | None = None,
) -> list[str]:
    """从进化 round 结果注册 elite 因子，返回注册名列表。"""
    classified = classify_candidates(payload, cfg)
    return _register_elite_list(classified["elite"])


def register_evolved_factors(
    path: str | Path | None = None,
    cfg: EvolutionGateConfig | None = None,
) -> list[str]:
    """读取进化产物 JSON，仅将 elite 因子注册进 FactorRegistry。

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

    return register_elite_from_payload(data, cfg)
