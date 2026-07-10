"""MCTS factor expression search with subtree avoidance + failure experience store.

Design aligned with AAAI-2026 Alpha Jungle (LLM+MCTS) and QuantaAlpha (trajectory
search / subtree frequency de-homogenization).
"""

from __future__ import annotations

import ast
import json
import logging
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from factor.evolution import (
    OUT_DIR as EVOLUTION_OUT_DIR,
    ROOT,
    FactorCandidate,
    _SEED_EXPRS,
    _validate_expr,
    mutate_expression,
    propose_via_llm,
    score_candidate,
)
from factor.evolution_registry import (
    EvolutionGateConfig,
    classify_candidates,
)

logger = logging.getLogger("factor-mcts")

OUT_DIR = EVOLUTION_OUT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_EXPERIENCE_PATH = OUT_DIR / "experience.jsonl"
FAILURE_REASONS = frozenset({
    "validation_error",
    "low_ic",
    "duplicate",
    "overused_subtree",
})


@dataclass
class MCTSNode:
    expr: str
    parent: MCTSNode | None = None
    children: list[MCTSNode] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0
    is_terminal: bool = False
    depth: int = 0
    candidate: FactorCandidate | None = None

    @property
    def mean_reward(self) -> float:
        if self.visits <= 0:
            return 0.0
        return self.total_reward / self.visits


class ExperienceStore:
    """JSONL-backed failure experience library for MCTS / LLM negative examples."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else DEFAULT_EXPERIENCE_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        expr: str,
        reason: str,
        *,
        ic_mean: float | None = None,
        ir: float | None = None,
    ) -> None:
        if reason not in FAILURE_REASONS:
            raise ValueError(f"unknown reason: {reason}")
        row = {
            "ts": time.time(),
            "expr": expr,
            "reason": reason,
            "ic_mean": ic_mean,
            "ir": ir,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def load_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def top_failure_patterns(self, n: int = 5) -> list[dict[str, Any]]:
        """Return most common failure reasons with a representative expr."""
        records = self.load_all()
        if not records:
            return []
        reason_counts = Counter(r.get("reason") for r in records if r.get("reason"))
        by_reason: dict[str, list[dict[str, Any]]] = {}
        for r in records:
            reason = r.get("reason")
            if not reason:
                continue
            by_reason.setdefault(reason, []).append(r)

        patterns: list[dict[str, Any]] = []
        for reason, count in reason_counts.most_common(n):
            reps = by_reason.get(reason, [])
            expr = reps[0].get("expr", "") if reps else ""
            patterns.append({"reason": reason, "count": count, "expr": expr})
        return patterns


def _strip_ast_locations(node: ast.AST) -> ast.AST:
    """Return a copy of *node* with lineno/col_offset cleared for stable dumps."""
    for child in ast.walk(node):
        for attr in ("lineno", "col_offset", "end_lineno", "end_col_offset"):
            if hasattr(child, attr):
                setattr(child, attr, None)
    return node


def extract_subtrees(expr: str, min_depth: int = 2) -> list[str]:
    """Collect normalized AST dumps of subtrees with depth >= min_depth.

    Depth of a leaf (Name/Constant) is 1; Call/BinOp etc. are 1 + max(child).
    """
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return []

    results: list[str] = []

    def _depth(node: ast.AST) -> int:
        kids = list(ast.iter_child_nodes(node))
        if not kids:
            return 1
        return 1 + max(_depth(k) for k in kids)

    # Walk body of Expression
    root = tree.body if isinstance(tree, ast.Expression) else tree
    for node in ast.walk(root):
        if isinstance(node, (ast.Expression,)):
            continue
        d = _depth(node)
        if d < min_depth:
            continue
        clone = _strip_ast_locations(ast.parse(ast.unparse(node), mode="eval").body)
        results.append(ast.dump(clone, include_attributes=False))
    return results


def apply_subtree_penalty(
    expr: str,
    reward: float,
    subtree_counts: dict[str, int],
    *,
    threshold: int = 5,
    penalty: float = 0.5,
    experience: ExperienceStore | None = None,
) -> tuple[float, bool]:
    """If any subtree of *expr* exceeds *threshold*, multiply reward by *penalty*."""
    subs = extract_subtrees(expr)
    overused = [s for s in subs if subtree_counts.get(s, 0) > threshold]
    if not overused:
        return reward, False
    if experience is not None:
        experience.record(expr, "overused_subtree")
    return reward * penalty, True


def select_ucb1(node: MCTSNode, c: float = 1.4) -> MCTSNode:
    """Pick child of *node* maximizing UCB1. Unvisited children get +inf."""
    if not node.children:
        raise ValueError("select_ucb1 requires children")
    parent_visits = max(node.visits, 1)
    best: MCTSNode | None = None
    best_ucb = float("-inf")
    for child in node.children:
        if child.visits == 0:
            ucb = float("inf")
        else:
            ucb = child.mean_reward + c * math.sqrt(math.log(parent_visits) / child.visits)
        if ucb > best_ucb:
            best_ucb = ucb
            best = child
    assert best is not None
    return best


def select_path(roots: list[MCTSNode], c: float = 1.4) -> MCTSNode:
    """Multi-root UCB selection: pick among roots, then descend to a leaf/expandable."""
    if not roots:
        raise ValueError("no roots")

    # Virtual parent for multi-root UCB
    total_root_visits = sum(r.visits for r in roots) or 1
    best_root: MCTSNode | None = None
    best_ucb = float("-inf")
    for r in roots:
        if r.visits == 0:
            ucb = float("inf")
        else:
            ucb = r.mean_reward + c * math.sqrt(math.log(total_root_visits) / r.visits)
        if ucb > best_ucb:
            best_ucb = ucb
            best_root = r
    assert best_root is not None

    node = best_root
    while node.children and not node.is_terminal:
        # Prefer expandable (non-fully-expanded) leaves: if all children exist,
        # descend via UCB; stop when we hit a node we want to expand further.
        # Standard MCTS: descend while fully expanded; here we always descend
        # while children exist, and expand at the leaf.
        node = select_ucb1(node, c=c)
    return node


def expand_node(
    parent: MCTSNode,
    *,
    k: int = 3,
    use_llm: bool = False,
    experience: ExperienceStore | None = None,
    subtree_counts: dict[str, int] | None = None,
    subtree_threshold: int = 5,
    seen_exprs: set[str] | None = None,
) -> list[MCTSNode]:
    """Generate up to *k* validated child nodes from *parent* via mutate/LLM."""
    _ = subtree_counts, subtree_threshold  # reserved; penalty applied at evaluate time
    seen_exprs = seen_exprs if seen_exprs is not None else set()
    proposals: list[str] = []

    if use_llm:
        ctx: dict[str, Any] = {
            "existing": [parent.expr],
            "parent_expr": parent.expr,
            "goal": "maximize |IC| vs forward return, diversify from parent",
        }
        if experience is not None:
            patterns = experience.top_failure_patterns(5)
            if patterns:
                ctx["failure_patterns"] = patterns
        llm_exprs = propose_via_llm(ctx, n=k)
        proposals.extend(llm_exprs)

    while len(proposals) < k:
        proposals.append(mutate_expression(parent.expr))

    children: list[MCTSNode] = []
    for expr in proposals[:k]:
        err = _validate_expr(expr)
        if err is not None:
            if experience is not None:
                experience.record(expr, "validation_error")
            continue
        if expr in seen_exprs:
            if experience is not None:
                experience.record(expr, "duplicate")
            continue
        child = MCTSNode(
            expr=expr,
            parent=parent,
            depth=parent.depth + 1,
        )
        parent.children.append(child)
        children.append(child)
        seen_exprs.add(expr)
    return children


def _backpropagate(node: MCTSNode, reward: float) -> None:
    cur: MCTSNode | None = node
    while cur is not None:
        cur.visits += 1
        cur.total_reward += reward
        cur = cur.parent


def _register_subtrees(expr: str, subtree_counts: dict[str, int]) -> None:
    for s in extract_subtrees(expr):
        subtree_counts[s] = subtree_counts.get(s, 0) + 1


def _collect_all_nodes(roots: list[MCTSNode]) -> list[MCTSNode]:
    out: list[MCTSNode] = []
    stack = list(roots)
    while stack:
        n = stack.pop()
        out.append(n)
        stack.extend(n.children)
    return out


def run_mcts_search(
    df: pd.DataFrame,
    *,
    n_iterations: int = 50,
    use_llm: bool = False,
    seed_exprs: list[str] | None = None,
    gate_cfg: EvolutionGateConfig | None = None,
    experience_path: str | Path | None = None,
    k_expand: int = 3,
    ucb_c: float = 1.4,
    subtree_threshold: int = 5,
    low_ic_abs: float = 0.005,
) -> dict[str, Any]:
    """Run MCTS over factor expressions; return gate-compatible payload."""
    seeds = list(seed_exprs) if seed_exprs else list(_SEED_EXPRS[:5])
    experience = ExperienceStore(experience_path)
    subtree_counts: dict[str, int] = {}
    seen_exprs: set[str] = set()
    subtree_penalties = 0

    # Build existing panel from seeds for dedupe
    existing_cols: dict[str, pd.Series] = {}
    for i, ex in enumerate(seeds[:12]):
        try:
            from factor.evolution import evaluate_expression
            existing_cols[f"e{i}"] = evaluate_expression(ex, df)
        except Exception:
            continue
    existing_df = pd.DataFrame(existing_cols) if existing_cols else None

    roots: list[MCTSNode] = []
    for expr in seeds:
        err = _validate_expr(expr)
        if err is not None:
            experience.record(expr, "validation_error")
            continue
        node = MCTSNode(expr=expr, depth=0)
        cand = score_candidate(expr, df, existing_factors=existing_df)
        cand.source = "mcts"
        reward = float(cand.score) if cand.score is not None and cand.error is None else 0.0
        if cand.error is None and cand.ic_mean is not None and abs(float(cand.ic_mean)) < low_ic_abs:
            experience.record(expr, "low_ic", ic_mean=cand.ic_mean, ir=cand.ir)
        if cand.dedupe_ok is False:
            experience.record(expr, "duplicate", ic_mean=cand.ic_mean, ir=cand.ir)
        reward, hit = apply_subtree_penalty(
            expr, reward, subtree_counts,
            threshold=subtree_threshold, experience=experience,
        )
        if hit:
            subtree_penalties += 1
        _register_subtrees(expr, subtree_counts)
        cand.score = reward
        cand.meta = {"source": "mcts", "depth": 0, "visits": 0}
        node.candidate = cand
        _backpropagate(node, reward)
        roots.append(node)
        seen_exprs.add(expr)

    if not roots:
        # Fallback: ensure at least one root
        expr = "roc(close, 5)"
        node = MCTSNode(expr=expr, depth=0)
        cand = score_candidate(expr, df, existing_factors=existing_df)
        cand.source = "mcts"
        reward = float(cand.score or 0.0)
        cand.meta = {"source": "mcts", "depth": 0, "visits": 0}
        node.candidate = cand
        _backpropagate(node, reward)
        roots.append(node)
        seen_exprs.add(expr)

    for _ in range(n_iterations):
        leaf = select_path(roots, c=ucb_c)
        children = expand_node(
            leaf,
            k=k_expand,
            use_llm=use_llm,
            experience=experience,
            subtree_counts=subtree_counts,
            subtree_threshold=subtree_threshold,
            seen_exprs=seen_exprs,
        )
        if not children:
            # No valid expansion — mark terminal and backprop small reward
            leaf.is_terminal = True
            _backpropagate(leaf, 0.0)
            continue

        for child in children:
            cand = score_candidate(child.expr, df, existing_factors=existing_df)
            cand.source = "mcts"
            reward = float(cand.score) if cand.score is not None and cand.error is None else 0.0

            if cand.error is not None:
                experience.record(child.expr, "validation_error")
                reward = 0.0
            elif cand.ic_mean is not None and abs(float(cand.ic_mean)) < low_ic_abs:
                experience.record(
                    child.expr, "low_ic", ic_mean=cand.ic_mean, ir=cand.ir,
                )
            if cand.dedupe_ok is False:
                experience.record(
                    child.expr, "duplicate", ic_mean=cand.ic_mean, ir=cand.ir,
                )

            reward, hit = apply_subtree_penalty(
                child.expr, reward, subtree_counts,
                threshold=subtree_threshold, experience=experience,
            )
            if hit:
                subtree_penalties += 1
            _register_subtrees(child.expr, subtree_counts)

            cand.score = reward
            depth = child.depth
            cand.meta = {"source": "mcts", "depth": depth, "visits": 0}
            child.candidate = cand
            _backpropagate(child, reward)

    all_nodes = _collect_all_nodes(roots)
    max_depth = max((n.depth for n in all_nodes), default=0)

    # Sync visits into meta and build candidate list (unique by expr, prefer higher score)
    by_expr: dict[str, FactorCandidate] = {}
    for n in all_nodes:
        if n.candidate is None:
            continue
        n.candidate.meta["visits"] = n.visits
        n.candidate.meta["depth"] = n.depth
        prev = by_expr.get(n.expr)
        if prev is None or (n.candidate.score or 0) > (prev.score or 0):
            by_expr[n.expr] = n.candidate

    candidates = sorted(
        by_expr.values(),
        key=lambda c: (c.score is not None, c.score or 0.0),
        reverse=True,
    )
    cand_dicts = [asdict(c) for c in candidates]

    payload: dict[str, Any] = {
        "ts": time.time(),
        "candidates": cand_dicts,
        "tree_stats": {
            "nodes": len(all_nodes),
            "max_depth": max_depth,
            "subtree_penalties": subtree_penalties,
        },
    }
    classified = classify_candidates(payload, gate_cfg)
    payload["elite"] = classified["elite"]
    payload["qualified"] = classified["qualified"]

    ts_int = int(time.time())
    out_path = OUT_DIR / f"mcts_round_{ts_int}.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    try:
        payload["path"] = str(out_path.relative_to(ROOT))
    except ValueError:
        payload["path"] = str(out_path)
    return payload
