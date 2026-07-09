"""LLM 因子表达式进化（RD-Agent(Q) 风格精简版）。

能力：
- Thompson / ε-greedy bandit：在「挖因子」与「调模型预算」之间分配
- 提案：LLM（PolarPrivate）生成表达式；404/不可用时模板变异降级
- 评估：单品种 IC + 与已有因子去重（|ρ|≥0.99 剔除）
- 产物：写入 output/factor_evolution/ 供 ResearchRun / Factors 消费

不自动开 live；仅产出候选因子元数据与分数。
"""

from __future__ import annotations

import ast
import json
import logging
import random
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger("factor-evolution")

ROOT = Path(__file__).resolve().parents[2]  # trading-platform/
OUT_DIR = ROOT / "output" / "factor_evolution"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Safe expression vocabulary (single-asset OHLCV columns + rolling helpers)
_ALLOWED_NAMES = {
    "open", "high", "low", "close", "volume",
    "log", "abs", "sign", "sqrt", "maximum", "minimum",
    "delta", "delay", "ts_mean", "ts_std", "ts_max", "ts_min",
    "ts_rank", "corr", "roc", "vwap",
}


@dataclass
class BanditArm:
    name: str
    alpha: float = 1.0  # Beta successes
    beta: float = 1.0   # Beta failures
    pulls: int = 0
    reward_sum: float = 0.0


@dataclass
class FactorCandidate:
    expr: str
    source: str  # llm | mutate | seed
    score: float | None = None
    ic_mean: float | None = None
    ir: float | None = None
    dedupe_ok: bool = True
    max_corr: float | None = None
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class FactorBandit:
    """Two-arm bandit: factor_mining vs model_tune."""

    def __init__(self, epsilon: float = 0.15):
        self.epsilon = epsilon
        self.arms = {
            "factor_mining": BanditArm("factor_mining"),
            "model_tune": BanditArm("model_tune"),
        }

    def select(self) -> str:
        if random.random() < self.epsilon:
            return random.choice(list(self.arms))
        # Thompson sampling
        samples = {
            name: random.betavariate(max(a.alpha, 1e-3), max(a.beta, 1e-3))
            for name, a in self.arms.items()
        }
        return max(samples, key=samples.get)

    def update(self, arm: str, reward: float) -> None:
        a = self.arms[arm]
        a.pulls += 1
        a.reward_sum += reward
        # map reward in [-1,1] approx to Bernoulli success
        p = 0.5 + 0.5 * max(-1.0, min(1.0, reward))
        a.alpha += p
        a.beta += 1.0 - p

    def snapshot(self) -> dict[str, Any]:
        return {
            name: {
                "pulls": a.pulls,
                "mean_reward": (a.reward_sum / a.pulls) if a.pulls else 0.0,
                "alpha": a.alpha,
                "beta": a.beta,
            }
            for name, a in self.arms.items()
        }


def _safe_env(df: pd.DataFrame) -> dict[str, Any]:
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    def delay(s, n: int = 1):
        return pd.Series(s).shift(int(n))

    def delta(s, n: int = 1):
        return pd.Series(s).diff(int(n))

    def ts_mean(s, w: int = 20):
        return pd.Series(s).rolling(int(w)).mean()

    def ts_std(s, w: int = 20):
        return pd.Series(s).rolling(int(w)).std()

    def ts_max(s, w: int = 20):
        return pd.Series(s).rolling(int(w)).max()

    def ts_min(s, w: int = 20):
        return pd.Series(s).rolling(int(w)).min()

    def ts_rank(s, w: int = 10):
        return pd.Series(s).rolling(int(w)).apply(
            lambda x: float(pd.Series(x).rank(pct=True).iloc[-1]) if len(x) else np.nan,
            raw=True,
        )

    def corr(a, b, w: int = 10):
        return pd.Series(a).rolling(int(w)).corr(pd.Series(b))

    def roc(s, n: int = 5):
        s = pd.Series(s)
        return s / s.shift(int(n)) - 1.0

    vwap = (h + l + c) / 3.0
    return {
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "log": np.log, "abs": np.abs, "sign": np.sign, "sqrt": np.sqrt,
        "maximum": np.maximum, "minimum": np.minimum,
        "delta": delta, "delay": delay,
        "ts_mean": ts_mean, "ts_std": ts_std, "ts_max": ts_max, "ts_min": ts_min,
        "ts_rank": ts_rank, "corr": corr, "roc": roc, "vwap": vwap,
    }


_ALLOWED_NODE_TYPES = (
    ast.Expression,
    ast.Load,
    ast.BinOp, ast.UnaryOp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
    ast.USub, ast.UAdd,
    ast.Call, ast.Name, ast.Constant, ast.keyword,
    ast.Compare, ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq,
)


def _validate_expr(expr: str) -> str | None:
    """Whitelist-based AST validation. Returns error message or None."""
    expr = expr.strip()
    if not expr or len(expr) > 240:
        return "empty or too long"
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        return f"syntax: {e}"
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            return f"disallowed node: {type(node).__name__}"
        if isinstance(node, ast.Name):
            if node.id not in _ALLOWED_NAMES:
                return f"disallowed name: {node.id}"
        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                return f"disallowed constant: {node.value!r}"
        elif isinstance(node, ast.Call):
            func = node.func
            if not (isinstance(func, ast.Name) and func.id in _ALLOWED_NAMES):
                return "disallowed call"
        elif isinstance(node, ast.keyword):
            if node.arg is None:  # **kwargs forbidden
                return "kwargs expansion forbidden"
    return None


def evaluate_expression(expr: str, df: pd.DataFrame) -> pd.Series:
    err = _validate_expr(expr)
    if err:
        raise ValueError(err)
    env = _safe_env(df)
    # Restricted eval — only env keys
    return pd.Series(eval(expr, {"__builtins__": {}}, env), index=df.index)  # noqa: S307


_SEED_EXPRS = [
    "roc(close, 5)",
    "roc(close, 20)",
    "-corr(open, volume, 10)",
    "(close - open) / ((high - low) + 0.001)",
    "ts_rank(delta(close, 1), 5)",
    "-ts_std(close, 20)",
    "(close - ts_min(low, 20)) / (ts_max(high, 20) - ts_min(low, 20) + 1e-9)",
    "sign(delta(volume, 1)) * (-delta(close, 1))",
    "corr(close, volume, 20)",
    "-delta(corr(high, volume, 5), 5)",
]


def mutate_expression(base: str | None = None) -> str:
    """Template mutation when LLM unavailable."""
    windows = [3, 5, 8, 10, 12, 20, 30, 60]
    w = random.choice(windows)
    w2 = random.choice(windows)
    templates = [
        f"roc(close, {w})",
        f"-corr(open, volume, {w})",
        f"ts_rank(delta(close, 1), {w})",
        f"-ts_std(close, {w})",
        f"corr(close, volume, {w})",
        f"delta(ts_mean(close, {w}), {w2})",
        f"sign(delta(volume, 1)) * (-delta(close, {w}))",
        f"(close - ts_mean(close, {w})) / (ts_std(close, {w}) + 1e-9)",
        f"-ts_rank(abs(delta(close, {w})), {w2}) * sign(delta(close, {w}))",
        f"(vwap - close) / (ts_std(close, {w}) + 1e-9)",
    ]
    if base and random.random() < 0.4:
        # tweak numbers in base
        def _repl(m: re.Match) -> str:
            return str(random.choice(windows))
        tweaked = re.sub(r"\b\d+\b", _repl, base, count=2)
        if _validate_expr(tweaked) is None:
            return tweaked
    return random.choice(templates)


def propose_via_llm(
    context: dict[str, Any],
    n: int = 3,
    llm_chat: Callable | None = None,
    llm_healthy: Callable | None = None,
) -> list[str]:
    """Ask LLM for factor expressions; return [] on failure."""
    if llm_chat is None or llm_healthy is None:
        try:
            import sys
            eo = str(ROOT / "eternal-optimizer")
            if eo not in sys.path:
                sys.path.insert(0, eo)
            from llm_client import llm_chat as _chat, llm_healthy as _healthy
            llm_chat, llm_healthy = _chat, _healthy
        except Exception as e:
            logger.debug("llm_client unavailable: %s", e)
            return []

    if not llm_healthy():
        return []

    prompt = f"""You are a quant researcher. Propose {n} single-asset alpha factor expressions.
Allowed tokens ONLY: open,high,low,close,volume,log,abs,sign,sqrt,maximum,minimum,
delta,delay,ts_mean,ts_std,ts_max,ts_min,ts_rank,corr,roc,vwap and numbers/operators.
Return JSON list of strings only, e.g. ["roc(close, 10)", "-corr(open, volume, 10)"].
Context: {json.dumps(context)[:800]}
Avoid duplicates of: {context.get('existing', [])[:15]}
"""
    raw = llm_chat(
        [
            {"role": "system", "content": "Output JSON array of factor expression strings only."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=800,
        temperature=0.8,
    )
    if not raw:
        return []
    try:
        # extract JSON array
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if not m:
            return []
        arr = json.loads(m.group(0))
        out = []
        for item in arr:
            if isinstance(item, str) and _validate_expr(item) is None:
                out.append(item.strip())
        return out[:n]
    except Exception as e:
        logger.warning("LLM parse failed: %s", e)
        return []


def score_candidate(
    expr: str,
    df: pd.DataFrame,
    existing_factors: pd.DataFrame | None = None,
    horizon: int = 1,
) -> FactorCandidate:
    from factor.analysis import deduplicate_factors, factor_ic, summarize_ic

    cand = FactorCandidate(expr=expr, source="eval")
    try:
        series = evaluate_expression(expr, df)
        if series.notna().sum() < 30:
            cand.error = "too many NaNs"
            return cand
        ic = factor_ic(series, df["close"], horizon=horizon)
        summary = summarize_ic(ic)
        cand.ic_mean = summary["ic_mean"]
        cand.ir = summary["ir"]
        # score: |IC| * (1 + clip(IR)) with sign preference for |IC|
        ic_m = abs(float(summary["ic_mean"] or 0))
        ir_v = float(summary["ir"] or 0)
        cand.score = ic_m * (1.0 + max(-0.5, min(1.5, abs(ir_v) * 0.2)))

        if existing_factors is not None and not existing_factors.empty:
            panel = existing_factors.copy()
            panel["__cand__"] = series
            dedupe = deduplicate_factors(panel.dropna(how="all"), threshold=0.99)
            if "__cand__" in dedupe.get("dropped", []):
                cand.dedupe_ok = False
                pair = next((p for p in dedupe.get("pairs", []) if p.get("dropped") == "__cand__"), None)
                cand.max_corr = pair.get("abs_corr") if pair else 0.99
                cand.score = (cand.score or 0) * 0.1  # heavy penalty
            else:
                # max corr vs existing
                corr = panel.corr(method="spearman")
                if "__cand__" in corr.columns:
                    others = corr["__cand__"].drop(labels=["__cand__"], errors="ignore").abs()
                    cand.max_corr = float(others.max()) if len(others) else None
        cand.meta = {"n_ic": summary["n"], "ic_pos_ratio": summary["ic_positive_ratio"]}
    except Exception as e:
        cand.error = str(e)
        cand.score = None
    return cand


def run_evolution_round(
    df: pd.DataFrame,
    *,
    n_proposals: int = 5,
    existing_exprs: list[str] | None = None,
    bandit: FactorBandit | None = None,
    use_llm: bool = True,
) -> dict[str, Any]:
    """One evolution round: bandit select → propose → score → update."""
    bandit = bandit or FactorBandit()
    arm = bandit.select()
    existing_exprs = existing_exprs or list(_SEED_EXPRS)

    # Build existing factor panel for dedupe
    existing_cols: dict[str, pd.Series] = {}
    for i, ex in enumerate(existing_exprs[:12]):
        try:
            existing_cols[f"e{i}"] = evaluate_expression(ex, df)
        except Exception:
            continue
    existing_df = pd.DataFrame(existing_cols) if existing_cols else None

    proposals: list[tuple[str, str]] = []
    if arm == "factor_mining":
        if use_llm:
            llm_exprs = propose_via_llm(
                {"existing": existing_exprs, "goal": "maximize |IC| vs forward return, diversify"},
                n=n_proposals,
            )
            proposals.extend((e, "llm") for e in llm_exprs)
        while len(proposals) < n_proposals:
            base = random.choice(existing_exprs) if existing_exprs else None
            proposals.append((mutate_expression(base), "mutate"))
    else:
        # model_tune arm: mutate windows of best seeds (proxy for model budget)
        for _ in range(n_proposals):
            base = random.choice(existing_exprs)
            proposals.append((mutate_expression(base), "mutate"))

    candidates: list[FactorCandidate] = []
    for expr, src in proposals:
        c = score_candidate(expr, df, existing_factors=existing_df)
        c.source = src
        candidates.append(c)

    valid = [c for c in candidates if c.score is not None and c.error is None]
    best = max(valid, key=lambda c: c.score or 0) if valid else None
    reward = float(best.score) if best and best.dedupe_ok else 0.0
    if best and not best.dedupe_ok:
        reward *= 0.2
    bandit.update(arm, reward)

    result = {
        "ts": time.time(),
        "arm": arm,
        "bandit": bandit.snapshot(),
        "candidates": [asdict(c) for c in candidates],
        "best": asdict(best) if best else None,
        "n_valid": len(valid),
    }
    out_path = OUT_DIR / f"round_{int(time.time())}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    latest = OUT_DIR / "latest.json"
    latest.write_text(json.dumps(result, indent=2, default=str))
    result["path"] = str(out_path.relative_to(ROOT))
    return result
