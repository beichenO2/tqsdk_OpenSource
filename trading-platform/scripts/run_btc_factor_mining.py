#!/usr/bin/env python3
"""BTC MCTS factor mining with strict OOS validation + dynamic combine + CS check.

Usage:
    PYTHONPATH=packages .venv/bin/python3 scripts/run_btc_factor_mining.py
    PYTHONPATH=packages .venv/bin/python3 scripts/run_btc_factor_mining.py --iterations 60
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ROOT / "packages"
if str(PACKAGES) not in sys.path:
    sys.path.insert(0, str(PACKAGES))

from datahub.crypto_loader import CryptoDataLoader  # noqa: E402
from factor.analysis import factor_ic, summarize_ic  # noqa: E402
from factor.alphalens_cs import analyze_cross_section  # noqa: E402
from factor.combine import compare_static_vs_dynamic  # noqa: E402
from factor.cs_pipeline import build_cs_panels_from_expr  # noqa: E402
from factor.evolution import evaluate_expression  # noqa: E402
from factor.mcts_search import run_mcts_search  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("btc-factor-mining")

OHLCV_COLS = ("datetime", "open", "high", "low", "close", "volume")
CS_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
OOS_IC_ABS_MIN = 0.015
MIN_VALID_SAMPLES = 500
MAX_COMBINE_FACTORS = 8
MAX_CS_FACTORS = 3


@dataclass
class OOSResult:
    expr: str
    train_ic: float | None
    train_ir: float | None
    test_ic: float | None
    test_ir: float | None
    n_valid: int
    status: str  # survived | failed_ic | failed_sign | insufficient | error
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "expr": self.expr,
            "train_ic": self.train_ic,
            "train_ir": self.train_ir,
            "test_ic": self.test_ic,
            "test_ir": self.test_ir,
            "n_valid": self.n_valid,
            "status": self.status,
            "error": self.error,
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BTC MCTS factor mining with OOS validation")
    p.add_argument("--iterations", type=int, default=120, help="MCTS iterations (default 120)")
    p.add_argument("--timeframe", type=str, default="1h", help="Bar timeframe")
    p.add_argument("--train-bars", type=int, default=30000, help="Train window size")
    p.add_argument("--test-bars", type=int, default=8000, help="OOS test window size")
    p.add_argument("--use-llm", action="store_true", default=False, help="Enable LLM proposals")
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--data-dir", type=str, default=None, help="Override CryptoDataLoader data_dir")
    p.add_argument("--out-dir", type=str, default=None, help="Output directory (default output/research)")
    p.add_argument("--min-valid", type=int, default=MIN_VALID_SAMPLES)
    p.add_argument("--skip-cs", action="store_true", default=False, help="Skip cross-section check")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def prepare_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize loader output to datetime/open/high/low/close/volume."""
    out = df.copy()
    if "datetime" not in out.columns and "open_time" in out.columns:
        out = out.rename(columns={"open_time": "datetime"})
    missing = [c for c in OHLCV_COLS if c not in out.columns]
    if missing:
        raise ValueError(f"OHLCV missing columns: {missing}")
    out = out[list(OHLCV_COLS)].copy()
    out["datetime"] = pd.to_datetime(out["datetime"], utc=True)
    out = out.sort_values("datetime").reset_index(drop=True)
    return out


def load_btc_ohlcv(
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    data_dir: str | None = None,
) -> pd.DataFrame:
    loader = CryptoDataLoader(data_dir)
    raw = loader.load(symbol, timeframe=timeframe)
    if raw.empty:
        raise FileNotFoundError(f"No data for {symbol} {timeframe} (data_dir={loader.data_dir})")
    return prepare_ohlcv(raw)


def split_train_test(
    df: pd.DataFrame,
    train_bars: int,
    test_bars: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tail split: earlier train_bars then later test_bars; no overlap; test after train."""
    need = train_bars + test_bars
    if len(df) < need:
        raise ValueError(f"Need >= {need} bars, got {len(df)}")
    tail = df.iloc[-need:].reset_index(drop=True)
    train = tail.iloc[:train_bars].reset_index(drop=True)
    test = tail.iloc[train_bars:].reset_index(drop=True)
    assert len(train) == train_bars and len(test) == test_bars
    assert train["datetime"].iloc[-1] < test["datetime"].iloc[0]
    return train, test


def _unique_candidates(elite: list[dict], qualified: list[dict]) -> list[dict]:
    """Merge elite + qualified by expr (elite first)."""
    seen: set[str] = set()
    out: list[dict] = []
    for bucket in (elite, qualified):
        for c in bucket or []:
            expr = str(c.get("expr") or "").strip()
            if not expr or expr in seen:
                continue
            seen.add(expr)
            out.append(c)
    return out


def score_oos(
    expr: str,
    train_ic: float | None,
    train_ir: float | None,
    test_df: pd.DataFrame,
    *,
    horizon: int = 1,
    min_valid: int = MIN_VALID_SAMPLES,
) -> OOSResult:
    """Re-evaluate on test with the same IC/IR pipeline as evolution.score_candidate."""
    try:
        series = evaluate_expression(expr, test_df)
        aligned = pd.concat(
            [
                series.rename("f"),
                (test_df["close"].shift(-horizon) / test_df["close"] - 1.0).rename("r"),
            ],
            axis=1,
        ).dropna()
        n_valid = int(len(aligned))
        if n_valid < min_valid:
            return OOSResult(
                expr=expr,
                train_ic=train_ic,
                train_ir=train_ir,
                test_ic=None,
                test_ir=None,
                n_valid=n_valid,
                status="insufficient",
            )
        ic = factor_ic(series, test_df["close"], horizon=horizon)
        summary = summarize_ic(ic)
        test_ic = summary["ic_mean"]
        test_ir = summary["ir"]
        if test_ic is None:
            return OOSResult(
                expr=expr,
                train_ic=train_ic,
                train_ir=train_ir,
                test_ic=None,
                test_ir=None,
                n_valid=n_valid,
                status="insufficient",
            )
        test_ic_f = float(test_ic)
        if abs(test_ic_f) < OOS_IC_ABS_MIN:
            status = "failed_ic"
        elif train_ic is None or train_ic == 0 or np.sign(test_ic_f) != np.sign(float(train_ic)):
            status = "failed_sign"
        else:
            status = "survived"
        return OOSResult(
            expr=expr,
            train_ic=float(train_ic) if train_ic is not None else None,
            train_ir=float(train_ir) if train_ir is not None else None,
            test_ic=test_ic_f,
            test_ir=float(test_ir) if test_ir is not None else None,
            n_valid=n_valid,
            status=status,
        )
    except Exception as e:
        return OOSResult(
            expr=expr,
            train_ic=float(train_ic) if train_ic is not None else None,
            train_ir=float(train_ir) if train_ir is not None else None,
            test_ic=None,
            test_ir=None,
            n_valid=0,
            status="error",
            error=str(e),
        )


def build_factor_df(exprs: list[str], df: pd.DataFrame) -> pd.DataFrame:
    cols: dict[str, pd.Series] = {}
    for i, expr in enumerate(exprs):
        try:
            cols[f"f{i}"] = evaluate_expression(expr, df)
        except Exception as e:
            logger.warning("factor eval failed %s: %s", expr, e)
    return pd.DataFrame(cols, index=df.index)


def run_cs_for_expr(
    expr: str,
    *,
    symbols: list[str] | None = None,
    timeframe: str = "4h",
    limit: int = 8000,
    data_dir: str | None = None,
) -> dict[str, Any]:
    symbols = symbols or CS_SYMBOLS
    try:
        fpanel, cpanel = build_cs_panels_from_expr(
            expr, symbols, timeframe=timeframe, limit=limit, data_dir=data_dir
        )
        result = analyze_cross_section(fpanel, cpanel, horizon=1)
        result["expr"] = expr
        result["symbols_used"] = list(fpanel.columns)
        return result
    except Exception as e:
        return {"expr": expr, "error": str(e), "summary": None}


def _fmt_ic(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v:.4f}"


def write_report(
    path: Path,
    *,
    params: dict[str, Any],
    split_info: dict[str, Any],
    mcts: dict[str, Any],
    oos_rows: list[OOSResult],
    combine: dict[str, Any] | None,
    cs_results: list[dict[str, Any]],
    conclusions: list[str],
) -> None:
    elite = mcts.get("elite") or []
    qualified = mcts.get("qualified") or []
    survived = [r for r in oos_rows if r.status == "survived"]
    rate = (len(survived) / len(oos_rows)) if oos_rows else 0.0

    lines: list[str] = [
        "# BTC Factor Mining Report (OOS)",
        "",
        f"Generated: `{params['ts']}`",
        "",
        "## Parameters",
        "",
        f"- symbol: `{params['symbol']}`",
        f"- timeframe: `{params['timeframe']}`",
        f"- MCTS iterations: `{params['iterations']}`",
        f"- use_llm: `{params['use_llm']}`",
        f"- train_bars: `{params['train_bars']}`",
        f"- test_bars: `{params['test_bars']}`",
        f"- OOS |IC| threshold: `{OOS_IC_ABS_MIN}`",
        f"- min valid samples: `{params['min_valid']}`",
        "",
        "## Train / Test Split",
        "",
        f"- full bars loaded: `{split_info['n_full']}`",
        f"- train: `{split_info['train_start']}` → `{split_info['train_end']}` ({split_info['n_train']} bars)",
        f"- test: `{split_info['test_start']}` → `{split_info['test_end']}` ({split_info['n_test']} bars)",
        "",
        "## MCTS (train)",
        "",
        f"- candidates: `{len(mcts.get('candidates') or [])}`",
        f"- elite: `{len(elite)}`",
        f"- qualified: `{len(qualified)}`",
        f"- tree_stats: `{json.dumps(mcts.get('tree_stats') or {}, default=str)}`",
        "",
        "## OOS Survival",
        "",
        f"- evaluated (elite∪qualified): `{len(oos_rows)}`",
        f"- survived: `{len(survived)}`",
        f"- survival rate: `{rate:.1%}`",
        "",
        "| expr | train IC | train IR | test IC | test IR | n_valid | status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in sorted(
        oos_rows,
        key=lambda x: (x.status != "survived", -(abs(x.test_ic or 0))),
    ):
        lines.append(
            f"| `{r.expr}` | {_fmt_ic(r.train_ic)} | {_fmt_ic(r.train_ir)} | "
            f"{_fmt_ic(r.test_ic)} | {_fmt_ic(r.test_ir)} | {r.n_valid} | {r.status} |"
        )

    lines += ["", "## Dynamic vs Static Combine (OOS survivors, top ≤8)", ""]
    if combine and combine.get("static") is not None:
        lines.append(f"- n_factors: `{combine.get('n_factors')}`")
        lines.append(
            f"- static IC/IR: `{_fmt_ic(combine['static'].get('ic'))}` / "
            f"`{_fmt_ic(combine['static'].get('ir'))}`"
        )
        lines.append(
            f"- dynamic IC/IR: `{_fmt_ic(combine['dynamic'].get('ic'))}` / "
            f"`{_fmt_ic(combine['dynamic'].get('ir'))}`"
        )
        lines.append(f"- weights_last: `{json.dumps(combine.get('weights_last') or {}, default=str)}`")
        if combine.get("exprs"):
            lines.append("- factors used:")
            for e in combine["exprs"]:
                lines.append(f"  - `{e}`")
    else:
        lines.append("_No OOS survivors — combine skipped._")

    lines += ["", "## Cross-Section (top ≤3 survivors, 4h, BTC/ETH/SOL/BNB)", ""]
    if not cs_results:
        lines.append("_Skipped or no survivors._")
    else:
        for cs in cs_results:
            if cs.get("error"):
                lines.append(f"- `{cs.get('expr')}`: ERROR `{cs['error']}`")
                continue
            s = cs.get("summary") or {}
            lines.append(
                f"- `{cs.get('expr')}`: CS IC=`{_fmt_ic(s.get('ic_mean'))}` "
                f"IR=`{_fmt_ic(s.get('ir'))}` n=`{s.get('n')}` "
                f"symbols=`{cs.get('symbols_used')}`"
            )

    lines += ["", "## Conclusions", ""]
    for c in conclusions:
        lines.append(f"- {c}")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_surviving_json(
    path: Path,
    *,
    ts: str,
    timeframe: str,
    survived: list[OOSResult],
) -> None:
    payload = {
        "ts": ts,
        "timeframe": timeframe,
        "factors": [
            {
                "expr": r.expr,
                "train_ic": r.train_ic,
                "test_ic": r.test_ic,
                "test_ir": r.test_ir,
            }
            for r in survived
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def build_conclusions(
    oos_rows: list[OOSResult],
    combine: dict[str, Any] | None,
    cs_results: list[dict[str, Any]],
) -> list[str]:
    survived = [r for r in oos_rows if r.status == "survived"]
    out: list[str] = []
    if not survived:
        out.append(
            "No factors survived OOS (|IC|≥0.015 with matching sign). "
            "Do not promote train-only elites into live strategies."
        )
        return out

    ranked = sorted(survived, key=lambda r: abs(r.test_ic or 0), reverse=True)
    out.append(
        f"{len(survived)} factor(s) survived OOS. "
        f"Strongest by |test IC|: `{ranked[0].expr}` "
        f"(train IC={_fmt_ic(ranked[0].train_ic)}, test IC={_fmt_ic(ranked[0].test_ic)})."
    )
    if combine and combine.get("dynamic") and combine.get("static"):
        d_ic = combine["dynamic"].get("ic")
        s_ic = combine["static"].get("ic")
        if d_ic is not None and s_ic is not None:
            if abs(float(d_ic)) > abs(float(s_ic)):
                out.append(
                    f"Dynamic combine beats static on OOS (|IC| {_fmt_ic(d_ic)} vs {_fmt_ic(s_ic)})."
                )
            else:
                out.append(
                    f"Static combine ≥ dynamic on OOS (|IC| {_fmt_ic(s_ic)} vs {_fmt_ic(d_ic)})."
                )
    cs_ok = [
        c for c in cs_results
        if c.get("summary") and c["summary"].get("ic_mean") is not None
        and abs(float(c["summary"]["ic_mean"])) >= 0.01
    ]
    if cs_ok:
        out.append(
            "Cross-section support for: "
            + ", ".join(f"`{c['expr']}` (CS IC={_fmt_ic(c['summary']['ic_mean'])})" for c in cs_ok)
            + ". Prefer these for multi-asset strategies."
        )
    elif cs_results:
        out.append(
            "Survivors lack meaningful cross-sectional IC on BTC/ETH/SOL/BNB 4h — "
            "treat as single-asset signals only."
        )
    worth = [
        r for r in ranked
        if abs(r.test_ic or 0) >= 0.02
    ] or ranked[:3]
    out.append(
        "Candidates worth strategy integration: "
        + ", ".join(f"`{r.expr}`" for r in worth)
        + "."
    )
    return out


def run_pipeline(
    df: pd.DataFrame,
    *,
    iterations: int = 120,
    train_bars: int = 30000,
    test_bars: int = 8000,
    use_llm: bool = False,
    timeframe: str = "1h",
    symbol: str = "BTCUSDT",
    min_valid: int = MIN_VALID_SAMPLES,
    skip_cs: bool = False,
    data_dir: str | None = None,
    out_dir: Path | None = None,
    ts: str | None = None,
) -> dict[str, Any]:
    """Full mining pipeline. Returns paths + summary metrics."""
    out_dir = out_dir or (ROOT / "output" / "research")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = ts or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")

    df = prepare_ohlcv(df) if "datetime" in df.columns or "open_time" in df.columns else df
    train, test = split_train_test(df, train_bars, test_bars)
    split_info = {
        "n_full": len(df),
        "n_train": len(train),
        "n_test": len(test),
        "train_start": str(train["datetime"].iloc[0]),
        "train_end": str(train["datetime"].iloc[-1]),
        "test_start": str(test["datetime"].iloc[0]),
        "test_end": str(test["datetime"].iloc[-1]),
    }
    logger.info(
        "Split train %s→%s (%d) | test %s→%s (%d)",
        split_info["train_start"],
        split_info["train_end"],
        split_info["n_train"],
        split_info["test_start"],
        split_info["test_end"],
        split_info["n_test"],
    )

    logger.info("Running MCTS on train (%d iterations, use_llm=%s)...", iterations, use_llm)
    mcts = run_mcts_search(train, n_iterations=iterations, use_llm=use_llm)
    elite = mcts.get("elite") or []
    qualified = mcts.get("qualified") or []
    candidates = _unique_candidates(elite, qualified)
    logger.info(
        "MCTS done: candidates=%d elite=%d qualified=%d unique_eval=%d",
        len(mcts.get("candidates") or []),
        len(elite),
        len(qualified),
        len(candidates),
    )

    oos_rows: list[OOSResult] = []
    for c in candidates:
        oos_rows.append(
            score_oos(
                str(c["expr"]),
                c.get("ic_mean"),
                c.get("ir"),
                test,
                min_valid=min_valid,
            )
        )
    survived = [r for r in oos_rows if r.status == "survived"]
    survived_ranked = sorted(survived, key=lambda r: abs(r.test_ic or 0), reverse=True)
    logger.info(
        "OOS: evaluated=%d survived=%d rate=%.1f%%",
        len(oos_rows),
        len(survived),
        100.0 * len(survived) / len(oos_rows) if oos_rows else 0.0,
    )

    combine: dict[str, Any] | None = None
    top_combine = survived_ranked[:MAX_COMBINE_FACTORS]
    if top_combine:
        exprs = [r.expr for r in top_combine]
        factor_df = build_factor_df(exprs, test)
        if not factor_df.empty and factor_df.shape[1] >= 1:
            cmp = compare_static_vs_dynamic(factor_df, test["close"], horizon=1)
            combine = {
                "n_factors": int(factor_df.shape[1]),
                "exprs": exprs,
                **cmp,
            }
            logger.info(
                "Combine OOS static=%s dynamic=%s",
                cmp.get("static"),
                cmp.get("dynamic"),
            )

    cs_results: list[dict[str, Any]] = []
    if not skip_cs:
        for r in survived_ranked[:MAX_CS_FACTORS]:
            logger.info("CS check: %s", r.expr)
            cs_results.append(run_cs_for_expr(r.expr, data_dir=data_dir))

    conclusions = build_conclusions(oos_rows, combine, cs_results)
    params = {
        "ts": ts,
        "symbol": symbol,
        "timeframe": timeframe,
        "iterations": iterations,
        "use_llm": use_llm,
        "train_bars": train_bars,
        "test_bars": test_bars,
        "min_valid": min_valid,
    }

    report_path = out_dir / f"btc_mining_report_{ts}.md"
    json_path = out_dir / "btc_surviving_factors.json"
    write_report(
        report_path,
        params=params,
        split_info=split_info,
        mcts=mcts,
        oos_rows=oos_rows,
        combine=combine,
        cs_results=cs_results,
        conclusions=conclusions,
    )
    write_surviving_json(json_path, ts=ts, timeframe=timeframe, survived=survived_ranked)
    logger.info("Report → %s", report_path)
    logger.info("Survivors JSON → %s", json_path)

    return {
        "report_path": str(report_path),
        "json_path": str(json_path),
        "n_elite": len(elite),
        "n_qualified": len(qualified),
        "n_oos_evaluated": len(oos_rows),
        "n_survived": len(survived),
        "oos_rows": [r.to_dict() for r in oos_rows],
        "combine": combine,
        "cs_results": cs_results,
        "conclusions": conclusions,
        "split_info": split_info,
        "params": params,
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "output" / "research")
    logger.info("Loading %s %s ...", args.symbol, args.timeframe)
    df = load_btc_ohlcv(args.symbol, args.timeframe, data_dir=args.data_dir)
    logger.info("Loaded %d bars", len(df))
    return run_pipeline(
        df,
        iterations=args.iterations,
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        use_llm=args.use_llm,
        timeframe=args.timeframe,
        symbol=args.symbol,
        min_valid=args.min_valid,
        skip_cs=args.skip_cs,
        data_dir=args.data_dir,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    result = main()
    print(
        json.dumps(
            {
                "report_path": result["report_path"],
                "json_path": result["json_path"],
                "n_elite": result["n_elite"],
                "n_qualified": result["n_qualified"],
                "n_survived": result["n_survived"],
            },
            indent=2,
        )
    )
