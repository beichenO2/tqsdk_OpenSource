#!/usr/bin/env python3
"""Holding-period × turnover-suppression sweep for H2/H4 factors.

Validates whether longer bar intervals rescue signals killed by trading costs
at 4h (high turnover under 5bp one-way cost).

Usage:
    .venv/bin/python scripts/run_holding_period_sweep.py
    .venv/bin/python scripts/run_holding_period_sweep.py --symbol BTCUSDT
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ROOT / "packages"
if str(PACKAGES) not in sys.path:
    sys.path.insert(0, str(PACKAGES))

from datahub.crypto_loader import CryptoDataLoader, DEFAULT_DATA_DIR  # noqa: E402
from factor.hypothesis_signals import (  # noqa: E402
    TF_CONFIG,
    apply_suppression,
    evaluate_position_gate,
    factor_h2_vol_adj_momentum,
    factor_h4_short_high_momentum,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("holding-period-sweep")

CANDIDATE_TIMEFRAMES = ("15m", "1h", "4h", "1d")
FACTORS = ("H2_vol_adj_momentum", "H4_short_high_momentum")
SUPPRESSIONS = ("none", "band_0.3", "min_hold_1d")


def discover_timeframes(loader: CryptoDataLoader, symbol: str) -> list[str]:
    """Return candidate timeframes that have parquet data for *symbol*."""
    avail = set(loader.available_timeframes(symbol))
    found = [tf for tf in CANDIDATE_TIMEFRAMES if tf in avail]
    missing = [tf for tf in CANDIDATE_TIMEFRAMES if tf not in avail]
    for tf in missing:
        logger.info("Skipping %s — no data for %s", tf, symbol)
    return found


def build_target_position(
    ohlcv: pd.DataFrame, factor: str, timeframe: str
) -> pd.Series:
    cfg = TF_CONFIG[timeframe]
    if factor == "H2_vol_adj_momentum":
        return factor_h2_vol_adj_momentum(ohlcv, cfg)
    if factor == "H4_short_high_momentum":
        return factor_h4_short_high_momentum(ohlcv, cfg)
    raise ValueError(f"Unknown factor: {factor}")


def evaluate_combo(
    position: pd.Series,
    ohlcv: pd.DataFrame,
    bars_per_year: int,
    cost_bps: float = 5.0,
) -> dict[str, Any]:
    return evaluate_position_gate(position, ohlcv, bars_per_year, cost_bps)


def run_sweep_matrix(
    loader: CryptoDataLoader,
    symbol: str,
    cost_bps: float = 5.0,
    timeframes: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    if timeframes is None:
        timeframes = discover_timeframes(loader, symbol)
    rows: list[dict[str, Any]] = []

    for tf in timeframes:
        ohlcv = loader.load(symbol, tf)
        if ohlcv.empty:
            logger.warning("Empty OHLCV for %s %s — skip", symbol, tf)
            continue
        cfg = TF_CONFIG[tf]
        for factor in FACTORS:
            target = build_target_position(ohlcv, factor, tf)
            for suppression in SUPPRESSIONS:
                position = apply_suppression(target, ohlcv, suppression)
                metrics = evaluate_combo(position, ohlcv, cfg["bars_per_year"], cost_bps)
                rows.append({
                    "timeframe": tf,
                    "factor": factor,
                    "suppression": suppression,
                    "n_bars": len(ohlcv),
                    **metrics,
                })
                logger.info(
                    "%s %s %s → net@5bp=%.4f turnover=%.1f verdict=%s",
                    tf, factor, suppression,
                    metrics["net_return_5bp"], metrics["annual_turnover"], metrics["verdict"],
                )
    return rows, timeframes


def _auto_conclusions(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return [
            "1. 无可用数据，无法比较单笔期望。",
            "2. 无可用数据，无法评估换手抑制边际贡献。",
            "3. 无可用数据，5bp 下无正净收益组合。",
        ]

    best_exp = max(rows, key=lambda r: r.get("trade_expectancy_bp", float("-inf")))
    c1 = (
        f"1. **单笔期望最高**：{best_exp['timeframe']} × {best_exp['factor']} × "
        f"{best_exp['suppression']}，期望 {best_exp['trade_expectancy_bp']:.2f} bp/笔。"
    )

    # Marginal: mean delta (suppressed - none) per factor×timeframe
    deltas: list[float] = []
    keys = {(r["timeframe"], r["factor"]) for r in rows}
    for tf, fac in keys:
        none_r = next(
            (r for r in rows if r["timeframe"] == tf and r["factor"] == fac and r["suppression"] == "none"),
            None,
        )
        if none_r is None:
            continue
        for sup in ("band_0.3", "min_hold_1d"):
            sup_r = next(
                (r for r in rows if r["timeframe"] == tf and r["factor"] == fac and r["suppression"] == sup),
                None,
            )
            if sup_r is not None:
                deltas.append(sup_r["net_return_5bp"] - none_r["net_return_5bp"])
    if deltas:
        avg_delta = float(np.mean(deltas))
        sign = "提升" if avg_delta >= 0 else "损害"
        c2 = (
            f"2. **换手抑制边际贡献**：band_0.3 / min_hold_1d 相对 none 的平均净收益变化 "
            f"{avg_delta:+.4f}（{sign} {abs(avg_delta):.4f}）。"
        )
    else:
        c2 = "2. **换手抑制边际贡献**：数据不足，无法计算。"

    positive = [r for r in rows if r.get("net_return_5bp", 0) > 0]
    if positive:
        names = ", ".join(
            f"{r['timeframe']}/{r['factor']}/{r['suppression']}" for r in positive[:5]
        )
        extra = f" 等共 {len(positive)} 个" if len(positive) > 5 else ""
        c3 = f"3. **5bp 正净收益**：存在 {len(positive)} 个组合（{names}{extra}）。"
    else:
        c3 = "3. **5bp 正净收益**：所有组合净收益 ≤ 0，假设未获验证。"

    return [c1, c2, c3]


def render_report_md(
    rows: list[dict[str, Any]],
    symbol: str,
    cost_bps: float,
    available_tfs: list[str],
    ts: str,
) -> str:
    lines = [
        "# Holding Period × Turnover Suppression Sweep",
        "",
        f"- **Symbol:** {symbol}",
        f"- **Generated:** {ts}",
        f"- **Default one-way cost:** {cost_bps} bp",
        f"- **Available timeframes:** {', '.join(available_tfs) or '(none)'}",
        f"- **Matrix size:** {len(rows)} rows ({len(available_tfs)} TF × 2 factors × 3 suppressions)",
        "",
        "## Results Matrix",
        "",
        "| Timeframe | Factor | Suppression | Net@5bp | Net@2bp | Sharpe | Ann.Turnover | E[bp]/trade | Cost Ratio | Gates | Verdict |",
        "|-----------|--------|-------------|---------|---------|--------|--------------|-------------|------------|-------|---------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['timeframe']} | {r['factor']} | {r['suppression']} | "
            f"{r['net_return_5bp']:.4f} | {r['net_return_2bp']:.4f} | "
            f"{r['net_sharpe']:.2f} | {r['annual_turnover']:.1f} | "
            f"{r['trade_expectancy_bp']:.1f} | {r['cost_ratio']:.4f} | "
            f"{r['gates_passed']}/5 | {r['verdict']} |"
        )

    lines.extend(["", "## Auto Conclusions", ""])
    for c in _auto_conclusions(rows):
        lines.append(c)
    return "\n".join(lines)


def run_pipeline(
    symbol: str = "BTCUSDT",
    data_dir: str | Path | None = None,
    cost_bps: float = 5.0,
    out_dir: str | Path | None = None,
    timeframes: list[str] | None = None,
) -> dict[str, Any]:
    data_path = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    out_path = Path(out_dir) if out_dir else ROOT / "output" / "research"
    out_path.mkdir(parents=True, exist_ok=True)

    loader = CryptoDataLoader(data_path)
    avail = discover_timeframes(loader, symbol) if timeframes is None else timeframes
    rows, used_tfs = run_sweep_matrix(loader, symbol, cost_bps, avail)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    md_path = out_path / f"holding_period_sweep_{ts}.md"
    json_path = out_path / f"holding_period_sweep_{ts}.json"

    payload = {
        "ts": ts,
        "symbol": symbol,
        "cost_bps": cost_bps,
        "data_dir": str(data_path),
        "available_timeframes": used_tfs,
        "rows": rows,
        "conclusions": _auto_conclusions(rows),
    }
    md_path.write_text(
        render_report_md(rows, symbol, cost_bps, used_tfs, ts),
        encoding="utf-8",
    )
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s and %s", md_path, json_path)

    return {
        "report_path": str(md_path),
        "json_path": str(json_path),
        "rows": rows,
        "available_timeframes": used_tfs,
        "conclusions": payload["conclusions"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Holding period × turnover suppression sweep")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--data-dir", default=None, help=f"Default: {DEFAULT_DATA_DIR}")
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--out-dir", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    return run_pipeline(
        symbol=args.symbol,
        data_dir=args.data_dir,
        cost_bps=args.cost_bps,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
