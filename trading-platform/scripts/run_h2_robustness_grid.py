#!/usr/bin/env python3
"""H2 vol_adj_momentum parameter-neighborhood robustness grid.

Grid: momentum/vol window {5,7,10} calendar days × band dead-zone {0.2,0.3,0.4}
on 1d and 4h timeframes (z-score window fixed at 180d, 5bp one-way cost).

Usage:
    .venv/bin/python scripts/run_h2_robustness_grid.py
    .venv/bin/python scripts/run_h2_robustness_grid.py --symbol BTCUSDT
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

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ROOT / "packages"
if str(PACKAGES) not in sys.path:
    sys.path.insert(0, str(PACKAGES))

from datahub.crypto_loader import CryptoDataLoader, DEFAULT_DATA_DIR  # noqa: E402
from factor.hypothesis_signals import (  # noqa: E402
    TF_CONFIG,
    apply_suppression_band,
    evaluate_position_gate,
    factor_h2_vol_adj_momentum,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("h2-robustness-grid")

MOMENTUM_DAYS_GRID = (5, 7, 10)
BAND_GRID = (0.2, 0.3, 0.4)
TIMEFRAMES = ("1d", "4h")
CENTER_MOM_DAYS = 7
CENTER_BAND = 0.3
ZSCORE_DAYS = 180
DEFAULT_COST_BPS = 5.0


def assess_timeframe_robustness(
    rows: list[dict[str, Any]],
    *,
    center_mom_days: int = CENTER_MOM_DAYS,
    center_band: float = CENTER_BAND,
    min_positive_frac: float = 7 / 9,
    peak_ratio_cap: float = 3.0,
) -> dict[str, Any]:
    """Return robustness verdict for one timeframe's 9-point grid."""
    if not rows:
        return {
            "passed": False,
            "positive_count": 0,
            "positive_fraction": 0.0,
            "center_return": float("nan"),
            "neighbor_median_return": float("nan"),
            "peak_ratio": float("nan"),
            "reason": "no rows",
        }

    positive_count = sum(1 for r in rows if r.get("net_return_5bp", 0) > 0)
    positive_fraction = positive_count / len(rows)
    center = next(
        (
            r for r in rows
            if r["momentum_days"] == center_mom_days and r["band"] == center_band
        ),
        None,
    )
    neighbors = [
        r for r in rows
        if not (r["momentum_days"] == center_mom_days and r["band"] == center_band)
    ]
    center_return = float(center["net_return_5bp"]) if center else float("nan")
    neighbor_returns = [float(r["net_return_5bp"]) for r in neighbors]
    neighbor_median = float(np.median(neighbor_returns)) if neighbor_returns else float("nan")
    if neighbor_median > 0:
        peak_ratio = center_return / neighbor_median
    elif neighbor_median == 0:
        peak_ratio = float("inf") if center_return > 0 else 0.0
    else:
        peak_ratio = 0.0 if center_return <= 0 else float("inf")

    positive_ok = positive_fraction >= min_positive_frac
    peak_ok = center_return <= peak_ratio_cap * neighbor_median if neighbor_median > 0 else center_return <= 0
    passed = positive_ok and peak_ok

    reasons: list[str] = []
    if not positive_ok:
        reasons.append(
            f"only {positive_count}/{len(rows)} points positive "
            f"(need ≥{int(np.ceil(min_positive_frac * len(rows)))})"
        )
    if not peak_ok:
        reasons.append(
            f"center return {center_return:.4f} > {peak_ratio_cap}× "
            f"neighbor median {neighbor_median:.4f}"
        )

    return {
        "passed": passed,
        "positive_count": positive_count,
        "positive_fraction": positive_fraction,
        "center_return": center_return,
        "neighbor_median_return": neighbor_median,
        "peak_ratio": peak_ratio,
        "reason": "; ".join(reasons) if reasons else "ok",
    }


def run_robustness_grid(
    loader: CryptoDataLoader,
    symbol: str,
    cost_bps: float = DEFAULT_COST_BPS,
    timeframes: tuple[str, ...] = TIMEFRAMES,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    robustness: dict[str, dict[str, Any]] = {}

    for tf in timeframes:
        ohlcv = loader.load(symbol, tf)
        if ohlcv.empty:
            logger.warning("Empty OHLCV for %s %s — skip", symbol, tf)
            continue
        cfg = TF_CONFIG[tf]
        tf_rows: list[dict[str, Any]] = []
        for mom_days in MOMENTUM_DAYS_GRID:
            for band in BAND_GRID:
                target = factor_h2_vol_adj_momentum(
                    ohlcv, cfg, momentum_days=mom_days, zscore_days=ZSCORE_DAYS
                )
                position = apply_suppression_band(target, ohlcv, band=band)
                metrics = evaluate_position_gate(
                    position, ohlcv, cfg["bars_per_year"], cost_bps
                )
                row = {
                    "timeframe": tf,
                    "momentum_days": mom_days,
                    "band": band,
                    "n_bars": len(ohlcv),
                    **metrics,
                }
                rows.append(row)
                tf_rows.append(row)
                logger.info(
                    "%s mom=%dd band=%.1f → net@5bp=%.4f sharpe=%.2f gates=%d/5 %s",
                    tf, mom_days, band,
                    metrics["net_return_5bp"], metrics["net_sharpe"],
                    metrics["gates_passed"], metrics["verdict"],
                )
        robustness[tf] = assess_timeframe_robustness(tf_rows)
    return rows, robustness


def render_report_md(
    rows: list[dict[str, Any]],
    robustness: dict[str, dict[str, Any]],
    symbol: str,
    cost_bps: float,
    ts: str,
) -> str:
    lines = [
        "# H2 Vol-Adj Momentum Robustness Grid",
        "",
        f"- **Symbol:** {symbol}",
        f"- **Generated:** {ts}",
        f"- **Factor:** H2 vol_adj_momentum",
        f"- **Z-score window:** {ZSCORE_DAYS} calendar days (fixed)",
        f"- **One-way cost:** {cost_bps} bp",
        f"- **Grid:** momentum/vol {list(MOMENTUM_DAYS_GRID)}d × band {list(BAND_GRID)} "
        f"= {len(MOMENTUM_DAYS_GRID) * len(BAND_GRID)} points per timeframe",
        f"- **Timeframes:** {', '.join(TIMEFRAMES)}",
        "",
    ]

    for tf in TIMEFRAMES:
        tf_rows = [r for r in rows if r["timeframe"] == tf]
        if not tf_rows:
            continue
        lines.extend([
            f"## {tf} grid",
            "",
            "| Mom (d) | Band | Net@5bp | Sharpe | Turnover | E[bp]/trade | Gates | Verdict |",
            "|---------|------|---------|--------|----------|-------------|-------|---------|",
        ])
        for r in sorted(tf_rows, key=lambda x: (x["momentum_days"], x["band"])):
            lines.append(
                f"| {r['momentum_days']} | {r['band']:.1f} | "
                f"{r['net_return_5bp']:.4f} | {r['net_sharpe']:.2f} | "
                f"{r['annual_turnover']:.1f} | {r['trade_expectancy_bp']:.1f} | "
                f"{r['gates_passed']}/5 | {r['verdict']} |"
            )
        rb = robustness.get(tf, {})
        verdict = "ROBUST" if rb.get("passed") else "NOT ROBUST"
        lines.extend([
            "",
            f"### {tf} robustness verdict: **{verdict}**",
            "",
            f"- Positive @5bp: {rb.get('positive_count', 0)}/9 "
            f"(need ≥7)",
            f"- Center ({CENTER_MOM_DAYS}d, band {CENTER_BAND}): "
            f"net {rb.get('center_return', float('nan')):.4f}",
            f"- Neighbor median net: {rb.get('neighbor_median_return', float('nan')):.4f}",
            f"- Center / neighbor median: {rb.get('peak_ratio', float('nan')):.2f} "
            f"(cap ≤ {3.0}×)",
            f"- Detail: {rb.get('reason', '')}",
            "",
        ])

    lines.extend(["## Summary", ""])
    for tf in TIMEFRAMES:
        rb = robustness.get(tf)
        if rb is None:
            lines.append(f"- **{tf}:** no data")
            continue
        label = "ROBUST" if rb["passed"] else "NOT ROBUST"
        lines.append(
            f"- **{tf}:** {label} — {rb['positive_count']}/9 positive, "
            f"center/neighbor ratio {rb['peak_ratio']:.2f}"
        )
    return "\n".join(lines)


def run_pipeline(
    symbol: str = "BTCUSDT",
    data_dir: str | Path | None = None,
    cost_bps: float = DEFAULT_COST_BPS,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    data_path = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    out_path = Path(out_dir) if out_dir else ROOT / "output" / "research"
    out_path.mkdir(parents=True, exist_ok=True)

    loader = CryptoDataLoader(data_path)
    rows, robustness = run_robustness_grid(loader, symbol, cost_bps)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    md_path = out_path / f"h2_robustness_{ts}.md"
    json_path = out_path / f"h2_robustness_{ts}.json"

    payload = {
        "ts": ts,
        "symbol": symbol,
        "cost_bps": cost_bps,
        "data_dir": str(data_path),
        "momentum_days_grid": list(MOMENTUM_DAYS_GRID),
        "band_grid": list(BAND_GRID),
        "timeframes": list(TIMEFRAMES),
        "zscore_days": ZSCORE_DAYS,
        "rows": rows,
        "robustness": robustness,
    }
    md_path.write_text(
        render_report_md(rows, robustness, symbol, cost_bps, ts),
        encoding="utf-8",
    )
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s and %s", md_path, json_path)

    return {
        "report_path": str(md_path),
        "json_path": str(json_path),
        "rows": rows,
        "robustness": robustness,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="H2 parameter neighborhood robustness grid")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--data-dir", default=None, help=f"Default: {DEFAULT_DATA_DIR}")
    p.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
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
