#!/usr/bin/env python3
"""2号位工具：扫描 results/ 生成 leaderboard.md + 标记异常。

用法:
  cd ~/Polarisor/tqsdk-gnhf-worktrees/pos2/trading-platform
  python scripts/validate_scan_leaderboard.py
  python scripts/validate_scan_leaderboard.py --output-leaderboard ../.coordination/leaderboard.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scan-leaderboard")

PROJ_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJ_DIR / "results"
COORDINATION_DIR = PROJ_DIR / ".coordination"

ANOMALY_THRESHOLDS = {
    "return_absurd": 10.0,
    "sharpe_absurd": 50.0,
    "trades_min_meaningful": 5,
}


def scan_latest_results() -> list[dict]:
    """Scan *_latest.json files for strategy results."""
    entries = []
    for f in sorted(RESULTS_DIR.glob("*_latest.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skip %s: %s", f.name, e)
            continue

        name = data.get("name", f.stem.replace("_latest", ""))
        symbols = data.get("symbols", {})

        if isinstance(symbols, dict):
            for sym, metrics in symbols.items():
                entries.append(_extract_entry(name, sym, metrics, f.name))
        elif isinstance(symbols, list):
            for sym in symbols:
                entries.append({"name": name, "symbol": sym, "source": f.name,
                                "sharpe": 0, "total_return": 0, "max_dd": 0, "trades": 0,
                                "anomalies": ["no per-symbol metrics"]})

    return entries


def scan_backtest_reports() -> list[dict]:
    """Scan *_report.json files for batch backtest results."""
    entries = []
    for f in sorted(RESULTS_DIR.glob("*_report.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skip %s: %s", f.name, e)
            continue

        results = data.get("results", [])
        for r in results:
            if "error" in r:
                continue
            name = r.get("strategy", "?")
            sym = r.get("symbol", "?")
            entries.append(_extract_entry(name, sym, r, f.name))

    return entries


def _extract_entry(name: str, symbol: str, metrics: dict, source: str) -> dict:
    sharpe = metrics.get("sharpe", 0)
    total_return = metrics.get("total_return", 0)
    max_dd = metrics.get("max_dd", 0)
    trades = metrics.get("trades", 0)

    anomalies = []
    if abs(total_return) > ANOMALY_THRESHOLDS["return_absurd"]:
        anomalies.append(f"return={total_return:.2e} (absurd)")
    if abs(sharpe) > ANOMALY_THRESHOLDS["sharpe_absurd"]:
        anomalies.append(f"sharpe={sharpe:.2e} (absurd)")
    if trades < ANOMALY_THRESHOLDS["trades_min_meaningful"]:
        anomalies.append(f"trades={trades} (too few)")

    return {
        "name": name,
        "symbol": symbol,
        "sharpe": sharpe if abs(sharpe) < ANOMALY_THRESHOLDS["sharpe_absurd"] else 0,
        "total_return": total_return if abs(total_return) < ANOMALY_THRESHOLDS["return_absurd"] else 0,
        "max_dd": max_dd,
        "trades": trades,
        "source": source,
        "anomalies": anomalies,
    }


def aggregate_by_strategy(entries: list[dict]) -> list[dict]:
    """Aggregate per-symbol results to strategy-level, rank by OOS sharpe."""
    from collections import defaultdict
    grouped: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        grouped[e["name"]].append(e)

    ranked = []
    for name, syms in grouped.items():
        valid = [s for s in syms if not s["anomalies"]]
        all_sharpes = [s["sharpe"] for s in valid]
        all_returns = [s["total_return"] for s in valid]
        total_trades = sum(s["trades"] for s in valid)
        max_dds = [s["max_dd"] for s in valid]
        anomaly_count = sum(1 for s in syms if s["anomalies"])

        import numpy as np
        ranked.append({
            "name": name,
            "sharpe_median": round(float(np.median(all_sharpes)), 4) if all_sharpes else 0,
            "sharpe_best": round(max(all_sharpes), 4) if all_sharpes else 0,
            "return_median": round(float(np.median(all_returns)), 6) if all_returns else 0,
            "max_dd_worst": round(max(max_dds), 6) if max_dds else 0,
            "total_trades": total_trades,
            "symbols_valid": len(valid),
            "symbols_total": len(syms),
            "anomaly_count": anomaly_count,
            "passed_gate": False,
        })

    ranked.sort(key=lambda x: -x["sharpe_median"])
    return ranked


def write_leaderboard(ranked: list[dict], path: Path) -> None:
    lines = [
        "# leaderboard.md — 策略竞赛排行榜（2号位维护）",
        "",
        f"_updated: {datetime.now().isoformat()}_",
        "",
        "hybrid 模式：1号位内部允许同时存在 strategies/A、B、C 等竞品，2号位每天对比验证后排序。",
        "",
        "| rank | name | sharpe(median) | sharpe(best) | return(median) | maxdd(worst) | trades | symbols | anomalies | passed_gate |",
        "|------|------|----------------|--------------|----------------|--------------|--------|---------|-----------|-------------|",
    ]

    for i, r in enumerate(ranked[:20], 1):
        gate_str = "✓" if r["passed_gate"] else "—"
        lines.append(
            f"| {i} | {r['name']} | {r['sharpe_median']:.4f} | {r['sharpe_best']:.4f} | "
            f"{r['return_median']:.4f} | {r['max_dd_worst']:.4f} | {r['total_trades']} | "
            f"{r['symbols_valid']}/{r['symbols_total']} | {r['anomaly_count']} | {gate_str} |"
        )

    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s (%d strategies)", path, len(ranked))


def write_anomaly_report(entries: list[dict]) -> None:
    anomalous = [e for e in entries if e["anomalies"]]
    if not anomalous:
        logger.info("No anomalies detected.")
        return

    logger.warning("ANOMALIES DETECTED: %d entries", len(anomalous))
    for e in anomalous[:20]:
        logger.warning("  %s/%s: %s [%s]", e["name"], e["symbol"],
                        ", ".join(e["anomalies"]), e["source"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan results and generate leaderboard")
    parser.add_argument("--output-leaderboard", default=str(COORDINATION_DIR / "leaderboard.md"))
    args = parser.parse_args()

    logger.info("Scanning results in %s...", RESULTS_DIR)

    entries_latest = scan_latest_results()
    entries_report = scan_backtest_reports()
    all_entries = entries_latest + entries_report

    logger.info("Found %d entries (%d from *_latest.json, %d from *_report.json)",
                len(all_entries), len(entries_latest), len(entries_report))

    write_anomaly_report(all_entries)

    ranked = aggregate_by_strategy(all_entries)

    gate_files = list(RESULTS_DIR.glob("*_gate.json"))
    for gf in gate_files:
        try:
            with open(gf) as fh:
                gdata = json.load(fh)
            strat_name = gdata.get("strategy", "")
            outcome = gdata.get("gate", {}).get("outcome", "reject")
            for r in ranked:
                if r["name"] == strat_name and outcome == "pass":
                    r["passed_gate"] = True
        except (json.JSONDecodeError, OSError):
            pass

    write_leaderboard(ranked, Path(args.output_leaderboard))

    logger.info("\nTop 5 strategies by median sharpe:")
    for i, r in enumerate(ranked[:5], 1):
        logger.info("  %d. %s: sharpe=%.4f return=%.4f trades=%d symbols=%d/%d",
                     i, r["name"], r["sharpe_median"], r["return_median"],
                     r["total_trades"], r["symbols_valid"], r["symbols_total"])


if __name__ == "__main__":
    main()
