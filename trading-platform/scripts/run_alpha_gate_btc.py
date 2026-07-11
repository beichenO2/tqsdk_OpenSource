#!/usr/bin/env python3
"""Run AlphaGate on four BTC hypothesis factors (4h primary, 1d for H3).

Usage:
    .venv/bin/python scripts/run_alpha_gate_btc.py
    .venv/bin/python scripts/run_alpha_gate_btc.py --symbol BTCUSDT --cost-bps 5
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
from factor.alpha_gate import AlphaGate, causal_rolling_zscore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("alpha-gate-btc")

BARS_PER_YEAR_4H = 6 * 365


def has_real_funding_data(data_dir: Path, symbol: str) -> bool:
    """True only when real funding/premium parquet exists (no synthetic fallback)."""
    sym_dir = data_dir / symbol.lower()
    if (sym_dir / "funding_rate.parquet").exists():
        return True
    return any(sym_dir.glob("premium_index*.parquet"))


def _normalize_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True).dt.as_unit("ns")


def load_funding_series(data_dir: Path, symbol: str, open_time: pd.Series) -> pd.Series | None:
    sym_dir = data_dir / symbol.lower()
    times = _normalize_dt(open_time).sort_values()
    base = pd.DataFrame({"open_time": times, "_idx": open_time.index}).sort_values("open_time")

    premium_files = sorted(sym_dir.glob("premium_index*.parquet"))
    if premium_files:
        pi = pd.read_parquet(premium_files[0])
        pi["open_time"] = _normalize_dt(pi["open_time"])
        pi = pi.sort_values("open_time")
        rate_col = "close" if "close" in pi.columns else "premium_index"
        right = pi[["open_time", rate_col]].rename(columns={rate_col: "funding_rate"})
        merged = pd.merge_asof(
            base,
            right,
            on="open_time",
            direction="backward",
        )
        out = pd.Series(merged["funding_rate"].values, index=merged["_idx"].values)
        return out.reindex(open_time.index)

    funding_path = sym_dir / "funding_rate.parquet"
    if funding_path.exists():
        fr = pd.read_parquet(funding_path)
        time_col = "funding_time" if "funding_time" in fr.columns else "open_time"
        fr[time_col] = _normalize_dt(fr[time_col])
        fr = fr.sort_values(time_col)
        right = fr[[time_col, "funding_rate"]].rename(columns={time_col: "open_time"})
        merged = pd.merge_asof(
            base,
            right,
            on="open_time",
            direction="backward",
        )
        out = pd.Series(merged["funding_rate"].values, index=merged["_idx"].values)
        return out.reindex(open_time.index).ffill().fillna(0.0)

    return None


def factor_h1_funding_contrarian(
    ohlcv: pd.DataFrame, data_dir: Path, symbol: str
) -> tuple[pd.Series | None, str]:
    if not has_real_funding_data(data_dir, symbol):
        return None, "SKIPPED (no real funding/premium data)"
    funding = load_funding_series(data_dir, symbol, ohlcv["open_time"])
    if funding is None or funding.dropna().empty:
        return None, "SKIPPED (funding merge failed)"
    window = 30 * 6  # 30d at 4h
    z = causal_rolling_zscore(funding, window)
    position = (-z).clip(-1, 1).fillna(0.0)
    return position, "OK"


def factor_h2_vol_adj_momentum(ohlcv: pd.DataFrame) -> pd.Series:
    close = ohlcv["close"].astype(float)
    roc = close.pct_change(42)
    vol = close.pct_change().rolling(42, min_periods=21).std()
    raw = roc / vol.replace(0.0, np.nan)
    z = causal_rolling_zscore(raw, 180)
    return z.clip(-1, 1).fillna(0.0)


def factor_h3_daily_mean_reversion(
    ohlcv_4h: pd.DataFrame, ohlcv_1d: pd.DataFrame
) -> pd.Series:
    close_1d = ohlcv_1d.set_index("open_time")["close"].astype(float)
    z = causal_rolling_zscore(close_1d, 30)
    pos_1d = (-z).clip(-1, 1).fillna(0.0)
    idx = pd.to_datetime(ohlcv_4h["open_time"], utc=True)
    pos_4h = pos_1d.reindex(idx, method="ffill").fillna(0.0)
    pos_4h.index = ohlcv_4h.index
    return pos_4h


def factor_h4_short_high_momentum(ohlcv: pd.DataFrame) -> pd.Series:
    close = ohlcv["close"].astype(float)
    roc6 = close.pct_change(6)
    z = causal_rolling_zscore(roc6, 180)
    sign = np.sign(roc6).replace(0, 0.0)
    position = -sign * np.minimum(z.abs(), 1.0)
    return position.fillna(0.0)


def build_factor_specs(
    ohlcv_4h: pd.DataFrame,
    ohlcv_1d: pd.DataFrame,
    data_dir: Path,
    symbol: str,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []

    pos, status = factor_h1_funding_contrarian(ohlcv_4h, data_dir, symbol)
    specs.append({"id": "H1", "name": "funding_contrarian", "position": pos, "status": status})

    specs.append({
        "id": "H2",
        "name": "vol_adj_momentum",
        "position": factor_h2_vol_adj_momentum(ohlcv_4h),
        "status": "OK",
    })

    if ohlcv_1d.empty:
        specs.append({
            "id": "H3",
            "name": "daily_mean_reversion",
            "position": None,
            "status": "SKIPPED (no 1d data)",
        })
    else:
        specs.append({
            "id": "H3",
            "name": "daily_mean_reversion",
            "position": factor_h3_daily_mean_reversion(ohlcv_4h, ohlcv_1d),
            "status": "OK",
        })

    specs.append({
        "id": "H4",
        "name": "short_high_momentum_4h",
        "position": factor_h4_short_high_momentum(ohlcv_4h),
        "status": "OK",
    })
    return specs


def evaluate_factors(
    specs: list[dict[str, Any]],
    ohlcv: pd.DataFrame,
    cost_bps: float,
) -> list[dict[str, Any]]:
    gate = AlphaGate(one_way_cost_bps=cost_bps, bars_per_year=BARS_PER_YEAR_4H)
    results: list[dict[str, Any]] = []
    for spec in specs:
        entry: dict[str, Any] = {
            "id": spec["id"],
            "name": spec["name"],
            "status": spec["status"],
        }
        if spec["position"] is None:
            entry["verdict"] = spec["status"]
            results.append(entry)
            continue
        report = gate.evaluate(spec["position"], ohlcv)
        entry.update(report.to_dict())
        entry["status"] = spec["status"]
        results.append(entry)
    return results


def render_report_md(
    results: list[dict[str, Any]],
    symbol: str,
    cost_bps: float,
    ts: str,
) -> str:
    lines = [
        f"# AlphaGate BTC Hypothesis Report",
        "",
        f"- **Symbol:** {symbol}",
        f"- **Timeframe:** 4h (H3 uses 1d → 4h ffill)",
        f"- **Generated:** {ts}",
        f"- **Default one-way cost:** {cost_bps} bp",
        "",
        "## Factor Summary",
        "",
        "| Factor | Status | Verdict | Net Return | Sharpe | Turnover | Trade E[bp] | Gates |",
        "|--------|--------|---------|------------|--------|----------|-------------|-------|",
    ]
    for r in results:
        if r.get("position") is None or "verdict" not in r or r["status"].startswith("SKIPPED"):
            verdict = r.get("verdict", r["status"])
            lines.append(f"| {r['id']} {r['name']} | {r['status']} | {verdict} | — | — | — | — | — |")
            continue
        m = r.get("metrics", {})
        gates = r.get("gates", {})
        n_pass = sum(g.get("passed", False) for g in gates.values()) if gates else 0
        lines.append(
            f"| {r['id']} {r['name']} | {r['status']} | {r['verdict']} | "
            f"{m.get('total_net_return', 0):.4f} | {m.get('net_sharpe', 0):.2f} | "
            f"{m.get('annual_turnover', 0):.1f} | {m.get('trade_expectancy_bp', 0):.1f} | "
            f"{n_pass}/5 |"
        )

    for r in results:
        lines.extend(["", f"## {r['id']}: {r['name']}", ""])
        if r["status"].startswith("SKIPPED") or "gates" not in r:
            lines.append(f"**Status:** {r['status']}")
            continue
        lines.append(f"**Verdict:** {r['verdict']}")
        lines.append("")
        lines.append("| Gate | Pass | Value | Threshold |")
        lines.append("|------|------|-------|-----------|")
        for gk in ("G1_walk_forward", "G2_cost_sensitivity", "G3_trade_expectancy",
                   "G4_turnover_cost", "G5_benchmark"):
            g = r["gates"][gk]
            mark = "✓" if g["passed"] else "✗"
            val = g.get("value")
            val_s = "—" if val is None else f"{val:.4f}"
            lines.append(f"| {g['name']} | {mark} | {val_s} | {g['threshold']} |")
        m = r["metrics"]
        lines.extend([
            "",
            f"- Net return @ 2/5/10bp: {m.get('net_return_2bp', 0):.4f} / "
            f"{m.get('net_return_5bp', 0):.4f} / {m.get('net_return_10bp', 0):.4f}",
            f"- Buy&hold: return {m.get('buy_hold_return', 0):.4f}, "
            f"Sharpe {m.get('buy_hold_sharpe', 0):.4f} | "
            f"Supertrend: return {m.get('supertrend_return', 0):.4f}, "
            f"Sharpe {m.get('supertrend_sharpe', 0):.4f}",
        ])
        g5 = r["gates"].get("G5_benchmark", {})
        g5d = g5.get("details", {})
        if g5d:
            lines.extend([
                f"- G5a (Sharpe): {'✓' if g5d.get('g5a_passed') else '✗'} | "
                f"G5b (WF vs B&H): {g5d.get('segments_beat_bh', 0)}/4 "
                f"{'✓' if g5d.get('g5b_passed') else '✗'}",
            ])
    return "\n".join(lines)


def run_pipeline(
    symbol: str = "BTCUSDT",
    data_dir: str | Path | None = None,
    cost_bps: float = 5.0,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    data_path = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    out_path = Path(out_dir) if out_dir else ROOT / "output" / "research"
    out_path.mkdir(parents=True, exist_ok=True)

    loader = CryptoDataLoader(data_path)
    ohlcv_4h = loader.load(symbol, "4h")
    ohlcv_1d = loader.load(symbol, "1d")
    if ohlcv_4h.empty:
        raise FileNotFoundError(f"No 4h data for {symbol} in {data_path}")

    specs = build_factor_specs(ohlcv_4h, ohlcv_1d, data_path, symbol)
    results = evaluate_factors(specs, ohlcv_4h, cost_bps)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    md_path = out_path / f"alpha_gate_report_{ts}.md"
    json_path = out_path / f"alpha_gate_report_{ts}.json"

    payload = {
        "ts": ts,
        "symbol": symbol,
        "timeframe": "4h",
        "cost_bps": cost_bps,
        "data_dir": str(data_path),
        "n_bars": len(ohlcv_4h),
        "factors": results,
    }
    md_path.write_text(render_report_md(results, symbol, cost_bps, ts), encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s and %s", md_path, json_path)

    return {
        "report_path": str(md_path),
        "json_path": str(json_path),
        "factors": results,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AlphaGate BTC hypothesis evaluation")
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
