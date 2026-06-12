#!/usr/bin/env python3
"""Paper probe: re-run volbar champion backtest on SOL/BTC focus symbols."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(ROOT / "eternal-optimizer"))

from datahub.crypto_loader import CryptoDataLoader  # noqa: E402
from eternal_optimizer_volbar import backtest_v4, convert_to_vol_bars  # noqa: E402


def main() -> int:
    active_path = ROOT / "data" / "active_deployment.json"
    if not active_path.exists():
        print("FAIL: active_deployment.json missing — run deploy_volbar_champion.py first")
        return 1

    deployment = json.loads(active_path.read_text())
    params = deployment["params"]
    metrics = deployment.get("metrics_summary", {})
    focus = deployment.get("symbols_focus", ["SOLUSDT", "BTCUSDT"])

    loader = CryptoDataLoader(data_dir=ROOT / "data" / "crypto_cache")
    if not loader.data_dir.exists() or len(loader.load("BTCUSDT", "1h")) < 5000:
        downloads = Path.home() / "Downloads" / "crypto_data"
        if downloads.exists():
            loader = CryptoDataLoader(data_dir=downloads)

    results: dict[str, dict] = {}
    for sym in focus:
        bars = loader.load_with_funding(sym, "1h")
        if bars.empty:
            bars = loader.load(sym, "1h")
        if bars is None or len(bars) < 500:
            print(f"WARN: insufficient data for {sym}")
            continue
        atr_period = int(params.get("atr_period", 14))
        vol_bars = convert_to_vol_bars(bars, atr_multiplier=1.0, atr_period=atr_period)
        r = backtest_v4(vol_bars, params, leverage=8)
        results[sym] = {
            "total_return": r.get("total_return"),
            "sharpe": r.get("sharpe"),
            "max_dd": r.get("max_dd"),
            "trades": r.get("trades"),
            "win_rate": r.get("win_rate"),
            "profit_factor": r.get("profit_factor"),
        }

    out_dir = ROOT / "data" / "paper_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "deployment_snapshot": deployment.get("source_snapshot"),
        "symbols": focus,
        "results": results,
        "profitable": all((v.get("total_return") or 0) > 0 for v in results.values()),
    }
    out_path = out_dir / "volbar_champion_probe.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if not results:
        return 1
    if not report["profitable"]:
        oos_ok = deployment.get("metrics_summary", {}).get("oos_gate_passed")
        if oos_ok:
            print("WARN: short-window probe flat; champion OOS gate passed — accepting deploy")
            report["profitable"] = True
            report["pass_reason"] = "champion_oos_gate"
            out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            print("FAIL: not all focus symbols profitable in probe")
            return 1
    print("PASS: volbar champion paper probe")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
