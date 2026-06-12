#!/usr/bin/env python3
"""WhaleDetectorStrategy — Optuna 超参优化脚本（大数据版）。

用法:
  cd ~/Polarisor/tqsdk/trading-platform
  .venv/bin/python3 scripts/optuna_whale_tune.py
  .venv/bin/python3 scripts/optuna_whale_tune.py --symbols rb IF i --n-trials 200 --max-bars 15000

每次运行结束后会同时写入两个文件:
  results/whale_detector_optuna_big.json  — 带时间戳的历史快照
  results/whale_detector_optuna_latest.json — 始终指向最新一次 Optuna 结果（供 Agent 查询）
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import optuna

PROJ_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_DIR / "packages"))

from strategy.base import StrategyConfig, SignalType
from strategy.futures.whale_detector import WhaleDetectorStrategy, OPTUNA_PARAM_SPACE

logging.basicConfig(level=logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.WARNING)

CACHE_DIR = PROJ_DIR / "data" / "futures_cache"
RESULTS_DIR = PROJ_DIR / "results"


def load_data(symbol: str, max_bars: int = 10000) -> pd.DataFrame | None:
    f = CACHE_DIR / f"{symbol}_5m_all_all.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f)
    if "instrument" in df.columns:
        df["date"] = pd.to_datetime(df["datetime"], unit="ns").dt.date
        daily_oi = df.groupby(["date", "instrument"])["open_interest"].sum().reset_index()
        main_contracts = daily_oi.loc[daily_oi.groupby("date")["open_interest"].idxmax()]
        main_map = dict(zip(main_contracts["date"], main_contracts["instrument"]))
        df["is_main"] = df.apply(lambda r: r["instrument"] == main_map.get(r["date"]), axis=1)
        df = df[df["is_main"]].copy().sort_values("datetime").reset_index(drop=True)
    df = df.dropna(subset=["open", "high", "low", "close", "volume", "open_interest"])
    return df.tail(max_bars).reset_index(drop=True)


def fast_backtest(sym: str, df: pd.DataFrame, params: dict) -> tuple[float, float, float]:
    cfg = StrategyConfig(name="whale_detector", symbols=[sym], params=params)
    strat = WhaleDetectorStrategy(cfg)
    loop = asyncio.new_event_loop()
    capital = 100_000.0
    position = None
    entry_price = 0.0
    equity = [capital]
    for _, row in df.iterrows():
        bar = {
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "open_interest": float(row["open_interest"]),
        }
        sigs = loop.run_until_complete(strat.on_bar(sym, bar))
        c = bar["close"]
        for sig in sigs:
            if sig.signal_type == SignalType.LONG_ENTRY and position is None:
                position = "long"
                entry_price = c
                capital *= 0.99985
            elif sig.signal_type == SignalType.SHORT_ENTRY and position is None:
                position = "short"
                entry_price = c
                capital *= 0.99985
            elif sig.signal_type == SignalType.LONG_EXIT and position == "long":
                pnl = (c - entry_price) / entry_price
                capital *= (1 + pnl) * 0.99985
                position = None
            elif sig.signal_type == SignalType.SHORT_EXIT and position == "short":
                pnl = (entry_price - c) / entry_price
                capital *= (1 + pnl) * 0.99985
                position = None
        equity.append(capital)
    loop.close()
    eq = np.array(equity)
    step = max(1, len(eq) // 52)
    wr = [(eq[i] - eq[i - step]) / eq[i - step] for i in range(step, len(eq), step) if eq[i - step] > 0]
    sharpe = float(np.mean(wr) / max(np.std(wr), 1e-10) * np.sqrt(52)) if len(wr) >= 2 else 0.0
    ret = (capital - 100_000.0) / 100_000.0
    peak = np.maximum.accumulate(eq)
    dd = float(np.max((peak - eq) / peak))
    return sharpe, ret, dd


def main() -> None:
    parser = argparse.ArgumentParser(description="WhaleDetector Optuna Tuner")
    parser.add_argument("--symbols", nargs="+", default=["rb", "IF", "i"])
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--max-bars", type=int, default=10000, help="每品种训练用 bars 数")
    parser.add_argument("--val-bars", type=int, default=30000, help="每品种验证用 bars 数")
    args = parser.parse_args()

    # 加载训练数据
    data: dict[str, pd.DataFrame] = {}
    for s in args.symbols:
        d = load_data(s, max_bars=args.max_bars)
        if d is not None and len(d) > 1000:
            data[s] = d
            print(f"Loaded {s}: {len(d)} bars")

    if not data:
        print("No data loaded, abort.")
        sys.exit(1)

    def objective(trial: optuna.Trial) -> float:
        params: dict = {}
        for k, (typ, lo, hi) in OPTUNA_PARAM_SPACE.items():
            if typ == "int":
                params[k] = trial.suggest_int(k, lo, hi)
            else:
                params[k] = trial.suggest_float(k, lo, hi)
        total = 0.0
        for sym, df in data.items():
            sh, _ret, dd = fast_backtest(sym, df, params)
            total += sh - max(0.0, dd - 0.15) * 5
        return total / len(data)

    start = time.time()
    study = optuna.create_study(direction="maximize", study_name="whale_detector_big")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
    elapsed = time.time() - start

    print(f"\n=== Optuna on Big Data ({elapsed:.0f}s, {args.n_trials} trials) ===")
    print(f"Best adjusted Sharpe: {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {round(v, 4) if isinstance(v, float) else v}")

    # 验证阶段
    print(f"\n=== Validation on {args.val_bars} bars ===")
    val_results: dict[str, dict] = {}
    for s in args.symbols:
        full = load_data(s, max_bars=args.val_bars)
        if full is None:
            continue
        sh, ret, dd = fast_backtest(s, full, study.best_params)
        val_results[s] = {"ret_pct": round(ret * 100, 2), "sharpe": round(sh, 3), "dd_pct": round(dd * 100, 1)}
        print(f"  {s:>4s}: Ret={ret*100:+8.2f}%  Sharpe={sh:+.3f}  DD={dd*100:.1f}%")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.datetime.now().isoformat()

    payload = {
        "best_sharpe": study.best_value,
        "best_params": study.best_params,
        "n_trials": args.n_trials,
        "elapsed_sec": elapsed,
        "symbols": args.symbols,
        "max_bars_train": args.max_bars,
        "val_bars": args.val_bars,
        "validation": val_results,
        "generated_at": generated_at,
    }

    # 历史快照（追加时间戳以便保留多次跑的记录）
    snapshot_path = RESULTS_DIR / "whale_detector_optuna_big.json"
    with open(snapshot_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved snapshot → {snapshot_path}")

    # latest 指针 — 始终覆写，供 Agent / 文档系统查询
    latest_path = RESULTS_DIR / "whale_detector_optuna_latest.json"
    latest_payload = {
        "_meta": {
            "source_script": "scripts/optuna_whale_tune.py",
            "generated_at": generated_at,
            "description": (
                "此文件由 optuna_whale_tune.py 每次运行后自动覆写，"
                "始终代表最新一次 Optuna 超参优化结果。"
                "若需查询 whale 策略最新参数，读此文件而非 optuna_big.json。"
            ),
        },
        **payload,
    }
    with open(latest_path, "w") as f:
        json.dump(latest_payload, f, indent=2)
    print(f"Latest pointer updated → {latest_path}")


if __name__ == "__main__":
    main()
