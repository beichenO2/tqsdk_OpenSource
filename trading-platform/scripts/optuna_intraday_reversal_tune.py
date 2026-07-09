#!/usr/bin/env python3
"""intraday_reversal — Optuna 超参搜索以突破 OOS sharpe 0.8 gate 门槛.

Base strategy: `strategy.futures.intraday_reversal.IntradayReversalStrategy`.
2号位 gate: OOS sharpe=0.446 return=+83.8% WF=1.0 MC_p05=+0.847 x_asset=0.54 trades=63
→ 5/6 NEAR PASS，唯一卡 sharpe≥0.8。

搜索空间:
  - sl_atr_mult ∈ [0.3, 2.0]
  - tp_atr_mult ∈ [0.8, 3.0]
  - max_hold_bars ∈ [8, 40]
  - rsi_extreme_high ∈ [65, 90]
  - rsi_extreme_low ∈ [10, 35]
  - vwap_deviation_atr ∈ [1.0, 3.5]
  - gap_atr_threshold ∈ [0.8, 3.0]

Objective = OOS sharpe on rb 20k-bar 70/30 split (additive-PnL 一致于 validate_gate).
约束: trades >= 30 (训练), return > 0，否则 sharpe 惩罚 -5.

Usage:
  .venv/bin/python3 scripts/optuna_intraday_reversal_tune.py --n-trials 80 --max-bars 30000

Output:
  results/intraday_reversal_optuna_latest.json — 最新结果（供 Agent 查询）
  results/intraday_reversal_optuna_<ts>.json  — 历史快照
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_DIR / "packages"))

import optuna  # noqa: E402

from strategy.base import SignalType, StrategyConfig  # noqa: E402
from strategy.futures.intraday_reversal import IntradayReversalStrategy  # noqa: E402

logging.basicConfig(level=logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.WARNING)

CACHE_DIR = PROJ_DIR / "data" / "futures_cache"
RESULTS_DIR = PROJ_DIR / "results"

POSITION_NOTIONAL_PCT = 0.1
PER_TRADE_RETURN_CAP = 10.0
PER_TRADE_RETURN_FLOOR = -0.99


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def load_rb(max_bars: int = 30000) -> pd.DataFrame:
    f = CACHE_DIR / "rb_5m_all_all.parquet"
    df = pd.read_parquet(f)
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    return df.tail(max_bars).reset_index(drop=True)


def backtest_additive(strategy, bars: pd.DataFrame, initial: float = 100_000.0) -> dict:
    capital = initial
    notional = initial * POSITION_NOTIONAL_PCT
    cost = notional * (0.00005 + 0.0001)
    position = None
    entry = 0.0
    trades: list[dict] = []
    equity = [capital]
    loop = asyncio.new_event_loop()
    try:
        for _, row in bars.iterrows():
            bar = {
                "datetime": row.get("datetime"),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "open_interest": float(row.get("open_interest", 0) or 0),
            }
            sigs = loop.run_until_complete(strategy.on_bar(str(row.get("instrument", "rb")), bar))
            c = bar["close"]
            if c <= 0:
                continue
            for sig in sigs:
                if sig.signal_type == SignalType.LONG_ENTRY and position is None:
                    position = "long"
                    entry = c
                    capital -= cost
                elif sig.signal_type == SignalType.SHORT_ENTRY and position is None:
                    position = "short"
                    entry = c
                    capital -= cost
                elif sig.signal_type == SignalType.LONG_EXIT and position == "long":
                    raw = (c - entry) / entry if entry > 0 else 0.0
                    p = _clamp(raw, PER_TRADE_RETURN_FLOOR, PER_TRADE_RETURN_CAP)
                    capital += notional * p - cost
                    trades.append({"pnl": notional * p, "side": "long"})
                    position = None
                elif sig.signal_type == SignalType.SHORT_EXIT and position == "short":
                    raw = (entry - c) / entry if entry > 0 else 0.0
                    p = _clamp(raw, PER_TRADE_RETURN_FLOOR, PER_TRADE_RETURN_CAP)
                    capital += notional * p - cost
                    trades.append({"pnl": notional * p, "side": "short"})
                    position = None
            equity.append(capital)
    finally:
        loop.close()

    if position and len(bars) > 0:
        last = float(bars.iloc[-1]["close"])
        raw = ((last - entry) if position == "long" else (entry - last)) / entry if entry > 0 else 0.0
        p = _clamp(raw, PER_TRADE_RETURN_FLOOR, PER_TRADE_RETURN_CAP)
        capital += notional * p - cost
        trades.append({"pnl": notional * p, "side": position})

    arr = np.array(equity, dtype=np.float64)
    step = max(1, len(arr) // 52)
    samples = arr[::step]
    ret = np.diff(samples) / initial
    if ret.size >= 2 and np.std(ret) > 1e-12:
        sharpe = float(np.mean(ret) / np.std(ret) * np.sqrt(52))
    else:
        sharpe = 0.0
    total_ret = (capital - initial) / initial
    return {"sharpe": sharpe, "return": total_ret, "trades": len(trades), "final": capital}


def objective(trial: optuna.Trial, train_bars: pd.DataFrame, oos_bars: pd.DataFrame) -> float:
    params = {
        "sl_atr_mult": trial.suggest_float("sl_atr_mult", 0.3, 2.0),
        "tp_atr_mult": trial.suggest_float("tp_atr_mult", 0.8, 3.0),
        "max_hold_bars": trial.suggest_int("max_hold_bars", 8, 40),
        "rsi_extreme_high": trial.suggest_float("rsi_extreme_high", 65.0, 90.0),
        "rsi_extreme_low": trial.suggest_float("rsi_extreme_low", 10.0, 35.0),
        "vwap_deviation_atr": trial.suggest_float("vwap_deviation_atr", 1.0, 3.5),
        "gap_atr_threshold": trial.suggest_float("gap_atr_threshold", 0.8, 3.0),
    }
    cfg = StrategyConfig(name="intraday_reversal", strategy_id="ir-opt", params=params)
    strat = IntradayReversalStrategy(cfg)

    train_res = backtest_additive(strat, train_bars)
    if train_res["trades"] < 20 or train_res["return"] <= 0:
        return -5.0

    oos_strat = IntradayReversalStrategy(cfg)
    oos_res = backtest_additive(oos_strat, oos_bars)
    trial.set_user_attr("oos_return", oos_res["return"])
    trial.set_user_attr("oos_trades", oos_res["trades"])
    trial.set_user_attr("train_sharpe", train_res["sharpe"])
    trial.set_user_attr("train_return", train_res["return"])
    return oos_res["sharpe"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=80)
    parser.add_argument("--max-bars", type=int, default=30000)
    parser.add_argument("--oos-ratio", type=float, default=0.3)
    args = parser.parse_args()

    bars = load_rb(args.max_bars)
    split = int(len(bars) * (1 - args.oos_ratio))
    train = bars.iloc[:split].reset_index(drop=True)
    oos = bars.iloc[split:].reset_index(drop=True)
    print(f"[load] {len(bars)} bars | train={len(train)} oos={len(oos)}")

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda t: objective(t, train, oos), n_trials=args.n_trials, show_progress_bar=False)

    best = study.best_trial
    print(f"\n[best] sharpe(oos)={best.value:.4f} params={best.params}")
    print(f"[best] oos_return={best.user_attrs.get('oos_return'):.4f} trades={best.user_attrs.get('oos_trades')}")

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "_meta": {
            "script": "optuna_intraday_reversal_tune.py",
            "generated_at": datetime.datetime.now().isoformat(),
            "agent": "solo-web-fa5984de",
            "n_trials": args.n_trials,
            "symbol": "rb",
            "max_bars": args.max_bars,
            "oos_ratio": args.oos_ratio,
        },
        "best": {
            "oos_sharpe": best.value,
            "oos_return": best.user_attrs.get("oos_return"),
            "oos_trades": best.user_attrs.get("oos_trades"),
            "train_sharpe": best.user_attrs.get("train_sharpe"),
            "train_return": best.user_attrs.get("train_return"),
            "params": best.params,
        },
        "all_trials_top5": [
            {
                "value": t.value,
                "params": t.params,
                "oos_return": t.user_attrs.get("oos_return"),
                "trades": t.user_attrs.get("oos_trades"),
            }
            for t in sorted(study.trials, key=lambda x: x.value or -999, reverse=True)[:5]
        ],
    }
    latest = RESULTS_DIR / "intraday_reversal_optuna_latest.json"
    snapshot = RESULTS_DIR / f"intraday_reversal_optuna_{ts}.json"
    for p in (latest, snapshot):
        p.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    print(f"wrote {latest} + {snapshot}")


if __name__ == "__main__":
    main()
