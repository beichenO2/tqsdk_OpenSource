#!/usr/bin/env python3
"""2号位核心工具：策略门禁验证（OOS + WF + MC + X-asset）。

用法:
  cd ~/Polarisor/tqsdk-gnhf-worktrees/pos2/trading-platform
  python scripts/validate_gate.py --strategy attack_defense --symbols rb SA MA
  python scripts/validate_gate.py --strategy attack_defense --full-gate

Gate 阈值 (contracts.md §4):
  oos.sharpe >= 0.8  且 oos.return > 0
  wf.consistency >= 0.6  (>=60% fold 正收益)
  mc.p05_return >= -0.15  (蒙特卡洛 5 分位回撤 <= 15%)
  x_asset.median_sharpe >= 0.5
  train.trades >= 30

输出:
  results/<name>_oos.json   — OOS 回测结果
  results/<name>_wf.json    — Walk-forward 结果
  results/<name>_mc.json    — Monte Carlo 结果
  results/<name>_gate.json  — 综合 gate 判定
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJ_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_DIR / "packages"))

from datahub.futures_loader import FuturesDataLoader
from strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig
from strategy.registry import StrategyRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validate-gate")

CACHE_DIR = str(PROJ_DIR / "data" / "futures_cache")

GATE_THRESHOLDS = {
    "oos_sharpe": 0.8,
    "oos_return_min": 0.0,
    "wf_consistency": 0.6,
    "mc_p05_return": -0.15,
    "x_asset_median_sharpe": 0.5,
    "train_trades_min": 30,
}

RETURN_CAP = 100.0
SHARPE_CAP = 50.0

_STRATEGY_INIT_DONE = False


def _ensure_strategy_registered(name: str) -> None:
    """Register strategies. First try package import, then individual modules."""
    global _STRATEGY_INIT_DONE
    if StrategyRegistry.get(name) is not None:
        return
    if _STRATEGY_INIT_DONE:
        return
    _STRATEGY_INIT_DONE = True

    try:
        import strategy.futures  # noqa: F401
    except Exception as exc:
        logger.warning("Failed to import strategy.futures: %s", exc)
    try:
        import strategy.btc  # noqa: F401
    except Exception as exc:
        logger.debug("Failed to import strategy.btc: %s", exc)

    if StrategyRegistry.get(name) is None:
        import importlib
        for pkg in ("strategy.futures", "strategy.btc"):
            try:
                importlib.import_module(f"{pkg}.{name}")
            except Exception:
                pass


def _clamp(value: float, lo: float = -1e10, hi: float = 1e10) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(lo, min(hi, value))


def backtest_on_bars(
    strategy: BaseStrategy,
    bars: pd.DataFrame,
    initial_capital: float = 100_000.0,
    commission_rate: float = 0.00005,
    slippage_pct: float = 0.0001,
) -> dict[str, Any]:
    """Additive PnL backtester (fixes the multiplicative compounding e+197 bug)."""
    capital = initial_capital
    position = None
    entry_price = 0.0
    trades: list[dict] = []
    equity_curve = [capital]

    loop = asyncio.new_event_loop()

    for _, row in bars.iterrows():
        bar = {
            "datetime": row.get("datetime"),
            "open": float(row.get("open", 0)),
            "high": float(row.get("high", 0)),
            "low": float(row.get("low", 0)),
            "close": float(row.get("close", 0)),
            "volume": float(row.get("volume", 0)),
            "open_interest": float(row.get("open_interest", 0)) if "open_interest" in row.index else 0.0,
        }
        symbol = str(row.get("instrument", "unknown"))

        try:
            signals = loop.run_until_complete(strategy.on_bar(symbol, bar))
        except Exception:
            signals = []

        close_price = bar["close"]
        if close_price <= 0:
            equity_curve.append(capital)
            continue

        for sig in signals:
            if sig.signal_type == SignalType.LONG_ENTRY and position is None:
                position = "long"
                entry_price = close_price
                capital -= capital * (commission_rate + slippage_pct)

            elif sig.signal_type == SignalType.SHORT_ENTRY and position is None:
                position = "short"
                entry_price = close_price
                capital -= capital * (commission_rate + slippage_pct)

            elif sig.signal_type == SignalType.LONG_EXIT and position == "long":
                pnl_pct = (close_price - entry_price) / entry_price
                pnl_pct = _clamp(pnl_pct, -0.99, RETURN_CAP)
                pnl_abs = capital * pnl_pct
                capital += pnl_abs - capital * (commission_rate + slippage_pct)
                capital = max(capital, 0.01)
                trades.append({"pnl_pct": pnl_pct, "pnl_abs": pnl_abs, "side": "long"})
                position = None

            elif sig.signal_type == SignalType.SHORT_EXIT and position == "short":
                pnl_pct = (entry_price - close_price) / entry_price
                pnl_pct = _clamp(pnl_pct, -0.99, RETURN_CAP)
                pnl_abs = capital * pnl_pct
                capital += pnl_abs - capital * (commission_rate + slippage_pct)
                capital = max(capital, 0.01)
                trades.append({"pnl_pct": pnl_pct, "pnl_abs": pnl_abs, "side": "short"})
                position = None

        equity_curve.append(capital)

    loop.close()

    if position and len(bars) > 0:
        last_close = float(bars.iloc[-1]["close"])
        if position == "long":
            pnl_pct = (last_close - entry_price) / entry_price if entry_price > 0 else 0.0
        else:
            pnl_pct = (entry_price - last_close) / entry_price if entry_price > 0 else 0.0
        pnl_pct = _clamp(pnl_pct, -0.99, RETURN_CAP)
        capital += capital * pnl_pct
        trades.append({"pnl_pct": pnl_pct, "pnl_abs": capital * pnl_pct, "side": position})

    total_return = _clamp((capital - initial_capital) / initial_capital)
    n_trades = len(trades)
    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]

    win_rate = len(wins) / n_trades if n_trades > 0 else 0.0
    avg_win = float(np.mean([t["pnl_pct"] for t in wins])) if wins else 0.0
    avg_loss = float(np.mean([abs(t["pnl_pct"]) for t in losses])) if losses else 0.0
    gross_profit = sum(t["pnl_pct"] for t in wins) if wins else 0.0
    gross_loss = sum(abs(t["pnl_pct"]) for t in losses) if losses else 0.0
    profit_factor = gross_profit / max(gross_loss, 1e-10) if gross_loss > 1e-10 else (10.0 if wins else 0.0)

    equity = np.array(equity_curve)
    peak = np.maximum.accumulate(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = np.where(peak > 0, (peak - equity) / peak, 0.0)
    max_dd = float(np.nanmax(drawdown)) if len(drawdown) > 0 else 0.0

    sharpe = _compute_sharpe(equity)

    return {
        "total_return": round(_clamp(total_return), 6),
        "sharpe": round(_clamp(sharpe, -SHARPE_CAP, SHARPE_CAP), 4),
        "max_dd": round(max_dd, 6),
        "trades": n_trades,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(_clamp(profit_factor, 0, 999), 4),
        "avg_win": round(avg_win, 6),
        "avg_loss": round(avg_loss, 6),
        "final_capital": round(capital, 2),
        "bars": len(bars),
        "pnl_series": [t["pnl_abs"] for t in trades],
    }


def _compute_sharpe(equity: np.ndarray, periods_per_year: int = 252) -> float:
    if len(equity) < 10:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.diff(equity) / equity[:-1]
    returns = returns[np.isfinite(returns)]
    if len(returns) < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std < 1e-12:
        return 0.0
    return float(np.mean(returns) / std * np.sqrt(periods_per_year))


# ── OOS validation ──────────────────────────────────────────────────────────

def run_oos(
    strategy_name: str,
    symbol: str,
    bars: pd.DataFrame,
    oos_ratio: float = 0.3,
) -> dict[str, Any]:
    """Split bars into train / OOS, run on OOS portion only."""
    n = len(bars)
    split_idx = int(n * (1 - oos_ratio))
    train_bars = bars.iloc[:split_idx].copy()
    oos_bars = bars.iloc[split_idx:].copy()

    if len(train_bars) < 50 or len(oos_bars) < 20:
        return {"error": f"insufficient data: train={len(train_bars)} oos={len(oos_bars)}"}

    _ensure_strategy_registered(strategy_name)
    config = StrategyConfig(name=strategy_name, strategy_id=f"{strategy_name}_train")
    strategy_cls = StrategyRegistry.get(strategy_name)
    if strategy_cls is None:
        return {"error": f"strategy {strategy_name} not found"}

    train_strat = strategy_cls(config)
    train_result = backtest_on_bars(train_strat, train_bars)

    oos_strat = strategy_cls(config)
    oos_result = backtest_on_bars(oos_strat, oos_bars)

    return {
        "symbol": symbol,
        "total_bars": n,
        "train_bars": len(train_bars),
        "oos_bars": len(oos_bars),
        "oos_ratio": oos_ratio,
        "train": {k: v for k, v in train_result.items() if k != "pnl_series"},
        "oos": {k: v for k, v in oos_result.items() if k != "pnl_series"},
        "oos_pnl_series": oos_result.get("pnl_series", []),
    }


# ── Walk-Forward validation ────────────────────────────────────────────────

def run_walk_forward(
    strategy_name: str,
    symbol: str,
    bars: pd.DataFrame,
    n_folds: int = 5,
) -> dict[str, Any]:
    """Rolling walk-forward: split into n_folds, train on k, test on k+1."""
    n = len(bars)
    fold_size = n // n_folds
    if fold_size < 30:
        return {"error": f"insufficient data for {n_folds} folds: {n} bars / {fold_size} per fold"}

    _ensure_strategy_registered(strategy_name)
    strategy_cls = StrategyRegistry.get(strategy_name)
    if strategy_cls is None:
        return {"error": f"strategy {strategy_name} not found"}

    folds: list[dict] = []
    positive_folds = 0

    for i in range(n_folds - 1):
        train_start = i * fold_size
        train_end = (i + 1) * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n)

        train_bars = bars.iloc[train_start:train_end].copy()
        test_bars = bars.iloc[test_start:test_end].copy()

        if len(test_bars) < 10:
            continue

        config = StrategyConfig(name=strategy_name, strategy_id=f"{strategy_name}_wf_{i}")
        strat = strategy_cls(config)
        test_result = backtest_on_bars(strat, test_bars)

        fold_result = {
            "fold": i,
            "train_range": [train_start, train_end],
            "test_range": [test_start, test_end],
            "test_return": test_result["total_return"],
            "test_sharpe": test_result["sharpe"],
            "test_trades": test_result["trades"],
        }
        folds.append(fold_result)

        if test_result["total_return"] > 0:
            positive_folds += 1

    n_completed = len(folds)
    consistency = positive_folds / n_completed if n_completed > 0 else 0.0
    median_sharpe = float(np.median([f["test_sharpe"] for f in folds])) if folds else 0.0

    return {
        "symbol": symbol,
        "n_folds": n_folds,
        "completed_folds": n_completed,
        "positive_folds": positive_folds,
        "consistency": round(consistency, 4),
        "median_sharpe": round(median_sharpe, 4),
        "folds": folds,
    }


# ── Monte Carlo validation ─────────────────────────────────────────────────

def run_monte_carlo(
    pnl_series: list[float],
    initial_capital: float = 100_000.0,
    n_sims: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Shuffle trade PnLs to estimate tail risk."""
    pnls = np.array(pnl_series, dtype=np.float64)
    if pnls.size < 3:
        return {"error": f"insufficient trades for MC: {pnls.size}"}

    rng = np.random.default_rng(seed)
    order = np.argsort(rng.random((n_sims, pnls.size)), axis=1)
    shuffled = pnls[order]
    cumulative = np.cumsum(shuffled, axis=1)
    equity = initial_capital + np.hstack([np.zeros((n_sims, 1)), cumulative])
    peaks = np.maximum.accumulate(equity, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd_pct = np.where(peaks > 0, (peaks - equity) / peaks, 0.0)
    max_dd = dd_pct.max(axis=1)
    final_ret = (equity[:, -1] - initial_capital) / initial_capital

    return {
        "n_simulations": n_sims,
        "n_trades": int(pnls.size),
        "p05_return": round(float(np.percentile(final_ret, 5)), 6),
        "p50_return": round(float(np.percentile(final_ret, 50)), 6),
        "p95_return": round(float(np.percentile(final_ret, 95)), 6),
        "mean_return": round(float(np.mean(final_ret)), 6),
        "std_return": round(float(np.std(final_ret, ddof=1)), 6),
        "mean_max_dd": round(float(np.mean(max_dd)), 6),
        "p95_max_dd": round(float(np.percentile(max_dd, 95)), 6),
    }


# ── Cross-asset validation ─────────────────────────────────────────────────

def run_cross_asset(
    strategy_name: str,
    symbols: list[str],
    loader: FuturesDataLoader,
    timeframe: str = "5m",
    max_bars: int = 100_000,
) -> dict[str, Any]:
    """Run strategy on multiple symbols, compute median sharpe."""
    _ensure_strategy_registered(strategy_name)
    strategy_cls = StrategyRegistry.get(strategy_name)
    if strategy_cls is None:
        return {"error": f"strategy {strategy_name} not found"}

    per_symbol: list[dict] = []
    for sym in symbols:
        bars = _load_bars(loader, sym, timeframe)
        if max_bars and len(bars) > max_bars:
            bars = bars.tail(max_bars).reset_index(drop=True)
        if bars.empty or len(bars) < 50:
            per_symbol.append({"symbol": sym, "error": "no data or too few bars"})
            continue

        config = StrategyConfig(name=strategy_name, strategy_id=f"{strategy_name}_xasset_{sym}")
        strat = strategy_cls(config)
        result = backtest_on_bars(strat, bars)
        per_symbol.append({
            "symbol": sym,
            "sharpe": result["sharpe"],
            "total_return": result["total_return"],
            "trades": result["trades"],
            "max_dd": result["max_dd"],
        })

    valid = [s for s in per_symbol if "error" not in s]
    sharpes = [s["sharpe"] for s in valid]
    median_sharpe = float(np.median(sharpes)) if sharpes else 0.0

    return {
        "symbols_tested": [s["symbol"] for s in per_symbol],
        "symbols_valid": len(valid),
        "median_sharpe": round(median_sharpe, 4),
        "per_symbol": per_symbol,
    }


# ── Gate judgment ───────────────────────────────────────────────────────────

def judge_gate(
    oos_result: dict,
    wf_result: dict,
    mc_result: dict,
    xasset_result: dict,
    train_trades: int,
) -> dict[str, Any]:
    """Apply gate thresholds. Returns pass/reject + reasons."""
    checks: list[dict] = []

    oos = oos_result.get("oos", {})
    oos_sharpe = oos.get("sharpe", 0)
    oos_return = oos.get("total_return", 0)
    checks.append({
        "criterion": "oos.sharpe >= 0.8",
        "value": oos_sharpe,
        "threshold": GATE_THRESHOLDS["oos_sharpe"],
        "pass": oos_sharpe >= GATE_THRESHOLDS["oos_sharpe"],
    })
    checks.append({
        "criterion": "oos.return > 0",
        "value": oos_return,
        "threshold": GATE_THRESHOLDS["oos_return_min"],
        "pass": oos_return > GATE_THRESHOLDS["oos_return_min"],
    })

    wf_consistency = wf_result.get("consistency", 0)
    checks.append({
        "criterion": "wf.consistency >= 0.6",
        "value": wf_consistency,
        "threshold": GATE_THRESHOLDS["wf_consistency"],
        "pass": wf_consistency >= GATE_THRESHOLDS["wf_consistency"],
    })

    mc_p05 = mc_result.get("p05_return", -1)
    checks.append({
        "criterion": "mc.p05_return >= -0.15",
        "value": mc_p05,
        "threshold": GATE_THRESHOLDS["mc_p05_return"],
        "pass": mc_p05 >= GATE_THRESHOLDS["mc_p05_return"],
    })

    xa_median = xasset_result.get("median_sharpe", 0)
    checks.append({
        "criterion": "x_asset.median_sharpe >= 0.5",
        "value": xa_median,
        "threshold": GATE_THRESHOLDS["x_asset_median_sharpe"],
        "pass": xa_median >= GATE_THRESHOLDS["x_asset_median_sharpe"],
    })

    checks.append({
        "criterion": "train.trades >= 30",
        "value": train_trades,
        "threshold": GATE_THRESHOLDS["train_trades_min"],
        "pass": train_trades >= GATE_THRESHOLDS["train_trades_min"],
    })

    all_pass = all(c["pass"] for c in checks)
    failures = [c for c in checks if not c["pass"]]

    return {
        "outcome": "pass" if all_pass else "reject",
        "checks": checks,
        "failures": [c["criterion"] for c in failures],
        "summary": f"{'PASS' if all_pass else 'REJECT'}: {len(checks) - len(failures)}/{len(checks)} criteria met",
    }


# ── Data loading helper ────────────────────────────────────────────────────

def _load_bars(loader: FuturesDataLoader, symbol: str, timeframe: str) -> pd.DataFrame:
    bars = loader.load_bars(symbol, timeframe, cache_dir=CACHE_DIR)
    if bars.empty:
        bars = loader.load_main_contract_bars(symbol, timeframe, cache_dir=CACHE_DIR)
    if "instrument" in bars.columns and bars["instrument"].nunique() > 1:
        bars = _filter_main_contract(bars)
    return bars


def _filter_main_contract(bars: pd.DataFrame) -> pd.DataFrame:
    """Keep only the most-liquid contract per time slot (by volume)."""
    if "instrument" not in bars.columns:
        return bars
    vol_col = "volume" if "volume" in bars.columns else None
    if vol_col is None:
        instruments = bars["instrument"].unique()
        main = max(instruments, key=lambda x: len(bars[bars["instrument"] == x]))
        return bars[bars["instrument"] == main].reset_index(drop=True)

    vol_by_inst = bars.groupby("instrument")[vol_col].sum()
    main_inst = vol_by_inst.idxmax()
    filtered = bars[bars["instrument"] == main_inst].reset_index(drop=True)
    logger.info("Filtered to main contract %s (%d bars, %.0f%% of volume)",
                main_inst, len(filtered),
                vol_by_inst[main_inst] / vol_by_inst.sum() * 100)
    return filtered


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="2号位 Gate Validation Pipeline")
    parser.add_argument("--strategy", required=True, help="Strategy name to validate")
    parser.add_argument("--symbols", nargs="+", default=["rb", "cu", "ag", "m", "i", "IF", "MA", "SA"],
                        help="Symbols for validation")
    parser.add_argument("--primary-symbol", default=None, help="Primary symbol for OOS/WF (default: first in list)")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--oos-ratio", type=float, default=0.3, help="OOS split ratio")
    parser.add_argument("--wf-folds", type=int, default=5, help="Walk-forward folds")
    parser.add_argument("--mc-sims", type=int, default=1000, help="Monte Carlo simulations")
    parser.add_argument("--full-gate", action="store_true", help="Run all gate checks")
    parser.add_argument("--max-bars", type=int, default=200_000,
                        help="Max bars to use (tail-sample). 0 = unlimited")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    args = parser.parse_args()

    _ensure_strategy_registered(args.strategy)

    primary_sym = args.primary_symbol or args.symbols[0]
    loader = FuturesDataLoader()

    logger.info("=" * 70)
    logger.info("GATE VALIDATION: %s", args.strategy)
    logger.info("Primary symbol: %s | All symbols: %s", primary_sym, args.symbols)
    logger.info("=" * 70)

    primary_bars = _load_bars(loader, primary_sym, args.timeframe)
    if primary_bars.empty:
        logger.error("No data for primary symbol %s", primary_sym)
        sys.exit(1)

    if args.max_bars and len(primary_bars) > args.max_bars:
        logger.info("Truncating %d bars to last %d (--max-bars)", len(primary_bars), args.max_bars)
        primary_bars = primary_bars.tail(args.max_bars).reset_index(drop=True)

    logger.info("Loaded %d bars for %s", len(primary_bars), primary_sym)

    # ── 1. OOS ──
    logger.info("\n[1/4] Running OOS validation...")
    t0 = time.time()
    oos_result = run_oos(args.strategy, primary_sym, primary_bars, args.oos_ratio)
    logger.info("  OOS done in %.1fs", time.time() - t0)
    if "error" in oos_result:
        logger.error("  OOS ERROR: %s", oos_result["error"])
    else:
        oos = oos_result["oos"]
        logger.info("  OOS sharpe=%.4f return=%.6f trades=%d", oos["sharpe"], oos["total_return"], oos["trades"])

    # ── 2. Walk-Forward ──
    logger.info("\n[2/4] Running Walk-Forward...")
    t0 = time.time()
    wf_result = run_walk_forward(args.strategy, primary_sym, primary_bars, args.wf_folds)
    logger.info("  WF done in %.1fs", time.time() - t0)
    if "error" in wf_result:
        logger.error("  WF ERROR: %s", wf_result["error"])
    else:
        logger.info("  WF consistency=%.4f median_sharpe=%.4f folds=%d/%d positive",
                     wf_result["consistency"], wf_result["median_sharpe"],
                     wf_result["positive_folds"], wf_result["completed_folds"])

    # ── 3. Monte Carlo ──
    logger.info("\n[3/4] Running Monte Carlo...")
    t0 = time.time()
    pnl_series = oos_result.get("oos_pnl_series", [])
    if len(pnl_series) < 3:
        if "train" in oos_result:
            config = StrategyConfig(name=args.strategy, strategy_id=f"{args.strategy}_mc_source")
            strategy_cls = StrategyRegistry.get(args.strategy)
            if strategy_cls:
                strat = strategy_cls(config)
                full_result = backtest_on_bars(strat, primary_bars)
                pnl_series = full_result.get("pnl_series", [])

    mc_result = run_monte_carlo(pnl_series, n_sims=args.mc_sims)
    logger.info("  MC done in %.1fs", time.time() - t0)
    if "error" in mc_result:
        logger.error("  MC ERROR: %s", mc_result["error"])
    else:
        logger.info("  MC p05=%.6f p50=%.6f p95=%.6f mean_max_dd=%.4f",
                     mc_result["p05_return"], mc_result["p50_return"],
                     mc_result["p95_return"], mc_result["mean_max_dd"])

    # ── 4. Cross-asset ──
    logger.info("\n[4/4] Running Cross-asset validation...")
    t0 = time.time()
    xasset_result = run_cross_asset(args.strategy, args.symbols, loader, args.timeframe,
                                     max_bars=args.max_bars or 100_000)
    logger.info("  X-asset done in %.1fs", time.time() - t0)
    if "error" in xasset_result:
        logger.error("  X-asset ERROR: %s", xasset_result["error"])
    else:
        logger.info("  X-asset median_sharpe=%.4f (%d/%d symbols valid)",
                     xasset_result["median_sharpe"],
                     xasset_result["symbols_valid"], len(xasset_result["symbols_tested"]))

    # ── Gate judgment ──
    train_trades = oos_result.get("train", {}).get("trades", 0) if "error" not in oos_result else 0
    gate = judge_gate(oos_result, wf_result, mc_result, xasset_result, train_trades)

    logger.info("\n" + "=" * 70)
    logger.info("GATE RESULT: %s", gate["summary"])
    for c in gate["checks"]:
        status = "✓" if c["pass"] else "✗"
        logger.info("  %s %s: %.4f (threshold: %s)", status, c["criterion"], c["value"], c["threshold"])
    logger.info("=" * 70)

    # ── Save results ──
    out_dir = PROJ_DIR / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().isoformat()
    name = args.strategy

    def _save(suffix: str, data: dict) -> None:
        path = out_dir / f"{name}_{suffix}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Saved %s", path)

    oos_out = {k: v for k, v in oos_result.items() if k != "oos_pnl_series"}
    _save("oos", {"_meta": {"generated_at": ts, "script": "validate_gate.py"}, **oos_out})
    _save("wf", {"_meta": {"generated_at": ts, "script": "validate_gate.py"}, **wf_result})
    _save("mc", {"_meta": {"generated_at": ts, "script": "validate_gate.py"}, **mc_result})
    _save("gate", {
        "_meta": {"generated_at": ts, "script": "validate_gate.py"},
        "strategy": name,
        "primary_symbol": primary_sym,
        "symbols": args.symbols,
        "gate": gate,
        "thresholds": GATE_THRESHOLDS,
    })


if __name__ == "__main__":
    main()
