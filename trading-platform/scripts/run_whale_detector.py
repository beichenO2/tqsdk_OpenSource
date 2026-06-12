#!/usr/bin/env python3
"""庄家识别策略 — 训练 + 回测脚本。

用法:
  cd ~/Polarisor/tqsdk/trading-platform
  .venv/bin/python3 scripts/run_whale_detector.py
  .venv/bin/python3 scripts/run_whale_detector.py --symbols MA rb i cu --mode train+backtest

流程:
  1. 加载期货5m数据 (含OI)
  2. Phase 1: 特征提取 (feature_export=True)
  3. Phase 2: 无监督异常检测训练 (IsolationForest)
  4. Phase 3: 回测 (默认参数 + 训练后标签)
  5. 输出报告
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_DIR / "packages"))

from strategy.base import StrategyConfig
from strategy.futures.whale_detector import (
    WhaleDetectorStrategy,
    WhalePhase,
    OPTUNA_PARAM_SPACE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("whale-detector")

CACHE_DIR = PROJ_DIR / "data" / "futures_cache"
RESULTS_DIR = PROJ_DIR / "results"

# 选择有 OI 数据的品种
DEFAULT_SYMBOLS = ["MA", "rb", "i", "cu", "ag", "IF", "SA", "m"]


def load_futures_data(symbol: str, timeframe: str = "5m") -> pd.DataFrame | None:
    """加载期货数据，适配 close_oi → open_interest 字段。"""
    pattern = f"KQ_m_*_{symbol}_{timeframe}.parquet"
    files = list(CACHE_DIR.glob(pattern))
    if not files:
        pattern2 = f"*{symbol}*{timeframe}*.parquet"
        files = list(CACHE_DIR.glob(pattern2))
    if not files:
        logger.warning(f"No data found for {symbol} ({timeframe})")
        return None

    df = pd.read_parquet(files[0])
    if "close_oi" in df.columns:
        df["open_interest"] = df["close_oi"]
    elif "oi" not in df.columns and "open_interest" not in df.columns:
        logger.warning(f"{symbol}: no OI column found, skipping")
        return None

    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            logger.warning(f"{symbol}: missing column {col}")
            return None

    df = df.dropna(subset=required)
    logger.info(f"Loaded {symbol}: {len(df)} bars, OI range: {df.get('open_interest', df.get('close_oi', pd.Series())).min():.0f} - {df.get('open_interest', df.get('close_oi', pd.Series())).max():.0f}")
    return df


def run_feature_extraction(symbol: str, df: pd.DataFrame) -> WhaleDetectorStrategy:
    """Phase 1: 用策略跑一遍数据，提取特征。"""
    config = StrategyConfig(
        name="whale_detector",
        symbols=[symbol],
        params={"feature_export": True},
    )
    strategy = WhaleDetectorStrategy(config)
    loop = asyncio.new_event_loop()

    for _, row in df.iterrows():
        bar = {
            "datetime": str(row.get("datetime", "")),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "open_interest": float(row.get("open_interest", row.get("close_oi", 0))),
        }
        loop.run_until_complete(strategy.on_bar(symbol, bar))

    loop.close()
    logger.info(f"  Features extracted: {len(strategy.feature_history)} rows, {strategy.get_features_array().shape}")
    return strategy


def train_anomaly_detector(features: np.ndarray) -> tuple[np.ndarray, dict]:
    """Phase 2: 无监督异常检测训练。

    使用 Z-score + Percentile 方法 (不需要 sklearn)
    标记 top 10% 异常分数的 bar 为潜在庄家活动。
    """
    if len(features) < 50:
        return np.full(len(features), -1, dtype=int), {"method": "skip", "reason": "too_few_samples"}

    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0)
    stds[stds < 1e-10] = 1.0

    z_scores = np.abs((features - means) / stds)
    anomaly_scores = np.mean(z_scores, axis=1)

    threshold_90 = np.percentile(anomaly_scores, 90)
    threshold_75 = np.percentile(anomaly_scores, 75)

    labels = np.zeros(len(features), dtype=int)
    labels[anomaly_scores > threshold_90] = 1  # strong whale signal
    labels[(anomaly_scores > threshold_75) & (anomaly_scores <= threshold_90)] = 2  # moderate

    stats = {
        "method": "zscore_percentile",
        "n_samples": len(features),
        "n_strong_whale": int(np.sum(labels == 1)),
        "n_moderate": int(np.sum(labels == 2)),
        "n_normal": int(np.sum(labels == 0)),
        "threshold_90": float(threshold_90),
        "threshold_75": float(threshold_75),
        "mean_anomaly_score": float(np.mean(anomaly_scores)),
        "max_anomaly_score": float(np.max(anomaly_scores)),
    }
    return labels, stats


def run_backtest(
    symbol: str,
    df: pd.DataFrame,
    params: dict | None = None,
    initial_capital: float = 100_000.0,
) -> dict:
    """Phase 3: 回测。"""
    config = StrategyConfig(
        name="whale_detector",
        symbols=[symbol],
        params={**(params or {}), "feature_export": True},
    )
    strategy = WhaleDetectorStrategy(config)
    loop = asyncio.new_event_loop()

    capital = initial_capital
    position = None
    entry_price = 0.0
    trades = []
    equity_curve = [capital]
    commission = 0.00005
    slippage = 0.0001

    for _, row in df.iterrows():
        bar = {
            "datetime": str(row.get("datetime", "")),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "open_interest": float(row.get("open_interest", row.get("close_oi", 0))),
        }

        signals = loop.run_until_complete(strategy.on_bar(symbol, bar))
        close_price = bar["close"]

        for sig in signals:
            from strategy.base import SignalType
            if sig.signal_type == SignalType.LONG_ENTRY and position is None:
                position = "long"
                entry_price = close_price
                capital *= (1 - commission - slippage)
            elif sig.signal_type == SignalType.SHORT_ENTRY and position is None:
                position = "short"
                entry_price = close_price
                capital *= (1 - commission - slippage)
            elif sig.signal_type == SignalType.LONG_EXIT and position == "long":
                pnl_pct = (close_price - entry_price) / entry_price
                capital *= (1 + pnl_pct) * (1 - commission - slippage)
                trades.append({
                    "pnl_pct": pnl_pct, "side": "long",
                    "reason": sig.reason,
                    "phase": sig.metadata.get("phase", ""),
                })
                position = None
            elif sig.signal_type == SignalType.SHORT_EXIT and position == "short":
                pnl_pct = (entry_price - close_price) / entry_price
                capital *= (1 + pnl_pct) * (1 - commission - slippage)
                trades.append({
                    "pnl_pct": pnl_pct, "side": "short",
                    "reason": sig.reason,
                    "phase": sig.metadata.get("phase", ""),
                })
                position = None

        equity_curve.append(capital)

    if position and len(df) > 0:
        last_close = float(df.iloc[-1]["close"])
        pnl_pct = ((last_close - entry_price) / entry_price if position == "long"
                    else (entry_price - last_close) / entry_price)
        capital *= (1 + pnl_pct) * (1 - commission - slippage)
        trades.append({"pnl_pct": pnl_pct, "side": position, "reason": "end_of_data", "phase": ""})

    loop.close()

    total_return = (capital - initial_capital) / initial_capital
    n_trades = len(trades)
    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    win_rate = len(wins) / n_trades if n_trades else 0.0

    equity = np.array(equity_curve)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

    step = max(1, len(equity) // 52)
    weekly_returns = [(equity[i] - equity[i - step]) / equity[i - step]
                      for i in range(step, len(equity), step) if equity[i - step] > 0]
    sharpe = float(np.mean(weekly_returns) / max(np.std(weekly_returns), 1e-10) * np.sqrt(52)) if len(weekly_returns) >= 2 else 0.0

    avg_win = float(np.mean([t["pnl_pct"] for t in wins])) if wins else 0.0
    avg_loss = float(np.mean([abs(t["pnl_pct"]) for t in losses])) if losses else 0.0
    pf_num = sum(t["pnl_pct"] for t in wins) if wins else 0.0
    pf_den = sum(abs(t["pnl_pct"]) for t in losses) if losses else 1e-10
    profit_factor = pf_num / pf_den if pf_den > 0 else (10.0 if wins else 0.0)

    phase_stats = {}
    for t in trades:
        ph = t.get("phase", "unknown") or "unknown"
        if ph not in phase_stats:
            phase_stats[ph] = {"count": 0, "wins": 0, "total_pnl": 0.0}
        phase_stats[ph]["count"] += 1
        phase_stats[ph]["total_pnl"] += t["pnl_pct"]
        if t["pnl_pct"] > 0:
            phase_stats[ph]["wins"] += 1

    features = strategy.get_features_array()
    feature_stats = {}
    if len(features) > 0:
        col_names = ["oi_zscore", "vol_zscore", "vpd_score", "vr", "spoof_proxy", "oi_vol_mismatch", "composite"]
        for i, name in enumerate(col_names):
            col = features[:, i]
            feature_stats[name] = {
                "mean": float(np.mean(col)),
                "std": float(np.std(col)),
                "max": float(np.max(col)),
                "p95": float(np.percentile(col, 95)),
            }

    return {
        "symbol": symbol,
        "bars": len(df),
        "total_return": round(total_return, 6),
        "sharpe": round(sharpe, 4),
        "max_dd": round(max_dd, 6),
        "trades": n_trades,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "avg_win": round(avg_win, 6),
        "avg_loss": round(avg_loss, 6),
        "final_capital": round(capital, 2),
        "phase_stats": phase_stats,
        "feature_stats": feature_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Whale Detector: Train + Backtest")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--mode", default="train+backtest", choices=["train", "backtest", "train+backtest"])
    parser.add_argument("--output", default="results/whale_detector_report.json")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {"mode": args.mode, "symbols": {}, "training": {}}

    for symbol in args.symbols:
        logger.info(f"{'='*60}")
        logger.info(f"Processing {symbol}")
        logger.info(f"{'='*60}")

        df = load_futures_data(symbol)
        if df is None or len(df) < 100:
            logger.warning(f"Skip {symbol}: insufficient data")
            continue

        if "train" in args.mode:
            logger.info(f"[{symbol}] Phase 1: Feature extraction...")
            strategy = run_feature_extraction(symbol, df)
            features = strategy.get_features_array()

            if len(features) > 0:
                logger.info(f"[{symbol}] Phase 2: Anomaly detection training...")
                labels, train_stats = train_anomaly_detector(features)
                results["training"][symbol] = train_stats
                logger.info(f"  Strong whale signals: {train_stats.get('n_strong_whale', 0)}")
                logger.info(f"  Moderate signals: {train_stats.get('n_moderate', 0)}")
                logger.info(f"  Normal bars: {train_stats.get('n_normal', 0)}")

        if "backtest" in args.mode:
            logger.info(f"[{symbol}] Phase 3: Backtesting...")
            bt_result = run_backtest(symbol, df)
            results["symbols"][symbol] = bt_result

            logger.info(f"  Return: {bt_result['total_return']*100:.2f}%")
            logger.info(f"  Sharpe: {bt_result['sharpe']:.4f}")
            logger.info(f"  MaxDD:  {bt_result['max_dd']*100:.2f}%")
            logger.info(f"  Trades: {bt_result['trades']} (Win: {bt_result['win_rate']*100:.1f}%)")
            logger.info(f"  PF:     {bt_result['profit_factor']:.2f}")
            if bt_result.get("phase_stats"):
                logger.info(f"  Phase breakdown:")
                for ph, st in bt_result["phase_stats"].items():
                    wr = st["wins"] / st["count"] * 100 if st["count"] > 0 else 0
                    logger.info(f"    {ph}: {st['count']} trades, {wr:.0f}% win, pnl={st['total_pnl']*100:.2f}%")

    if results["symbols"]:
        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY")
        logger.info(f"{'='*60}")
        for sym, r in results["symbols"].items():
            logger.info(f"  {sym:>6s}: Ret={r['total_return']*100:+7.2f}%  Sharpe={r['sharpe']:+.3f}  DD={r['max_dd']*100:.1f}%  Trades={r['trades']:>3d}  WR={r['win_rate']*100:.0f}%  PF={r['profit_factor']:.2f}")

    output_path = PROJ_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nReport saved to {output_path}")

    # 同步写入 latest 指针文件，供 Agent / 文档查询时获取最新结果
    import datetime as _dt
    latest_payload = {
        "_meta": {
            "source_file": str(output_path.name),
            "generated_at": _dt.datetime.now().isoformat(),
            "description": "此文件由 run_whale_detector.py 每次运行后自动覆写，始终代表最新一次回测结果。",
        },
        **results,
    }
    latest_path = RESULTS_DIR / "whale_detector_latest.json"
    with open(latest_path, "w") as f:
        json.dump(latest_payload, f, indent=2, default=str)
    logger.info(f"Latest pointer updated → {latest_path}")


if __name__ == "__main__":
    main()
