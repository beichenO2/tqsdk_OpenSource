"""Run all frontier strategies and produce comparison report.

Usage:
    python scripts/research/frontier/run_all.py [--symbol BTCUSDT] [--weeks 80]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))


def run_tft(args):
    logger.info("=" * 70)
    logger.info("STRATEGY 1/3: Temporal Fusion Transformer (TFT)")
    logger.info("=" * 70)
    try:
        from tft_strategy import load_crypto_bars, build_tft_dataset, train_tft, backtest_tft
        bars = load_crypto_bars(args.symbol, args.timeframe, args.weeks)
        split_idx = int(len(bars) * 0.7)
        train_bars = bars.iloc[:split_idx]
        test_bars = bars.iloc[split_idx:]

        X, y, _, feature_cols = build_tft_dataset(train_bars, seq_len=30)
        model, mean, std, val_acc = train_tft(X, y, n_features=len(feature_cols), epochs=args.epochs)
        results = backtest_tft(test_bars, model, mean, std, feature_cols, leverage=args.leverage)
        results["val_accuracy"] = round(val_acc, 4)
        return results
    except Exception as e:
        logger.error("TFT failed: %s", e, exc_info=True)
        return {"error": str(e)}


def run_sac(args):
    logger.info("=" * 70)
    logger.info("STRATEGY 2/3: Soft Actor-Critic (SAC) RL")
    logger.info("=" * 70)
    try:
        from sac_rl_strategy import load_crypto_bars, train_sac_agent, backtest_sac
        bars = load_crypto_bars(args.symbol, args.timeframe, args.weeks)
        split_idx = int(len(bars) * 0.7)
        train_bars = bars.iloc[:split_idx].reset_index(drop=True)
        test_bars = bars.iloc[split_idx:].reset_index(drop=True)

        model, _ = train_sac_agent(train_bars, total_timesteps=args.timesteps)
        results = backtest_sac(test_bars, model, leverage=args.leverage)
        return results
    except Exception as e:
        logger.error("SAC failed: %s", e, exc_info=True)
        return {"error": str(e)}


def run_hybrid(args):
    logger.info("=" * 70)
    logger.info("STRATEGY 3/3: CNN-BiLSTM-Attention Hybrid")
    logger.info("=" * 70)
    try:
        from cnn_bilstm_attention_strategy import (
            load_crypto_bars, build_dataset, train_model, backtest_hybrid,
        )
        bars = load_crypto_bars(args.symbol, args.timeframe, args.weeks)
        split_idx = int(len(bars) * 0.7)
        train_bars = bars.iloc[:split_idx]
        test_bars = bars.iloc[split_idx:]

        X, y_dir, y_regime, feature_cols = build_dataset(train_bars, seq_len=30)
        model, mean, std, val_acc = train_model(X, y_dir, y_regime,
                                                  n_features=len(feature_cols), epochs=args.epochs)
        results = backtest_hybrid(test_bars, model, mean, std, feature_cols, leverage=args.leverage)
        results["val_accuracy"] = round(val_acc, 4)
        return results
    except Exception as e:
        logger.error("Hybrid failed: %s", e, exc_info=True)
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Run all frontier strategies")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--weeks", type=int, default=80)
    parser.add_argument("--leverage", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--timesteps", type=int, default=100_000)
    args = parser.parse_args()

    results = {}

    results["TFT"] = run_tft(args)
    results["SAC_RL"] = run_sac(args)
    results["CNN_BiLSTM_Attention"] = run_hybrid(args)

    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON REPORT")
    logger.info("=" * 70)

    key_metrics = ["total_return_pct", "sharpe", "calmar", "win_rate",
                   "profit_factor", "max_drawdown_pct", "n_trades"]

    header = f"{'Metric':<25}" + "".join(f"{name:<22}" for name in results.keys())
    logger.info(header)
    logger.info("-" * len(header))

    for metric in key_metrics:
        row = f"{metric:<25}"
        for name, res in results.items():
            val = res.get(metric, "N/A")
            row += f"{str(val):<22}"
        logger.info(row)

    out_dir = Path(__file__).resolve().parent.parent.parent.parent / "models" / "frontier"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "comparison_report.json"
    with open(out_path, "w") as f:
        json.dump({
            "symbol": args.symbol, "timeframe": args.timeframe,
            "leverage": args.leverage, "results": results,
        }, f, indent=2)
    logger.info("\nFull report saved to %s", out_path)


if __name__ == "__main__":
    main()
