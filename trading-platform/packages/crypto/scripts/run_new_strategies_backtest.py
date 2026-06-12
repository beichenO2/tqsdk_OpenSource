"""Backtest all new crypto strategies on BTC 4h data.

Usage: python3 packages/crypto/scripts/run_new_strategies_backtest.py
Output: results printed to stdout + JSON saved to output/
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

from strategy.base import StrategyConfig

STRATEGIES = [
    ("crypto.strategies.regime_adaptive", "RegimeAdaptiveStrategy", "regime_adaptive"),
    ("crypto.strategies.trend_following_v2", "TrendFollowingV2Strategy", "trend_following_v2"),
    ("crypto.strategies.adaptive_trend", "AdaptiveTrendStrategy", "adaptive_trend"),
    ("crypto.strategies.vwap_reversion", "VWAPReversionStrategy", "vwap_reversion"),
    ("crypto.strategies.wyckoff_phases", "WyckoffPhasesStrategy", "wyckoff_phases"),
    ("crypto.strategies.liquidation_reversal", "LiquidationReversalStrategy", "liquidation_reversal"),
    ("crypto.strategies.funding_rate_v2", "FundingRateV2Strategy", "funding_rate_v2"),
    ("crypto.strategies.grid_v2", "GridV2Strategy", "grid_v2"),
    ("crypto.strategies.volume_profile_flow", "VolumeProfileFlowStrategy", "volume_profile_flow"),
    ("crypto.strategies.hurst_regime_switch", "HurstRegimeSwitchStrategy", "hurst_regime_switch"),
    ("crypto.strategies.smart_money_fvg", "SmartMoneyFVGStrategy", "smart_money_fvg"),
    ("crypto.strategies.taker_imbalance", "TakerImbalanceStrategy", "taker_imbalance"),
    ("crypto.strategies.ichimoku_cloud", "IchimokuCloudStrategy", "ichimoku_cloud"),
    ("crypto.strategies.crypto_pairs", "CryptoPairsStrategy", "crypto_pairs"),
    ("crypto.strategies.squeeze_breakout", "SqueezeBreakoutStrategy", "squeeze_breakout"),
    ("crypto.strategies.funding_rate_alpha", "FundingRateAlphaStrategy", "funding_rate_alpha"),
    ("crypto.strategies.scalp_momentum", "ScalpMomentumStrategy", "scalp_momentum"),
    ("crypto.strategies.cross_sectional_momentum", "TimeSeriesMomentumStrategy", "time_series_momentum"),
    ("crypto.strategies.volatility_breakout_scalp", "VolatilityBreakoutScalpStrategy", "vol_breakout_scalp"),
    ("crypto.strategies.funding_meta_ensemble", "FundingMetaEnsembleStrategy", "funding_meta_ensemble"),
    ("crypto.strategies.ensemble_strategy", "EnsembleStrategy", "ensemble"),
    ("crypto.strategies.funding_rate_arb", "FundingRateArbitrage", "funding_rate_arb"),
    ("crypto.strategies.ou_mean_reversion", "OUMeanReversionStrategy", "ou_mean_reversion"),
    ("crypto.strategies.kalman_trend", "KalmanTrendStrategy", "kalman_trend"),
    ("crypto.strategies.mtf_confluence", "MTFConfluenceStrategy", "mtf_confluence"),
    ("crypto.strategies.momentum_rotation", "MomentumRotationStrategy", "momentum_rotation"),
    ("crypto.strategies.rsi_divergence", "RSIDivergenceStrategy", "rsi_divergence"),
    ("crypto.strategies.keltner_pullback", "KeltnerPullbackStrategy", "keltner_pullback"),
    ("crypto.strategies.fibonacci_pullback", "FibonacciPullbackStrategy", "fibonacci_pullback"),
    ("crypto.strategies.supertrend", "SupertrendStrategy", "supertrend"),
    ("crypto.strategies.dual_momentum", "DualMomentumStrategy", "dual_momentum"),
    ("crypto.strategies.range_breakout", "RangeBreakoutStrategy", "range_breakout"),
    ("crypto.strategies.extreme_reversal", "ExtremeReversalStrategy", "extreme_reversal"),
    ("crypto.strategies.engulfing_pattern", "EngulfingPatternStrategy", "engulfing_pattern"),
    ("crypto.strategies.session_momentum", "SessionMomentumStrategy", "session_momentum"),
]

DATA_DIR = Path.home() / "Downloads" / "crypto_data"
OUTPUT_DIR = Path(__file__).resolve().parents[3] / "output" / "new_strategy_backtest"
SYMBOL = "BTCUSDT"
TIMEFRAME = "4h"
INITIAL_CAPITAL = 10000.0
COMMISSION = 0.0004
SLIPPAGE = 0.0003


def load_data(symbol: str, timeframe: str) -> pd.DataFrame:
    path = DATA_DIR / symbol.lower() / f"{timeframe}.parquet"
    if not path.exists():
        print(f"Data not found: {path}")
        sys.exit(1)
    df = pd.read_parquet(path)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df


async def run_simple_backtest(strategy_instance, bars: pd.DataFrame, symbol: str) -> dict:
    """Simple bar-by-bar backtest with position tracking."""
    capital = INITIAL_CAPITAL
    position = 0.0
    entry_price = 0.0
    trades = []
    equity_curve = [capital]
    peak = capital

    await strategy_instance.on_start()

    for i in range(len(bars)):
        row = bars.iloc[i]
        bar = {
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row.get("volume", 0)),
        }
        if "taker_buy_volume" in row:
            bar["taker_buy_volume"] = float(row["taker_buy_volume"])
        if "funding_rate" in row:
            bar["funding_rate"] = float(row["funding_rate"])

        signals = await strategy_instance.on_bar(symbol, bar)

        for sig in signals:
            price = bar["close"]
            cost = price * COMMISSION + price * SLIPPAGE

            if sig.signal_type.value in ("long_entry", "short_entry") and position == 0:
                pos_fraction = sig.metadata.get("position_fraction", 0.95)
                qty = capital * pos_fraction / price
                position = qty if "long" in sig.signal_type.value else -qty
                entry_price = price + cost
                strategy_instance.update_position(
                    __import__("strategy.base", fromlist=["Position"]).Position(
                        symbol=symbol,
                        side=__import__("strategy.base", fromlist=["OrderSide"]).OrderSide.BUY
                        if position > 0
                        else __import__("strategy.base", fromlist=["OrderSide"]).OrderSide.SELL,
                        qty=abs(position),
                        avg_price=entry_price,
                    )
                )

            elif sig.signal_type.value in ("long_exit", "short_exit") and position != 0:
                exit_price = price - cost if position > 0 else price + cost
                pnl = (exit_price - entry_price) * position
                capital += pnl
                trades.append({
                    "entry": entry_price, "exit": exit_price,
                    "pnl": pnl, "side": "long" if position > 0 else "short",
                    "bars_held": 0,
                })
                position = 0
                entry_price = 0
                strategy_instance.remove_position(symbol)

        if position != 0:
            mtm = (bar["close"] - entry_price) * position
            equity_curve.append(capital + mtm)
        else:
            equity_curve.append(capital)

        peak = max(peak, equity_curve[-1])

    if position != 0:
        close_price = float(bars.iloc[-1]["close"])
        pnl = (close_price - entry_price) * position
        capital += pnl
        trades.append({"entry": entry_price, "exit": close_price, "pnl": pnl})

    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    max_dd = 0
    peak_eq = equity_curve[0]
    for eq in equity_curve:
        peak_eq = max(peak_eq, eq)
        dd = (peak_eq - eq) / peak_eq if peak_eq > 0 else 0
        max_dd = max(max_dd, dd)

    winning = [t for t in trades if t["pnl"] > 0]
    win_rate = len(winning) / len(trades) if trades else 0

    returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
    returns = returns[np.isfinite(returns)]
    sharpe = 0
    if len(returns) > 10:
        sharpe = float(np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(365 * 6))

    return {
        "total_return_pct": round(total_return * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "total_trades": len(trades),
        "win_rate": round(win_rate * 100, 1),
        "final_capital": round(capital, 2),
    }


async def main():
    print(f"Loading {SYMBOL} {TIMEFRAME} data...")
    bars = load_data(SYMBOL, TIMEFRAME)
    print(f"Loaded {len(bars)} bars")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for mod_name, cls_name, strategy_name in STRATEGIES:
        print(f"\n{'='*60}")
        print(f"Running: {strategy_name}")
        print(f"{'='*60}")

        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
            config = StrategyConfig(name=strategy_name, symbols=[SYMBOL])
            instance = cls(config)

            t0 = time.time()
            result = await run_simple_backtest(instance, bars, SYMBOL)
            elapsed = time.time() - t0

            result["elapsed_sec"] = round(elapsed, 1)
            results[strategy_name] = result

            print(f"  Return: {result['total_return_pct']}%")
            print(f"  Sharpe: {result['sharpe']}")
            print(f"  MaxDD: {result['max_drawdown_pct']}%")
            print(f"  Trades: {result['total_trades']}")
            print(f"  WinRate: {result['win_rate']}%")
            print(f"  Time: {elapsed:.1f}s")

        except Exception as e:
            print(f"  ERROR: {e}")
            results[strategy_name] = {"error": str(e)}

    out_path = OUTPUT_DIR / f"backtest_{SYMBOL}_{TIMEFRAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, r in results.items():
        if "error" in r:
            print(f"  {name}: ERROR — {r['error']}")
        else:
            print(f"  {name}: Ret={r['total_return_pct']}% Sharpe={r['sharpe']} DD={r['max_drawdown_pct']}% Trades={r['total_trades']}")


if __name__ == "__main__":
    asyncio.run(main())
