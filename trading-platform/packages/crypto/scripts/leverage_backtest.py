"""High-leverage strategy backtester v4 — cumulative mode with Kelly sizing.

Evaluates strategies on 1-minute BTCUSDT data:
  - Starting capital: $100
  - Period: 40 weeks (~280 days)
  - Kelly criterion position sizing (fractional Kelly)
  - NO weekly margin reset — cumulative compounding
  - Strategies that blow up (equity <= $1) get deleted
  - PnL as % of risked capital (not raw price %), fed into Kelly

Usage:
    python scripts/leverage_backtest.py [--delete]
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages"))

from datahub.crypto_loader import CryptoDataLoader
from strategy.base import Signal, SignalType, StrategyConfig, Position, OrderSide

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TAKER_FEE = 0.0002
SLIPPAGE_BPS = 1
FUNDING_RATE_PER_8H = 0.0001

INITIAL_CAPITAL = 100.0
LEVERAGE_RANGE = [50, 75, 100]
EVAL_TIMEFRAME = "1m"
EVAL_WEEKS = 40
KELLY_FRACTION = 0.5
MIN_KELLY_HISTORY = 10


@dataclass
class TradeRecord:
    entry_time: str
    exit_time: str
    side: str
    entry_price: float
    exit_price: float
    qty: float
    pnl_usd: float
    return_on_risk: float
    leverage_used: int
    position_frac: float
    reason_entry: str
    reason_exit: str


@dataclass
class WeeklySnapshot:
    week: int
    equity: float
    return_pct: float
    trades: int
    wins: int


class CumulativeBacktester:
    """Backtester with cumulative P&L and Kelly position sizing."""

    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        max_leverage: int = 100,
        taker_fee: float = TAKER_FEE,
        slippage_bps: float = SLIPPAGE_BPS,
        kelly_fraction: float = KELLY_FRACTION,
    ) -> None:
        self.initial_capital = initial_capital
        self.max_leverage = max_leverage
        self.taker_fee = taker_fee
        self.slippage_bps = slippage_bps
        self.kelly_fraction = kelly_fraction

    def _calc_kelly(self, trades: list[TradeRecord]) -> float:
        """Kelly fraction from return_on_risk (PnL / risked capital)."""
        if len(trades) < MIN_KELLY_HISTORY:
            return 0.05

        wins = [t for t in trades if t.return_on_risk > 0]
        losses = [t for t in trades if t.return_on_risk <= 0]

        if not losses or not wins:
            return 0.05

        win_rate = len(wins) / len(trades)
        avg_win = np.mean([t.return_on_risk for t in wins])
        avg_loss = abs(np.mean([t.return_on_risk for t in losses]))

        if avg_loss < 1e-10:
            return 0.05

        b = avg_win / avg_loss
        kelly = (win_rate * b - (1 - win_rate)) / b

        kelly = max(kelly, 0.01)
        kelly = min(kelly, 0.4)

        return kelly * self.kelly_fraction

    def _get_stop_pct(self, strategy: Any, price: float) -> float:
        """Get stop distance as fraction of price, using strategy's ATR or min_stop_pct."""
        fee_floor = self.taker_fee * 2 * 2.5
        try:
            atr_val = strategy._calc_atr(strategy.config.symbols[0])
            stop_mult = strategy.get_param("stop_atr_mult")
            min_stop = strategy.get_param("min_stop_pct") / 100.0
            atr_stop = (atr_val * stop_mult) / price if price > 0 else min_stop
            return max(atr_stop, min_stop, fee_floor)
        except Exception:
            try:
                return max(strategy.get_param("min_stop_pct") / 100.0, fee_floor)
            except Exception:
                return max(0.005, fee_floor)

    def _close_position(
        self,
        side: str,
        entry_price: float,
        exit_price: float,
        qty: float,
        risk_capital: float,
        position_fraction: float,
        entry_reason: str,
        exit_reason: str,
        entry_time: str,
        bar_time: str,
    ) -> TradeRecord:
        if side == "long":
            pnl = (exit_price - entry_price) * qty
        else:
            pnl = (entry_price - exit_price) * qty
        fee = abs(exit_price * qty) * self.taker_fee
        net_pnl = pnl - fee
        ror = net_pnl / risk_capital if risk_capital > 0 else 0.0
        return TradeRecord(
            entry_time=entry_time, exit_time=str(bar_time),
            side=side, entry_price=entry_price, exit_price=exit_price,
            qty=qty, pnl_usd=net_pnl, return_on_risk=ror,
            leverage_used=self.max_leverage,
            position_frac=position_fraction,
            reason_entry=entry_reason, reason_exit=exit_reason,
        ), net_pnl, fee

    async def run(
        self,
        strategy: Any,
        bars: pd.DataFrame,
        symbol: str = "BTCUSDT",
    ) -> dict[str, Any]:
        equity = self.initial_capital
        position_qty = 0.0
        position_side: str | None = None
        entry_price = 0.0
        entry_reason = ""
        entry_time = ""
        risk_capital_for_trade = 0.0

        trades: list[TradeRecord] = []
        equity_curve: list[float] = [equity]
        peak_equity = equity
        max_dd = 0.0
        total_fees = 0.0
        total_funding = 0.0
        pending_signals: list[Signal] = []
        bars_since_funding = 0

        weekly_snapshots: list[WeeklySnapshot] = []
        week_start_equity = equity
        week_start_time = None
        week_number = 0
        week_trades = 0
        week_wins = 0

        blown_up = False
        stop_loss_price: float = 0.0
        position_fraction: float = 0.05

        for idx, row in bars.iterrows():
            bar = {
                "open": row["open"], "high": row["high"],
                "low": row["low"], "close": row["close"],
                "volume": row.get("volume", 0),
                "taker_buy_volume": row.get("taker_buy_volume", row.get("volume", 0) * 0.5),
            }

            bar_time = row.get("open_time", None)
            if week_start_time is None and bar_time is not None:
                week_start_time = bar_time

            if bar_time is not None and week_start_time is not None:
                if (bar_time - week_start_time).total_seconds() >= 7 * 86400:
                    weekly_snapshots.append(WeeklySnapshot(
                        week=week_number,
                        equity=equity,
                        return_pct=((equity - week_start_equity) / week_start_equity * 100)
                        if week_start_equity > 0 else 0,
                        trades=week_trades,
                        wins=week_wins,
                    ))
                    week_start_equity = equity
                    week_start_time = bar_time
                    week_number += 1
                    week_trades = 0
                    week_wins = 0

            if equity <= 1.0:
                blown_up = True
                break

            exec_price = bar["open"]

            for sig in pending_signals:
                if equity <= 1.0:
                    break

                if sig.signal_type in (SignalType.LONG_ENTRY, SignalType.SHORT_ENTRY) and position_qty == 0:
                    kelly_frac = self._calc_kelly(trades)
                    position_fraction = kelly_frac
                    risk_capital_for_trade = equity * position_fraction
                    notional = risk_capital_for_trade * self.max_leverage
                    qty = notional / exec_price

                    slip_dir = 1 if sig.signal_type == SignalType.LONG_ENTRY else -1
                    slip = exec_price * (self.slippage_bps / 10000) * slip_dir
                    fill_price = exec_price + slip
                    fee = notional * self.taker_fee
                    equity -= fee
                    total_fees += fee
                    position_qty = qty
                    position_side = "long" if sig.signal_type == SignalType.LONG_ENTRY else "short"
                    entry_price = fill_price
                    entry_reason = sig.reason
                    entry_time = str(bar_time)

                    sl_pct = self._get_stop_pct(strategy, fill_price)
                    if position_side == "long":
                        stop_loss_price = fill_price * (1 - sl_pct)
                    else:
                        stop_loss_price = fill_price * (1 + sl_pct)

                    order_side = OrderSide.BUY if position_side == "long" else OrderSide.SELL
                    strategy.update_position(Position(
                        symbol=symbol, side=order_side,
                        qty=qty, avg_price=fill_price,
                    ))

                elif sig.signal_type == SignalType.LONG_EXIT and position_side == "long":
                    slip = exec_price * (self.slippage_bps / 10000)
                    fill_price = exec_price - slip
                    tr, net_pnl, fee = self._close_position(
                        "long", entry_price, fill_price, position_qty,
                        risk_capital_for_trade, position_fraction,
                        entry_reason, sig.reason, entry_time, str(bar_time))
                    equity += net_pnl
                    total_fees += fee
                    week_trades += 1
                    if net_pnl > 0:
                        week_wins += 1
                    trades.append(tr)
                    strategy.remove_position(symbol)
                    position_qty = 0
                    position_side = None

                elif sig.signal_type == SignalType.SHORT_EXIT and position_side == "short":
                    slip = exec_price * (self.slippage_bps / 10000)
                    fill_price = exec_price + slip
                    tr, net_pnl, fee = self._close_position(
                        "short", entry_price, fill_price, position_qty,
                        risk_capital_for_trade, position_fraction,
                        entry_reason, sig.reason, entry_time, str(bar_time))
                    equity += net_pnl
                    total_fees += fee
                    week_trades += 1
                    if net_pnl > 0:
                        week_wins += 1
                    trades.append(tr)
                    strategy.remove_position(symbol)
                    position_qty = 0
                    position_side = None

            pending_signals.clear()

            if position_side and equity > 0:
                bars_since_funding += 1
                if bars_since_funding % 480 == 0:
                    fn_notional = position_qty * bar["close"]
                    funding = fn_notional * FUNDING_RATE_PER_8H
                    equity -= funding
                    total_funding += funding

            if position_side and equity > 0:
                hit_sl = False
                if position_side == "long" and bar["low"] <= stop_loss_price:
                    tr, net_pnl, fee = self._close_position(
                        "long", entry_price, stop_loss_price, position_qty,
                        risk_capital_for_trade, position_fraction,
                        entry_reason, "SL_HIT", entry_time, str(bar_time))
                    equity += net_pnl
                    total_fees += fee
                    week_trades += 1
                    trades.append(tr)
                    strategy.remove_position(symbol)
                    position_qty = 0
                    position_side = None
                    hit_sl = True

                elif position_side == "short" and bar["high"] >= stop_loss_price:
                    tr, net_pnl, fee = self._close_position(
                        "short", entry_price, stop_loss_price, position_qty,
                        risk_capital_for_trade, position_fraction,
                        entry_reason, "SL_HIT", entry_time, str(bar_time))
                    equity += net_pnl
                    total_fees += fee
                    week_trades += 1
                    trades.append(tr)
                    strategy.remove_position(symbol)
                    position_qty = 0
                    position_side = None
                    hit_sl = True

                if not hit_sl and position_side:
                    if position_side == "long":
                        unrealized = (bar["close"] - entry_price) * position_qty
                    else:
                        unrealized = (entry_price - bar["close"]) * position_qty

                    liq_threshold = risk_capital_for_trade * 0.95
                    if unrealized <= -liq_threshold:
                        equity -= liq_threshold
                        equity = max(equity, 0.01)
                        strategy.remove_position(symbol)
                        position_qty = 0
                        position_side = None
                        if equity <= 1.0:
                            blown_up = True
                            break

            if position_side:
                if position_side == "long":
                    unrealized = (bar["close"] - entry_price) * position_qty
                else:
                    unrealized = (entry_price - bar["close"]) * position_qty
            else:
                unrealized = 0.0

            total_equity = max(equity + unrealized, 0)
            equity_curve.append(total_equity)
            peak_equity = max(peak_equity, total_equity)
            dd = (peak_equity - total_equity) / peak_equity if peak_equity > 0 else 0
            max_dd = max(max_dd, dd)

            try:
                signals = await strategy.on_bar(symbol, bar)
            except Exception:
                signals = []
            pending_signals.extend(signals)

        if week_start_time is not None and equity > 0:
            weekly_snapshots.append(WeeklySnapshot(
                week=week_number, equity=equity,
                return_pct=((equity - week_start_equity) / week_start_equity * 100)
                if week_start_equity > 0 else 0,
                trades=week_trades, wins=week_wins,
            ))

        winning = [t for t in trades if t.pnl_usd > 0]
        losing = [t for t in trades if t.pnl_usd <= 0]

        weekly_returns = [w.return_pct for w in weekly_snapshots]
        avg_weekly_return = np.mean(weekly_returns) if weekly_returns else 0
        median_weekly = float(np.median(weekly_returns)) if weekly_returns else 0

        win_rate = len(winning) / len(trades) * 100 if trades else 0
        win_rors = [t.return_on_risk for t in winning]
        loss_rors = [abs(t.return_on_risk) for t in losing]
        avg_rr = np.mean(win_rors) / np.mean(loss_rors) if loss_rors and win_rors else 0

        total_return = (equity - self.initial_capital) / self.initial_capital * 100
        avg_hold = 0
        if trades:
            hold_bars = []
            for t in trades:
                try:
                    entry_ts = pd.Timestamp(t.entry_time)
                    exit_ts = pd.Timestamp(t.exit_time)
                    hold_bars.append((exit_ts - entry_ts).total_seconds() / 60)
                except Exception:
                    pass
            if hold_bars:
                avg_hold = np.mean(hold_bars)

        return {
            "leverage": self.max_leverage,
            "initial_capital": self.initial_capital,
            "final_equity": round(equity, 2),
            "total_return_pct": round(total_return, 2),
            "total_trades": len(trades),
            "win_rate": round(win_rate, 1),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "total_fees": round(total_fees, 2),
            "total_funding": round(total_funding, 2),
            "blown_up": blown_up,
            "weeks_evaluated": len(weekly_snapshots),
            "avg_weekly_return_pct": round(avg_weekly_return, 2),
            "median_weekly_return_pct": round(median_weekly, 2),
            "weeks_positive": sum(1 for r in weekly_returns if r > 0),
            "weekly_returns": [round(r, 2) for r in weekly_returns],
            "avg_rr": round(avg_rr, 2),
            "avg_hold_min": round(avg_hold, 1),
            "trades_per_week": round(len(trades) / max(len(weekly_snapshots), 1), 1),
        }


async def evaluate_strategy(
    name: str,
    strategy_factory,
    symbol: str,
    bars: pd.DataFrame,
) -> dict[str, Any]:
    best_result = None
    best_leverage = 0

    for lev in LEVERAGE_RANGE:
        strat = strategy_factory()
        bt = CumulativeBacktester(
            initial_capital=INITIAL_CAPITAL,
            max_leverage=lev,
        )
        result = await bt.run(strat, bars, symbol)

        logger.info(
            "  %s @ %dx: $%.2f->$%.2f (%.1f%%) wr=%.1f%% trades=%d(%.1f/wk) "
            "avg_wk=%.1f%% RR=%.2f hold=%.0fm dd=%.1f%% %s",
            name, lev,
            result["initial_capital"], result["final_equity"],
            result["total_return_pct"],
            result["win_rate"], result["total_trades"],
            result["trades_per_week"],
            result["avg_weekly_return_pct"],
            result["avg_rr"],
            result["avg_hold_min"],
            result["max_drawdown_pct"],
            "BLOWN UP" if result["blown_up"] else "OK",
        )

        if not result["blown_up"]:
            if best_result is None or result["total_return_pct"] > best_result["total_return_pct"]:
                best_result = result
                best_leverage = lev

    if best_result is None:
        best_result = {
            "avg_weekly_return_pct": 0, "weeks_evaluated": 1,
            "leverage": 0, "total_return_pct": 0, "win_rate": 0,
            "blown_up": True, "final_equity": 0,
            "median_weekly_return_pct": 0, "weeks_positive": 0,
            "max_drawdown_pct": 100, "total_trades": 0, "avg_rr": 0,
            "avg_hold_min": 0, "trades_per_week": 0,
        }

    passes = not best_result["blown_up"] and best_result["avg_weekly_return_pct"] >= 100.0

    return {
        "strategy": name,
        "best_leverage": best_leverage,
        "best_result": best_result,
        "passes": passes,
    }


async def main() -> None:
    loader = CryptoDataLoader()
    symbol = "BTCUSDT"

    logger.info("Loading %s %s data...", symbol, EVAL_TIMEFRAME)
    bars = loader.load(symbol, EVAL_TIMEFRAME)
    if bars.empty:
        logger.error("No data available")
        return

    tf_min = 1
    bars_per_week = 7 * 24 * 60
    needed = bars_per_week * EVAL_WEEKS + 2000
    if len(bars) > needed:
        bars = bars.iloc[-needed:].reset_index(drop=True)
    bars.attrs["timeframe"] = EVAL_TIMEFRAME

    actual_weeks = len(bars) // bars_per_week
    logger.info("Evaluating on %d bars (~%d weeks of 1m data), $%.0f initial capital",
                len(bars), actual_weeks, INITIAL_CAPITAL)

    from strategy.btc.scalp_momentum import ScalpMomentumStrategy
    from strategy.btc.volatility_breakout_scalp import VolatilityBreakoutScalpStrategy

    base = lambda name, params=None: StrategyConfig(
        name=name, symbols=[symbol], params=params or {}
    )

    candidates = {
        "scalp_momentum": lambda: ScalpMomentumStrategy(base("ScalpMomentum")),
        "vol_breakout_scalp": lambda: VolatilityBreakoutScalpStrategy(base("VolBreakoutScalp")),
    }

    results: list[dict] = []
    for name, factory in candidates.items():
        logger.info("=== Evaluating: %s ===", name)
        r = await evaluate_strategy(name, factory, symbol, bars)
        results.append(r)

    logger.info("\n" + "=" * 80)
    logger.info("FINAL RESULTS ($%.0f capital, %d weeks, Kelly sizing)",
                INITIAL_CAPITAL, EVAL_WEEKS)
    logger.info("=" * 80)

    passing = []
    failing = []
    for r in results:
        br = r["best_result"]
        status = "PASS" if r["passes"] else ("BLOWN UP" if br.get("blown_up") else "FAIL")
        logger.info(
            "%s [%s] best@%dx: $%.0f->$%.2f (%.1f%%) wr=%.1f%% trades=%d(%.1f/wk) "
            "avg_wk=%.1f%% RR=%.2f hold=%.0fm dd=%.1f%%",
            r["strategy"], status, r["best_leverage"],
            INITIAL_CAPITAL, br.get("final_equity", 0),
            br["total_return_pct"],
            br.get("win_rate", 0), br.get("total_trades", 0),
            br.get("trades_per_week", 0),
            br["avg_weekly_return_pct"],
            br.get("avg_rr", 0),
            br.get("avg_hold_min", 0),
            br.get("max_drawdown_pct", 100),
        )
        if r["passes"]:
            passing.append(r)
        else:
            failing.append(r)

    strategy_dir = Path(__file__).resolve().parent.parent / "packages" / "strategy" / "btc"
    strategy_files = {
        "scalp_momentum": "scalp_momentum.py",
        "vol_breakout_scalp": "volatility_breakout_scalp.py",
    }

    auto_delete = "--delete" in sys.argv
    if failing and auto_delete:
        logger.info("\n--- Deleting failing/blown-up strategies ---")
        for r in failing:
            br = r["best_result"]
            if br.get("blown_up", False):
                fname = strategy_files.get(r["strategy"])
                if fname:
                    fpath = strategy_dir / fname
                    if fpath.exists():
                        fpath.unlink()
                        logger.info("DELETED (blown up): %s", fpath.name)
    elif failing:
        logger.info("\n--- Failing strategies (use --delete to auto-delete blown-up ones) ---")
        for r in failing:
            br = r["best_result"]
            logger.info("  %s: %s (avg_wk=%.1f%%, $%.2f final)",
                        r["strategy"],
                        "BLOWN UP" if br.get("blown_up") else "below target",
                        br["avg_weekly_return_pct"],
                        br.get("final_equity", 0))

    if passing:
        logger.info("\n--- Strategies that PASSED ---")
        for r in passing:
            br = r["best_result"]
            logger.info("  KEPT: %s @ %dx ($%.0f->$%.2f, avg_wk=%.1f%%, wr=%.1f%%)",
                        r["strategy"], r["best_leverage"],
                        INITIAL_CAPITAL, br["final_equity"],
                        br["avg_weekly_return_pct"], br.get("win_rate", 0))
    else:
        logger.info("\nNO strategies passed.")

    output_dir = Path(__file__).resolve().parent.parent / "models"
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / "leverage_backtest_results.json"
    serializable = []
    for r in results:
        sr = {k: v for k, v in r.items()}
        sr["best_result"] = {k: v for k, v in r["best_result"].items()
                             if k != "weekly_returns"}
        serializable.append(sr)
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    logger.info("\nResults saved to %s", out_path)


if __name__ == "__main__":
    asyncio.run(main())
