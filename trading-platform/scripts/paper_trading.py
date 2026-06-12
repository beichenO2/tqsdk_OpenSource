#!/usr/bin/env python3
"""Paper trading service — runs best strategies on live/simulated data.

Connects to Binance public websocket for real-time klines,
feeds them to strategies, and logs signals without placing real orders.

Usage:
    python scripts/paper_trading.py --strategies momentum trend_following
    python scripts/paper_trading.py --symbol BTCUSDT --interval 1h
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys

import os as _os
import sys as _sys

try:
    from polarisor_port_sdk import submit_task as _sdk_submit, complete_task as _sdk_complete
except ImportError:
    _sdk_submit = _sdk_complete = None

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from strategy.base import Signal, SignalType, StrategyConfig, Position, OrderSide

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("paper_trading.log")],
)
logger = logging.getLogger(__name__)

OPTIMIZED_PARAMS = {
    "momentum": {
        "fast_period": 12, "slow_period": 19, "volume_ma_period": 19,
        "momentum_threshold": 0.033, "volume_surge_ratio": 1.003,
        "atr_period": 14, "trailing_stop_atr_mult": 2.26,
    },
    "trend_following": {
        "ema_fast": 8, "ema_slow": 18, "ema_trend": 59,
        "adx_period": 17, "adx_threshold": 31.69,
        "atr_period": 16, "trailing_stop_atr_mult": 3.72,
        "partial_take_profit_atr_mult": 3.91,
        "partial_close_pct": 0.61, "risk_per_trade_pct": 0.018,
    },
}


class PaperAccount:
    """Track paper positions and PnL."""

    def __init__(self, initial_capital: float = 100_000.0) -> None:
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions: dict[str, dict[str, Any]] = {}
        self.trades: list[dict[str, Any]] = []
        self.equity_history: list[dict[str, Any]] = []

    def open_position(self, symbol: str, side: str, price: float, qty: float) -> None:
        self.positions[symbol] = {"side": side, "price": price, "qty": qty, "time": datetime.now(timezone.utc).isoformat()}
        logger.info("[PAPER] OPEN %s %s @ %.2f qty=%.6f", side.upper(), symbol, price, qty)

    def close_position(self, symbol: str, price: float) -> float:
        pos = self.positions.pop(symbol, None)
        if not pos:
            return 0.0
        if pos["side"] == "long":
            pnl = (price - pos["price"]) * pos["qty"]
        else:
            pnl = (pos["price"] - price) * pos["qty"]
        self.capital += pnl
        self.trades.append({
            "symbol": symbol, "side": pos["side"], "entry": pos["price"],
            "exit": price, "qty": pos["qty"], "pnl": round(pnl, 2),
            "time": datetime.now(timezone.utc).isoformat(),
        })
        logger.info("[PAPER] CLOSE %s %s @ %.2f PnL=%.2f Capital=%.2f",
                     pos["side"].upper(), symbol, price, pnl, self.capital)
        return pnl

    @property
    def equity(self) -> float:
        return self.capital

    def snapshot(self) -> dict[str, Any]:
        return {
            "capital": round(self.capital, 2),
            "positions": dict(self.positions),
            "total_trades": len(self.trades),
            "total_pnl": round(self.capital - self.initial_capital, 2),
            "return_pct": round((self.capital - self.initial_capital) / self.initial_capital * 100, 2),
        }


class PaperTradingService:
    """Main paper trading loop."""

    def __init__(
        self,
        strategies: dict[str, Any],
        symbol: str = "BTCUSDT",
        account: PaperAccount | None = None,
    ) -> None:
        self.strategies = strategies
        self.symbol = symbol
        self.account = account or PaperAccount()
        self._running = False

    async def on_kline(self, bar: dict[str, Any]) -> None:
        """Process a single kline bar through all strategies."""
        price = float(bar["close"])
        ts = bar.get("time", datetime.now(timezone.utc).isoformat())

        for name, strategy in self.strategies.items():
            try:
                signals = await strategy.on_bar(self.symbol, bar)
            except Exception as e:
                logger.debug("Strategy %s error: %s", name, e)
                continue

            for sig in signals:
                if not isinstance(sig, Signal):
                    continue

                pos = self.account.positions.get(self.symbol)
                trade_value = self.account.capital * 0.1
                qty = trade_value / price if price > 0 else 0

                if sig.signal_type == SignalType.LONG_ENTRY and not pos:
                    self.account.open_position(self.symbol, "long", price, qty)
                    strategy.update_position(Position(symbol=self.symbol, side=OrderSide.BUY, qty=qty, avg_price=price))
                    logger.info("[SIGNAL] %s → LONG_ENTRY @ %.2f (%s)", name, price, sig.reason or "")

                elif sig.signal_type == SignalType.SHORT_ENTRY and not pos:
                    self.account.open_position(self.symbol, "short", price, qty)
                    strategy.update_position(Position(symbol=self.symbol, side=OrderSide.SELL, qty=qty, avg_price=price))
                    logger.info("[SIGNAL] %s → SHORT_ENTRY @ %.2f (%s)", name, price, sig.reason or "")

                elif sig.signal_type == SignalType.LONG_EXIT and pos and pos["side"] == "long":
                    self.account.close_position(self.symbol, price)
                    strategy.remove_position(self.symbol)
                    logger.info("[SIGNAL] %s → LONG_EXIT @ %.2f (%s)", name, price, sig.reason or "")

                elif sig.signal_type == SignalType.SHORT_EXIT and pos and pos["side"] == "short":
                    self.account.close_position(self.symbol, price)
                    strategy.remove_position(self.symbol)
                    logger.info("[SIGNAL] %s → SHORT_EXIT @ %.2f (%s)", name, price, sig.reason or "")

        self.account.equity_history.append({"time": ts, "equity": round(self.account.equity, 2), "price": price})

    async def run_websocket(self, interval: str = "1h") -> None:
        """Connect to Binance websocket for real-time klines."""
        try:
            import websockets
        except ImportError:
            logger.error("websockets not installed. Install with: pip install websockets")
            logger.info("Falling back to polling mode...")
            await self.run_polling(interval)
            return

        stream = f"{self.symbol.lower()}@kline_{interval}"
        url = f"wss://stream.binance.com:9443/ws/{stream}"

        logger.info("Connecting to Binance WebSocket: %s", stream)
        self._running = True

        while self._running:
            try:
                async with websockets.connect(url) as ws:
                    logger.info("Connected. Waiting for klines...")
                    async for msg in ws:
                        if not self._running:
                            break
                        data = json.loads(msg)
                        k = data.get("k", {})
                        if not k.get("x"):
                            continue
                        bar = {
                            "open": float(k["o"]),
                            "high": float(k["h"]),
                            "low": float(k["l"]),
                            "close": float(k["c"]),
                            "volume": float(k["v"]),
                            "taker_buy_volume": float(k.get("V", k["v"])) * 0.5,
                            "time": datetime.fromtimestamp(k["t"] / 1000, tz=timezone.utc).isoformat(),
                        }
                        await self.on_kline(bar)
                        snap = self.account.snapshot()
                        logger.info("[STATUS] Capital=%.2f PnL=%.2f%% Trades=%d",
                                     snap["capital"], snap["return_pct"], snap["total_trades"])
            except Exception as e:
                if self._running:
                    logger.warning("WebSocket error: %s. Reconnecting in 5s...", e)
                    await asyncio.sleep(5)

    async def run_polling(self, interval: str = "1h") -> None:
        """Fallback: poll Binance REST API for latest kline."""
        import urllib.request

        interval_seconds = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
        sleep_time = interval_seconds.get(interval, 3600)

        logger.info("Polling mode: %s every %ds", self.symbol, sleep_time)
        self._running = True

        while self._running:
            try:
                url = f"https://api.binance.com/api/v3/klines?symbol={self.symbol}&interval={interval}&limit=2"
                with urllib.request.urlopen(url, timeout=10) as resp:
                    klines = json.loads(resp.read())
                if klines and len(klines) >= 2:
                    k = klines[-2]
                    bar = {
                        "open": float(k[1]), "high": float(k[2]),
                        "low": float(k[3]), "close": float(k[4]),
                        "volume": float(k[5]),
                        "taker_buy_volume": float(k[9]) if len(k) > 9 else float(k[5]) * 0.5,
                        "time": datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc).isoformat(),
                    }
                    await self.on_kline(bar)
                    snap = self.account.snapshot()
                    logger.info("[STATUS] Capital=%.2f PnL=%.2f%% Trades=%d Price=%.2f",
                                 snap["capital"], snap["return_pct"], snap["total_trades"], bar["close"])
            except Exception as e:
                logger.warning("Polling error: %s", e)

            await asyncio.sleep(sleep_time)

    def stop(self) -> None:
        self._running = False
        snap = self.account.snapshot()
        logger.info("\n[FINAL] %s", json.dumps(snap, indent=2))

        out = Path("models") / "paper_trading_history.json"
        out.parent.mkdir(exist_ok=True)
        with open(out, "w") as f:
            json.dump({
                "account": snap,
                "trades": self.account.trades,
                "equity_history": self.account.equity_history[-100:],
            }, f, indent=2, default=str)
        logger.info("History saved to %s", out)


def create_strategies(names: list[str], symbol: str) -> dict[str, Any]:
    from strategy.btc.momentum import BTCMomentumStrategy
    from strategy.btc.trend_following import BTCTrendFollowingStrategy

    FACTORY = {
        "momentum": (BTCMomentumStrategy, OPTIMIZED_PARAMS.get("momentum", {})),
        "trend_following": (BTCTrendFollowingStrategy, OPTIMIZED_PARAMS.get("trend_following", {})),
    }

    strats: dict[str, Any] = {}
    for name in names:
        if name in FACTORY:
            cls, params = FACTORY[name]
            strats[name] = cls(StrategyConfig(name=name, symbols=[symbol], params=params))
        else:
            logger.warning("Unknown strategy: %s", name)
    return strats


async def main() -> None:

    _task_id = None
    if _sdk_submit:
        try:
            _tr = _sdk_submit(task_type="paper-trading", command="paper_trading.py", requester="paper-trading", estimated_duration_sec=86400)
            _task_id = _tr.get("task_id")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="Paper trading service")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h", help="Kline interval (1m,5m,15m,1h,4h,1d)")
    parser.add_argument("--strategies", nargs="+", default=["momentum", "trend_following"])
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--mode", choices=["ws", "poll"], default="poll", help="WebSocket or REST polling")
    args = parser.parse_args()

    strategies = create_strategies(args.strategies, args.symbol)
    if not strategies:
        logger.error("No valid strategies specified")
        return

    account = PaperAccount(args.capital)
    service = PaperTradingService(strategies, args.symbol, account)

    loop = asyncio.get_event_loop()
    for sig_name in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig_name, service.stop)

    logger.info("=" * 60)
    logger.info("PAPER TRADING SERVICE")
    logger.info("Symbol: %s | Interval: %s | Capital: $%s", args.symbol, args.interval, f"{args.capital:,.0f}")
    logger.info("Strategies: %s", ", ".join(strategies.keys()))
    logger.info("Mode: %s", "WebSocket" if args.mode == "ws" else "REST Polling")
    logger.info("=" * 60)

    if args.mode == "ws":
        await service.run_websocket(args.interval)
    else:
        await service.run_polling(args.interval)


    if _task_id and _sdk_complete:
        try:
            _sdk_complete(_task_id)
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
