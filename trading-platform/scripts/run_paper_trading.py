"""模拟实盘主入口 — 加载数据、创建200个策略、回放历史进行模拟撮合。

用法:
    cd trading-platform
    PYTHONPATH=packages:src python scripts/run_paper_trading.py [--market crypto|futures|all]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_crypto_data(data_dir: Path) -> dict[str, list[dict]]:
    """加载加密货币 parquet 数据。

    目录结构: data/crypto_cache/{symbol_lower}/{timeframe}.parquet
    """
    bars_by_symbol: dict[str, list[dict]] = {}
    crypto_dir = data_dir / "crypto_cache"
    if not crypto_dir.exists():
        logger.warning("Crypto data dir not found: %s", crypto_dir)
        return bars_by_symbol

    symbol_dirs = {
        "btcusdt": "BTCUSDT",
        "ethusdt": "ETHUSDT",
        "bnbusdt": "BNBUSDT",
        "solusdt": "SOLUSDT",
        "xrpusdt": "XRPUSDT",
    }

    # 优先 1h 数据用于模拟实盘（速度与精度均衡）
    preferred_tf = ["1h", "4h", "1d"]

    for dir_name, symbol in symbol_dirs.items():
        sym_dir = crypto_dir / dir_name
        if not sym_dir.exists():
            continue
        for tf in preferred_tf:
            path = sym_dir / f"{tf}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                bars = []
                for _, row in df.iterrows():
                    bar = {
                        "timestamp": str(row.get("timestamp", row.name)),
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row.get("volume", 0)),
                    }
                    if "taker_buy_volume" in row:
                        bar["taker_buy_volume"] = float(row["taker_buy_volume"])
                    bars.append(bar)
                bars_by_symbol[symbol] = bars
                logger.info("Loaded %s: %d bars from %s/%s", symbol, len(bars), dir_name, tf)
                break

    return bars_by_symbol


def load_futures_data(data_dir: Path) -> dict[str, list[dict]]:
    """加载期货数据。"""
    bars_by_symbol: dict[str, list[dict]] = {}
    futures_dir = data_dir / "futures_cache"
    if not futures_dir.exists():
        logger.warning("Futures data dir not found: %s", futures_dir)
        return bars_by_symbol

    for parquet in futures_dir.glob("*.parquet"):
        symbol = parquet.stem
        df = pd.read_parquet(parquet)
        bars = []
        for _, row in df.iterrows():
            bar = {
                "timestamp": str(row.get("datetime", row.name)),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0)),
            }
            bars.append(bar)
        bars_by_symbol[symbol] = bars
        logger.info("Loaded futures %s: %d bars", symbol, len(bars))

    return bars_by_symbol


async def main() -> None:
    parser = argparse.ArgumentParser(description="模拟实盘系统")
    parser.add_argument("--market", choices=["crypto", "futures", "all"], default="all")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output-dir", default="output/paper_trading", help="输出目录")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    logger.info("=" * 60)
    logger.info("模拟实盘系统启动")
    logger.info("市场: %s", args.market)
    logger.info("=" * 60)

    # 1) 加载数据
    all_bars: dict[str, list[dict]] = {}
    if args.market in ("crypto", "all"):
        all_bars.update(load_crypto_data(data_dir))
    if args.market in ("futures", "all"):
        all_bars.update(load_futures_data(data_dir))

    if not all_bars:
        logger.error("No data loaded! Check data directory: %s", data_dir)
        sys.exit(1)

    logger.info("Total symbols loaded: %d", len(all_bars))
    for sym, bars in all_bars.items():
        logger.info("  %s: %d bars", sym, len(bars))

    # 2) 创建账号管理器
    from sim_live.account_manager import AccountManager
    accounts = AccountManager(
        crypto_count=100 if args.market in ("crypto", "all") else 0,
        futures_count=100 if args.market in ("futures", "all") else 0,
        crypto_capital=100_000.0,
        futures_capital=1_000_000.0,
    )

    # 3) 创建策略
    from sim_live.strategy_factory import create_all_strategies
    market_filter = None if args.market == "all" else args.market
    strategies = create_all_strategies(market=market_filter)
    logger.info("Created %d strategies", len(strategies))

    # 分配策略到账号
    from sim_live.strategy_catalog import get_catalog
    for entry in get_catalog(market_filter):
        accounts.assign_strategy(entry["account_id"], entry["name"], entry.get("params", {}))

    # 4) 创建调度器
    from sim_live.paper_scheduler import PaperScheduler
    scheduler = PaperScheduler(accounts, strategies)

    # 5) 运行模拟
    logger.info("Starting simulation...")
    result = await scheduler.run_history(all_bars, progress_every=1000)

    # 6) 生成报告
    from sim_live.reporter import PaperReporter
    reporter = PaperReporter(accounts)

    print("\n" + reporter.leaderboard_report())
    print("\n" + reporter.category_report())

    reporter.save_full_report(output_dir)
    logger.info("All reports saved to %s", output_dir)

    # 打印摘要
    summary = result["summary"]
    print(f"\n{'=' * 60}")
    print(f"模拟完成: {result['total_bars']} bars, {result['total_signals']} signals")
    print(f"加密货币: 平均收益 {summary['crypto']['avg_return']:.2f}%, "
          f"盈利 {summary['crypto']['profitable']}/{summary['crypto']['count']}")
    print(f"期    货: 平均收益 {summary['futures']['avg_return']:.2f}%, "
          f"盈利 {summary['futures']['profitable']}/{summary['futures']['count']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
