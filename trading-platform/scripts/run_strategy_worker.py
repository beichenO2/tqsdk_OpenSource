#!/usr/bin/env python3
"""独立策略 worker 进程 — 配置驱动、状态持久化、launchd 友好。

用法:
    cd trading-platform
    .venv/bin/python scripts/run_strategy_worker.py --config config/strategy_worker.json

paper 模式使用本地 SimAccount 撮合；live 模式需 --allow-live（M2 前默认拒绝）。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
for p in [_REPO, _REPO / "packages"]:
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Strategy worker (paper/live)")
    parser.add_argument(
        "--config",
        default="config/strategy_worker.json",
        help="Worker JSON config path",
    )
    parser.add_argument(
        "--state-dir",
        default="data/worker_state",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--allow-live",
        action="store_true",
        help="Allow live mode (M2 fill-feedback not yet wired)",
    )
    args = parser.parse_args()

    from sim_live.strategy_worker import StrategyWorker, load_worker_config

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _REPO / config_path

    config = load_worker_config(config_path)
    worker = StrategyWorker(
        config,
        state_dir=args.state_dir,
        allow_live=args.allow_live,
    )
    logger.info(
        "Starting worker %s market=%s mode=%s strategies=%d",
        config.worker_id,
        config.market,
        config.mode,
        len(config.strategies),
    )
    asyncio.run(worker.run())


if __name__ == "__main__":
    main()
