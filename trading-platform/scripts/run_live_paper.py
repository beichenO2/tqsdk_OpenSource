"""实时模拟实盘 — 接入 Binance WebSocket 实时 K 线驱动100个加密策略。

用法:
    cd trading-platform
    PYTHONPATH=packages:src python scripts/run_live_paper.py [--interval 1m]

先安装依赖:
    pip install websockets
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys

import os as _os
import sys as _sys

try:
    from polarisor_port_sdk import submit_task as _sdk_submit, complete_task as _sdk_complete
except ImportError:
    _sdk_submit = _sdk_complete = None

from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:

    _task_id = None
    if _sdk_submit:
        try:
            _tr = _sdk_submit(task_type="paper-trading", command="run_live_paper.py", requester="run-live-paper", estimated_duration_sec=86400)
            _task_id = _tr.get("task_id")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="实时模拟实盘")
    parser.add_argument("--interval", default="1m", help="K线周期 (1m/5m/15m/1h)")
    parser.add_argument("--symbols", nargs="+", default=None, help="交易品种列表")
    parser.add_argument("--report-every", type=int, default=30, help="每 N 根 bar 保存报告")
    args = parser.parse_args()

    from sim_live.realtime_feed import RealtimePaperEngine

    engine = RealtimePaperEngine(
        symbols=args.symbols,
        interval=args.interval,
        report_interval=args.report_every,
    )

    # 优雅关闭
    loop = asyncio.get_running_loop()
    for sig_name in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig_name, lambda: asyncio.create_task(engine.stop()))

    logger.info("=" * 60)
    logger.info("实时模拟实盘启动")
    logger.info("品种: %s", engine.symbols)
    logger.info("周期: %s", args.interval)
    logger.info("策略数: %d", len(engine.strategies))
    logger.info("Ctrl+C 停止")
    logger.info("=" * 60)

    await engine.start()


    if _task_id and _sdk_complete:
        try:
            _sdk_complete(_task_id)
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
