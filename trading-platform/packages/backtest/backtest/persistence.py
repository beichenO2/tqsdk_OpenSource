"""回测结果持久化 - 将 BacktestResult 写入数据库。

对接 Ch36 的 BacktestRun / BacktestTrade / BacktestMetric ORM 模型。
"""

from __future__ import annotations

import json
import logging
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

from .models import BacktestResult

logger = logging.getLogger(__name__)


class BacktestPersistence:
    """将回测结果写入 PostgreSQL。"""

    METRIC_FIELDS = [
        "total_return",
        "annual_return",
        "max_drawdown",
        "max_drawdown_pct",
        "sharpe_ratio",
        "sortino_ratio",
        "win_rate",
        "profit_factor",
        "calmar_ratio",
        "avg_trade_pnl",
        "avg_holding_period",
    ]

    async def save(
        self,
        session: AsyncSession,
        result: BacktestResult,
        user_id: str,
        run_name: str | None = None,
    ) -> str:
        """保存回测结果到数据库，返回 run_id。"""
        from core.db.models.backtest import (
            BacktestMetric,
            BacktestRun,
            BacktestStatus,
            BacktestTrade,
        )

        run = BacktestRun(
            user_id=user_id,
            strategy_id=result.config.strategy_id,
            name=run_name,
            status=BacktestStatus.COMPLETED,
            start_date=result.config.start_date,
            end_date=result.config.end_date,
            initial_capital=result.config.initial_capital,
            instruments_json=json.dumps(result.config.symbols),
            params_json=json.dumps(result.config.metadata, default=str),
            started_at=result.start_time,
            finished_at=result.end_time,
        )
        session.add(run)
        await session.flush()

        for trade in result.trades:
            bt_trade = BacktestTrade(
                run_id=run.id,
                instrument_symbol=trade.symbol,
                side=trade.side.value,
                price=trade.price,
                quantity=Decimal(str(trade.volume)),
                commission=trade.commission,
                traded_at=trade.dt,
            )
            session.add(bt_trade)

        for field_name in self.METRIC_FIELDS:
            value = getattr(result, field_name, None)
            if value is not None:
                metric = BacktestMetric(
                    run_id=run.id,
                    metric_name=field_name,
                    metric_value=Decimal(str(value)),
                )
                session.add(metric)

        session.add(
            BacktestMetric(
                run_id=run.id,
                metric_name="total_trades",
                metric_value=Decimal(str(result.total_trades)),
            )
        )
        session.add(
            BacktestMetric(
                run_id=run.id,
                metric_name="final_equity",
                metric_value=result.final_equity,
            )
        )

        await session.commit()
        logger.info("Backtest result saved: run_id=%s, trades=%d", run.id, len(result.trades))
        return run.id

    async def load(self, session: AsyncSession, run_id: str) -> dict[str, Any]:
        """从数据库加载回测结果摘要。"""
        from sqlalchemy import select
        from core.db.models.backtest import BacktestRun, BacktestMetric

        run = await session.get(BacktestRun, run_id)
        if run is None:
            raise ValueError(f"BacktestRun {run_id} not found")

        stmt = select(BacktestMetric).where(BacktestMetric.run_id == run_id)
        result = await session.execute(stmt)
        metrics = {m.metric_name: float(m.metric_value) for m in result.scalars()}

        return {
            "run_id": run.id,
            "strategy_id": run.strategy_id,
            "status": run.status.value,
            "start_date": run.start_date.isoformat() if run.start_date else None,
            "end_date": run.end_date.isoformat() if run.end_date else None,
            "initial_capital": float(run.initial_capital),
            "metrics": metrics,
        }
