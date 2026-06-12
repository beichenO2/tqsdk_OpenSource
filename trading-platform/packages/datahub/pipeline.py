"""数据管道 - 编排 数据采集 → 清洗 → 存储 的完整流程"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Optional

from datahub.cleaning import clean_ohlcv
from datahub.models import OHLCV, DataQualityReport, TimeFrame
from datahub.providers.base import DataProvider
from datahub.storage import StorageEngine, StorageLayer

logger = logging.getLogger(__name__)


class DataPipeline:
    """数据 ETL 管道

    Provider → Bronze → Clean → Silver → (特征工程后) → Gold
    """

    def __init__(
        self,
        provider: DataProvider,
        storage: StorageEngine,
        fill_method: str = "ffill",
        outlier_std: float = 5.0,
    ):
        self._provider = provider
        self._storage = storage
        self._fill_method = fill_method
        self._outlier_std = outlier_std

    async def ingest(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: datetime,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> DataQualityReport:
        """执行完整的数据摄入流程: 采集 → Bronze → 清洗 → Silver

        Returns:
            数据质量报告
        """
        logger.info("Ingesting %s/%s from %s", symbol, timeframe.value, self._provider.name)

        bars = await self._provider.get_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            limit=limit,
        )

        if not bars:
            raise ValueError(f"No data returned for {symbol}/{timeframe.value}")

        self._storage.write_ohlcv(bars, layer=StorageLayer.BRONZE)

        import pandas as pd
        df = pd.DataFrame([b.model_dump() for b in bars])
        cleaned_df, report = clean_ohlcv(
            df, timeframe,
            fill_method=self._fill_method,
            outlier_std=self._outlier_std,
        )

        cleaned_bars = [
            OHLCV(
                symbol=symbol,
                exchange=bars[0].exchange,
                timeframe=timeframe,
                timestamp=row["timestamp"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                turnover=row.get("turnover"),
                open_interest=row.get("open_interest"),
            )
            for _, row in cleaned_df.iterrows()
        ]
        self._storage.write_ohlcv(cleaned_bars, layer=StorageLayer.SILVER)

        logger.info(
            "Ingest complete: %d raw → %d clean bars, coverage=%.1f%%",
            len(bars), len(cleaned_bars), report.coverage_pct,
        )
        return report

    async def batch_ingest(
        self,
        symbols: list[str],
        timeframe: TimeFrame,
        start: datetime,
        end: Optional[datetime] = None,
        concurrency: int = 5,
    ) -> dict[str, DataQualityReport]:
        """批量摄入多个合约的数据"""
        semaphore = asyncio.Semaphore(concurrency)
        reports: dict[str, DataQualityReport] = {}

        async def _ingest_one(sym: str) -> None:
            async with semaphore:
                try:
                    report = await self.ingest(sym, timeframe, start, end)
                    reports[sym] = report
                except Exception as e:
                    logger.error("Failed to ingest %s: %s", sym, e)

        await asyncio.gather(*[_ingest_one(s) for s in symbols])
        return reports

    def promote_to_gold(self, symbol: str, timeframe: TimeFrame) -> None:
        """将 Silver 层数据提升到 Gold 层（由特征工程模块调用后触发）"""
        df = self._storage.read_ohlcv(symbol, timeframe, layer=StorageLayer.SILVER)
        bars = [
            OHLCV(
                symbol=row.get("symbol", symbol),
                exchange=row.get("exchange", "UNKNOWN"),
                timeframe=timeframe,
                timestamp=row["timestamp"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                turnover=row.get("turnover"),
                open_interest=row.get("open_interest"),
            )
            for _, row in df.iterrows()
        ]
        self._storage.write_ohlcv(bars, layer=StorageLayer.GOLD)
        logger.info("Promoted %s/%s to Gold layer", symbol, timeframe.value)
