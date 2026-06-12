"""分层存储引擎 - Bronze/Silver/Gold 数据湖架构

Bronze: 原始数据（原样保存，不做任何变换）
Silver: 清洗后数据（去重、补缺、校验）
Gold:   特征就绪数据（对齐、标准化、可直接用于因子计算）
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from datahub.models import OHLCV, TimeFrame

logger = logging.getLogger(__name__)


class StorageLayer(str, Enum):
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"


class StorageEngine:
    """基于 DuckDB + Parquet 的分层存储引擎"""

    def __init__(self, base_dir: str | Path, db_path: Optional[str | Path] = None):
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        for layer in StorageLayer:
            (self._base_dir / layer.value).mkdir(exist_ok=True)

        db_file = str(db_path) if db_path else str(self._base_dir / "datahub.duckdb")
        self._conn = duckdb.connect(db_file)
        self._init_catalog()

    def _init_catalog(self) -> None:
        """初始化元数据目录表"""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS data_catalog (
                symbol      VARCHAR,
                exchange    VARCHAR,
                timeframe   VARCHAR,
                layer       VARCHAR,
                file_path   VARCHAR,
                row_count   BIGINT,
                min_ts      TIMESTAMP,
                max_ts      TIMESTAMP,
                updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, timeframe, layer)
            )
        """)

    def write_ohlcv(
        self,
        bars: list[OHLCV],
        layer: StorageLayer = StorageLayer.BRONZE,
    ) -> Path:
        """写入K线数据到指定层"""
        if not bars:
            raise ValueError("Empty bar list")

        df = pd.DataFrame([b.model_dump() for b in bars])
        symbol = bars[0].symbol
        timeframe = bars[0].timeframe.value
        exchange = bars[0].exchange

        file_path = (
            self._base_dir
            / layer.value
            / exchange
            / f"{symbol}_{timeframe}.parquet"
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if file_path.exists():
            existing = pd.read_parquet(file_path)
            df = pd.concat([existing, df]).drop_duplicates(
                subset=["timestamp"], keep="last"
            )
            df = df.sort_values("timestamp").reset_index(drop=True)

        table = pa.Table.from_pandas(df)
        pq.write_table(table, file_path, compression="zstd")

        self._conn.execute("""
            INSERT OR REPLACE INTO data_catalog
            (symbol, exchange, timeframe, layer, file_path, row_count, min_ts, max_ts, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            symbol, exchange, timeframe, layer.value, str(file_path),
            len(df), df["timestamp"].min(), df["timestamp"].max(),
            datetime.now(UTC),
        ])

        logger.info(
            "Wrote %d bars to %s [%s/%s/%s]",
            len(df), layer.value, exchange, symbol, timeframe,
        )
        return file_path

    def read_ohlcv(
        self,
        symbol: str,
        timeframe: TimeFrame,
        layer: StorageLayer = StorageLayer.GOLD,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """从指定层读取K线数据为 DataFrame"""
        row = self._conn.execute("""
            SELECT file_path FROM data_catalog
            WHERE symbol = ? AND timeframe = ? AND layer = ?
        """, [symbol, timeframe.value, layer.value]).fetchone()

        if row is None:
            raise FileNotFoundError(
                f"No data for {symbol}/{timeframe.value} in {layer.value}"
            )

        df = pd.read_parquet(row[0])
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        if start:
            df = df[df["timestamp"] >= pd.Timestamp(start)]
        if end:
            df = df[df["timestamp"] <= pd.Timestamp(end)]

        return df.reset_index(drop=True)

    def query(self, sql: str) -> pd.DataFrame:
        """直接执行 DuckDB SQL 查询（支持跨 parquet 文件查询）"""
        return self._conn.execute(sql).df()

    def list_datasets(self, layer: Optional[StorageLayer] = None) -> pd.DataFrame:
        """列出数据目录"""
        sql = "SELECT * FROM data_catalog"
        if layer:
            sql += f" WHERE layer = '{layer.value}'"
        return self._conn.execute(sql).df()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> StorageEngine:
        return self

    def __exit__(self, *args) -> None:
        self.close()
