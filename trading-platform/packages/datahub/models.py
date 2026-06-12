"""数据模型定义 - 行情、K线、快照等核心数据结构"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TimeFrame(str, Enum):
    """K线时间周期"""
    TICK = "tick"
    S1 = "1s"
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"
    MN1 = "1M"


class OHLCV(BaseModel):
    """标准K线数据"""
    model_config = ConfigDict(frozen=True)

    symbol: str = Field(..., description="合约/交易对标识")
    exchange: str = Field(..., description="交易所标识")
    timeframe: TimeFrame
    timestamp: datetime = Field(..., description="K线开始时间 (UTC)")
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: Optional[float] = Field(None, description="成交额")
    open_interest: Optional[float] = Field(None, description="持仓量（期货）")


class TickData(BaseModel):
    """逐笔/快照行情"""
    symbol: str
    exchange: str
    timestamp: datetime
    last_price: float
    volume: float
    bid_price_1: Optional[float] = None
    bid_volume_1: Optional[float] = None
    ask_price_1: Optional[float] = None
    ask_volume_1: Optional[float] = None
    bid_price_2: Optional[float] = None
    bid_volume_2: Optional[float] = None
    ask_price_2: Optional[float] = None
    ask_volume_2: Optional[float] = None
    bid_price_3: Optional[float] = None
    bid_volume_3: Optional[float] = None
    ask_price_3: Optional[float] = None
    ask_volume_3: Optional[float] = None
    bid_price_4: Optional[float] = None
    bid_volume_4: Optional[float] = None
    ask_price_4: Optional[float] = None
    ask_volume_4: Optional[float] = None
    bid_price_5: Optional[float] = None
    bid_volume_5: Optional[float] = None
    ask_price_5: Optional[float] = None
    ask_volume_5: Optional[float] = None
    open_interest: Optional[float] = None
    turnover: Optional[float] = None


class MarketSnapshot(BaseModel):
    """市场概况快照"""
    symbol: str
    exchange: str
    timestamp: datetime
    last_price: float
    open: float
    high: float
    low: float
    pre_close: float
    volume: float
    turnover: float
    open_interest: Optional[float] = None
    upper_limit: Optional[float] = Field(None, description="涨停价")
    lower_limit: Optional[float] = Field(None, description="跌停价")


class DataQualityReport(BaseModel):
    """数据质量报告"""
    symbol: str
    timeframe: TimeFrame
    total_bars: int
    missing_bars: int
    duplicate_bars: int
    outlier_bars: int
    gap_count: int
    coverage_pct: float = Field(..., ge=0, le=100)
    checked_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_clean(self) -> bool:
        return (
            self.missing_bars == 0
            and self.duplicate_bars == 0
            and self.outlier_bars == 0
        )
