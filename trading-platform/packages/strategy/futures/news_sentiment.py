"""新闻情绪联动策略 — 地缘/宏观事件驱动的期货交易信号。

信息传导链假说：
  地缘事件(美伊冲突) → 大宗商品供需预期 → 期货价格 → 交易信号

品种-事件映射（基于经济学先验）：
  原油相关(sc,lu) ← 中东冲突, OPEC减产, 制裁
  黄金(au,ag) ← 避险情绪, 美元指数, 央行政策
  金属(cu,al,zn) ← 全球经济景气, 中国PMI, 基建政策
  农产品(m,p,CF) ← 天气, 贸易政策, 种植面积
  化工(TA,MA,pp) ← 原油联动, 下游需求

实现方式：
  1. 从 DiGist/KnowLever 获取新闻摘要
  2. 提取情绪分数和关键实体
  3. 映射到品种影响
  4. 与价格趋势结合产生交易信号

Method: 新闻情绪分析属于 NLP 应用领域，实际信号来自品种-事件关联表
（经济学/金融学先验知识），非特定论文方法。
如后续需要 NLP 模型，应使用 ≤3年白名单方法。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS: dict[str, Any] = {
    "sentiment_threshold": 0.3,
    "impact_decay_hours": 4.0,
    "max_hold_bars": 40,
    "atr_period": 14,
    "trailing_stop_atr_mult": 2.0,
    "digist_url": "http://127.0.0.1:3800",
    "check_interval_bars": 12,
}

COMMODITY_EVENT_MAP: dict[str, list[str]] = {
    "sc": ["oil", "opec", "iran", "middle_east", "sanctions", "crude"],
    "lu": ["oil", "refinery", "crude", "energy"],
    "au": ["risk_aversion", "fed", "dollar", "central_bank", "gold", "geopolitics"],
    "ag": ["silver", "industrial_metals", "solar", "risk_aversion"],
    "cu": ["china_pmi", "construction", "ev", "infrastructure", "copper"],
    "al": ["aluminum", "energy_cost", "infrastructure", "china"],
    "rb": ["steel", "infrastructure", "housing", "china_policy"],
    "i": ["iron_ore", "steel", "china_demand", "australia"],
    "m": ["soybean", "trade_war", "weather", "usda", "brazil"],
    "TA": ["oil_linkage", "pta", "polyester", "crude"],
    "MA": ["methanol", "coal", "natural_gas", "chemical"],
    "CF": ["cotton", "weather", "india", "trade_policy"],
}

SENTIMENT_DIRECTION: dict[str, dict[str, int]] = {
    "middle_east_conflict": {"sc": 1, "lu": 1, "au": 1, "ag": 1, "TA": 1, "MA": 1},
    "china_stimulus": {"rb": 1, "i": 1, "cu": 1, "al": 1},
    "fed_rate_hike": {"au": -1, "ag": -1, "cu": -1},
    "fed_rate_cut": {"au": 1, "ag": 1, "cu": 1},
    "trade_war_escalation": {"m": -1, "CF": -1, "cu": -1},
    "oil_supply_disruption": {"sc": 1, "lu": 1, "TA": 1},
    "weather_crisis": {"m": 1, "CF": 1},
}


def _fetch_digist_headlines(url: str, topics: list[str], limit: int = 20) -> list[dict]:
    """Fetch recent headlines from DiGist service."""
    try:
        query = json.dumps({
            "topics": topics,
            "limit": limit,
            "lang": "zh",
        }).encode()
        req = Request(
            f"{url}/api/search",
            data=query,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return data.get("results", [])
    except Exception as e:
        logger.debug("DiGist fetch failed: %s", e)
        return []


def _simple_sentiment_score(text: str) -> float:
    """Rule-based sentiment scoring for commodity news (Chinese + English).

    Returns float in [-1, 1]. Positive = bullish, Negative = bearish.
    """
    bullish_keywords = [
        "上涨", "突破", "利好", "增产", "供应紧张", "需求强劲", "减产",
        "surge", "rally", "bullish", "shortage", "cut production",
        "冲突", "制裁", "封锁", "战争", "导弹", "打击",
        "stimulus", "基建", "刺激", "投资",
    ]
    bearish_keywords = [
        "下跌", "暴跌", "利空", "过剩", "需求疲弱", "增产",
        "crash", "bearish", "oversupply", "recession", "衰退",
        "和平", "停火", "解除制裁", "谈判成功",
        "加息", "紧缩", "收紧",
    ]

    text_lower = text.lower()
    bull_count = sum(1 for kw in bullish_keywords if kw in text_lower or kw.lower() in text_lower)
    bear_count = sum(1 for kw in bearish_keywords if kw in text_lower or kw.lower() in text_lower)

    total = bull_count + bear_count
    if total == 0:
        return 0.0

    return (bull_count - bear_count) / total


def _match_event_type(text: str) -> str | None:
    """Match text to a predefined event type."""
    event_keywords = {
        "middle_east_conflict": ["伊朗", "以色列", "中东", "iran", "israel", "middle east", "导弹", "missile"],
        "china_stimulus": ["刺激", "基建", "降准", "降息", "stimulus", "infrastructure"],
        "fed_rate_hike": ["加息", "rate hike", "tightening", "hawkish"],
        "fed_rate_cut": ["降息", "rate cut", "dovish", "easing"],
        "oil_supply_disruption": ["断供", "封锁", "制裁", "disruption", "sanctions", "blockade"],
        "weather_crisis": ["干旱", "洪水", "暴风", "drought", "flood", "hurricane"],
        "trade_war_escalation": ["贸易战", "关税", "trade war", "tariff"],
    }

    text_lower = text.lower()
    for event_type, keywords in event_keywords.items():
        if any(kw in text_lower or kw.lower() in text_lower for kw in keywords):
            return event_type
    return None


@auto_register("news_sentiment")
class NewsSentimentStrategy(BaseStrategy):
    """新闻情绪联动日内策略。

    定期查询 DiGist 获取新闻 → 情绪分析 → 品种映射 → 交易信号。
    当 DiGist 不可用时降级为纯价格策略（不产生新闻信号）。
    """

    def __init__(self, config: StrategyConfig) -> None:
        config = config.model_copy(
            update={"params": {**DEFAULT_PARAMS, **config.params}}
        )
        super().__init__(config)
        self._bar_count = 0
        self._last_check_bar = 0
        self._current_sentiment: dict[str, float] = {}
        self._current_event: str | None = None
        self._position_side: str | None = None
        self._entry_price = 0.0
        self._hold_bars = 0
        self._closes: list[float] = []

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        c = float(bar.get("close", 0))
        self._closes.append(c)
        self._bar_count += 1

        check_interval = self.get_param("check_interval_bars", 12)
        if self._bar_count - self._last_check_bar >= check_interval:
            self._update_sentiment(symbol)
            self._last_check_bar = self._bar_count

        base_symbol = "".join(ch for ch in symbol if ch.isalpha())

        signals = []

        if self._position_side:
            self._hold_bars += 1
            max_hold = self.get_param("max_hold_bars", 40)

            if self._hold_bars >= max_hold:
                exit_type = SignalType.LONG_EXIT if self._position_side == "long" else SignalType.SHORT_EXIT
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_type, strength=0.7, price=c,
                    reason=f"news_max_hold: {self._hold_bars} bars",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = None
                self._hold_bars = 0
                return signals

            sentiment = self._current_sentiment.get(base_symbol, 0.0)
            if self._position_side == "long" and sentiment < -0.3:
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_EXIT, strength=abs(sentiment), price=c,
                    reason=f"news_reversal: sentiment={sentiment:.2f} event={self._current_event}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = None
                self._hold_bars = 0
                return signals

            if self._position_side == "short" and sentiment > 0.3:
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_EXIT, strength=abs(sentiment), price=c,
                    reason=f"news_reversal: sentiment={sentiment:.2f} event={self._current_event}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = None
                self._hold_bars = 0
                return signals

        if not self._position_side:
            sentiment = self._current_sentiment.get(base_symbol, 0.0)
            threshold = self.get_param("sentiment_threshold", 0.3)

            if sentiment > threshold and self._current_event:
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY, strength=min(abs(sentiment), 1.0), price=c,
                    reason=f"news_bullish: {self._current_event} sentiment={sentiment:.2f}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "long"
                self._entry_price = c
                self._hold_bars = 0

            elif sentiment < -threshold and self._current_event:
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY, strength=min(abs(sentiment), 1.0), price=c,
                    reason=f"news_bearish: {self._current_event} sentiment={sentiment:.2f}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "short"
                self._entry_price = c
                self._hold_bars = 0

        return signals

    def _update_sentiment(self, symbol: str) -> None:
        base_symbol = "".join(ch for ch in symbol if ch.isalpha())
        topics = COMMODITY_EVENT_MAP.get(base_symbol, ["commodity"])

        digist_url = self.get_param("digist_url", "http://127.0.0.1:3800")
        headlines = _fetch_digist_headlines(digist_url, topics, limit=10)

        if not headlines:
            return

        total_sentiment = 0.0
        event_detected = None

        for item in headlines:
            text = item.get("title", "") + " " + item.get("summary", "")
            score = _simple_sentiment_score(text)
            total_sentiment += score

            event = _match_event_type(text)
            if event and event_detected is None:
                event_detected = event

        avg_sentiment = total_sentiment / len(headlines) if headlines else 0.0

        if event_detected:
            direction_map = SENTIMENT_DIRECTION.get(event_detected, {})
            event_direction = direction_map.get(base_symbol, 0)
            if event_direction != 0:
                avg_sentiment = avg_sentiment * 0.5 + event_direction * 0.5

        self._current_sentiment[base_symbol] = avg_sentiment
        self._current_event = event_detected

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        return []
