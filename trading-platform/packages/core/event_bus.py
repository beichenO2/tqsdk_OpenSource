"""事件总线 — 连接 ExecutionEngine/LiveScheduler 与 WebSocket 推送。

事件类型：
- position_update: 持仓变化
- trade_fill:      成交回报
- account_update:  账户资金变化
- strategy_status: 策略状态变化
- signal_generated: 策略产生信号
- risk_alert:      风控告警
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)

EventHandler = Callable[[dict[str, Any]], Awaitable[None] | None]


class EventBus:
    """进程内异步事件总线。"""

    _instance: EventBus | None = None

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._history: list[dict[str, Any]] = []
        self._max_history = 1000

    @classmethod
    def get_instance(cls) -> EventBus:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        handlers = self._handlers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    async def emit(self, event_type: str, data: dict[str, Any]) -> None:
        event = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        handlers = self._handlers.get(event_type, [])
        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("Event handler error for %s", event_type)

    def get_recent_events(
        self, event_type: str | None = None, limit: int = 50,
    ) -> list[dict[str, Any]]:
        events = self._history
        if event_type:
            events = [e for e in events if e["type"] == event_type]
        return events[-limit:]
