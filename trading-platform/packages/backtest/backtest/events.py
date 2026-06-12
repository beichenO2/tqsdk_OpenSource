"""事件驱动系统 - 回测引擎的通信骨架。"""

from __future__ import annotations

import enum
import itertools
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

logger = logging.getLogger(__name__)

_event_counter = itertools.count(1)


class EventType(str, enum.Enum):
    BAR = "BAR"
    TICK = "TICK"
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_PARTIAL_FILLED = "ORDER_PARTIAL_FILLED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    ORDER_REJECTED = "ORDER_REJECTED"
    TRADE = "TRADE"
    POSITION_UPDATE = "POSITION_UPDATE"
    EQUITY_UPDATE = "EQUITY_UPDATE"
    ENGINE_START = "ENGINE_START"
    ENGINE_STOP = "ENGINE_STOP"
    STRATEGY_SIGNAL = "STRATEGY_SIGNAL"


@dataclass(slots=True)
class Event:
    """事件对象。"""
    type: EventType
    data: Any = None
    dt: datetime = field(default_factory=datetime.now)
    id: int = field(default_factory=lambda: next(_event_counter))
    source: str = ""


EventHandler = Callable[[Event], None]


class EventBus:
    """事件总线 - 发布/订阅模式。"""

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[EventHandler]] = defaultdict(list)
        self._queue: deque[Event] = deque()
        self._processing = False

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        handlers = self._handlers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    def publish(self, event: Event) -> None:
        self._queue.append(event)
        if not self._processing:
            self._drain()

    def _drain(self) -> None:
        self._processing = True
        try:
            while self._queue:
                event = self._queue.popleft()
                for handler in self._handlers.get(event.type, []):
                    try:
                        handler(event)
                    except Exception:
                        logger.exception(
                            "Handler %s failed for event %s",
                            handler.__qualname__,
                            event.type.value,
                        )
        finally:
            self._processing = False

    def clear(self) -> None:
        self._handlers.clear()
        self._queue.clear()
