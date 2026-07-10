"""WebSocket endpoints for real-time market data and trading event streaming."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()

LIVE_WS_EVENT_TYPES = [
    "position_update",
    "trade_fill",
    "account_update",
    "strategy_status",
    "signal_generated",
    "risk_alert",
    "order_rejected",
    "order_cancelled",
    "order_partially_filled",
]


@dataclass
class _Client:
    ws: WebSocket
    subscriptions: set[str] = field(default_factory=set)


_connections: dict[str, dict[WebSocket, _Client]] = {
    "futures": {},
    "btc": {},
    "live": {},
}
_lock = asyncio.Lock()


async def _safe_send(ws: WebSocket, data: dict[str, Any]) -> bool:
    try:
        await ws.send_json(data)
        return True
    except Exception:
        return False


@router.websocket("/ws")
async def ws_futures(ws: WebSocket) -> None:
    await ws.accept()
    async with _lock:
        _connections["futures"][ws] = _Client(ws=ws)
    logger.info("Futures WS client connected (%d total)", len(_connections["futures"]))
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
                action = msg.get("action")
                if action == "ping":
                    await ws.send_json({"action": "pong"})
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        pass
    finally:
        async with _lock:
            _connections["futures"].pop(ws, None)
        logger.info("Futures WS client disconnected (%d remain)", len(_connections["futures"]))


@router.websocket("/ws/btc")
async def ws_btc(ws: WebSocket) -> None:
    await ws.accept()
    client = _Client(ws=ws)
    async with _lock:
        _connections["btc"][ws] = client
    logger.info("BTC WS client connected (%d total)", len(_connections["btc"]))
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
                action = msg.get("action")
                channel = msg.get("channel", "")

                if action == "subscribe" and channel:
                    client.subscriptions.add(channel)
                    await ws.send_json({"action": "subscribed", "channel": channel})
                elif action == "unsubscribe" and channel:
                    client.subscriptions.discard(channel)
                    await ws.send_json({"action": "unsubscribed", "channel": channel})
                elif action == "ping":
                    await ws.send_json({"action": "pong"})
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        pass
    finally:
        async with _lock:
            _connections["btc"].pop(ws, None)
        logger.info("BTC WS client disconnected (%d remain)", len(_connections["btc"]))


@router.websocket("/ws/live")
async def ws_live(ws: WebSocket) -> None:
    """实盘交易实时推送 WebSocket。

    客户端可订阅频道：
    - position_update: 持仓变化
    - trade_fill: 成交回报
    - account_update: 账户资金
    - strategy_status: 策略状态
    - signal_generated: 新信号
    - risk_alert: 风控告警
    """
    await ws.accept()
    client = _Client(ws=ws, subscriptions={"position_update", "trade_fill", "account_update", "strategy_status"})
    async with _lock:
        _connections["live"][ws] = client
    logger.info("Live WS client connected (%d total)", len(_connections["live"]))

    from event_bus import EventBus
    bus = EventBus.get_instance()

    async def _event_forwarder(event: dict[str, Any]) -> None:
        event_type = event.get("type", "")
        if event_type in client.subscriptions:
            await _safe_send(ws, event)

    event_types = LIVE_WS_EVENT_TYPES
    for et in event_types:
        bus.subscribe(et, _event_forwarder)

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
                action = msg.get("action")
                channel = msg.get("channel", "")

                if action == "subscribe" and channel:
                    client.subscriptions.add(channel)
                    await ws.send_json({"action": "subscribed", "channel": channel})
                elif action == "unsubscribe" and channel:
                    client.subscriptions.discard(channel)
                    await ws.send_json({"action": "unsubscribed", "channel": channel})
                elif action == "ping":
                    await ws.send_json({"action": "pong"})
                elif action == "get_recent":
                    events = bus.get_recent_events(channel or None, limit=50)
                    await ws.send_json({"action": "recent_events", "events": events})
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        pass
    finally:
        for et in event_types:
            bus.unsubscribe(et, _event_forwarder)
        async with _lock:
            _connections["live"].pop(ws, None)
        logger.info("Live WS client disconnected (%d remain)", len(_connections["live"]))


async def broadcast(group: str, data: dict[str, Any], channel: str | None = None) -> None:
    """Broadcast a message to connected clients. If channel is set, only send to subscribers."""
    async with _lock:
        targets = list(_connections.get(group, {}).values())
    dead: list[WebSocket] = []
    for client in targets:
        if channel and client.subscriptions and channel not in client.subscriptions:
            continue
        if not await _safe_send(client.ws, data):
            dead.append(client.ws)
    if dead:
        async with _lock:
            group_conns = _connections.get(group, {})
            for ws in dead:
                group_conns.pop(ws, None)


async def broadcast_live_event(event_type: str, data: dict[str, Any]) -> None:
    """通过 EventBus 发布事件并推送给所有 live WebSocket 客户端。"""
    from event_bus import EventBus
    bus = EventBus.get_instance()
    await bus.emit(event_type, data)
