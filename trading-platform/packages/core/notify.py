"""Feishu notification bridge — subscribes to EventBus trading alerts."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

import httpx

from event_bus import EventBus

logger = logging.getLogger(__name__)

DEFAULT_CHANNELS = frozenset({"risk_alert", "order_rejected"})


class FeishuNotifier:
    """Push selected EventBus events to Feishu webhook (optional)."""

    def __init__(
        self,
        *,
        channels: frozenset[str] | set[str] | None = None,
        throttle_seconds: float = 60.0,
        webhook_url: Optional[str] = None,
    ) -> None:
        self._channels = frozenset(channels) if channels is not None else DEFAULT_CHANNELS
        self._throttle_seconds = throttle_seconds
        self._webhook_url = (webhook_url or os.getenv("FEISHU_WEBHOOK_URL", "")).strip()
        self._last_sent: dict[tuple[str, str], float] = {}
        self._attached = False

        if not self._webhook_url:
            logger.info("FeishuNotifier disabled — FEISHU_WEBHOOK_URL not configured")
        else:
            logger.info("FeishuNotifier enabled for channels: %s", sorted(self._channels))

    @property
    def enabled(self) -> bool:
        return bool(self._webhook_url)

    def attach(self, bus: EventBus) -> None:
        if self._attached:
            return
        for channel in self._channels:
            bus.subscribe(channel, self._on_event)
        self._attached = True

    async def _on_event(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return
        event_type = event.get("type", "")
        data = event.get("data") or {}
        key = self._throttle_key(event_type, data)
        if self._is_throttled(event_type, key):
            return
        text = self._format_message(event_type, data)
        await self._send(text)
        self._last_sent[(event_type, key)] = time.monotonic()

    def _throttle_key(self, event_type: str, data: dict[str, Any]) -> str:
        symbol = data.get("symbol") or ""
        rule = (
            data.get("limit")
            or data.get("rule")
            or data.get("source")
            or data.get("order_id")
            or ""
        )
        return f"{symbol}:{rule}"

    def _is_throttled(self, event_type: str, key: str) -> bool:
        last = self._last_sent.get((event_type, key))
        if last is None:
            return False
        return (time.monotonic() - last) < self._throttle_seconds

    def _format_message(self, event_type: str, data: dict[str, Any]) -> str:
        symbol = data.get("symbol") or "?"
        reason = data.get("reason") or data.get("message") or ""
        limit = data.get("limit") or data.get("source") or ""
        level = data.get("level") or ""
        lines = [f"[TqTrader/{event_type}] {symbol}"]
        if limit:
            lines.append(f"规则: {limit}")
        if level:
            lines.append(f"级别: {level}")
        if reason:
            lines.append(f"原因: {reason}")
        return "\n".join(lines)

    async def _send(self, text: str) -> None:
        if not self.enabled:
            return
        payload = {"msg_type": "text", "content": {"text": text}}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(self._webhook_url, json=payload)
                resp.raise_for_status()
                body = resp.json()
                if body.get("code") == 0 or body.get("StatusCode") == 0:
                    logger.info("Feishu alert sent")
                else:
                    logger.warning("Feishu webhook rejected: %s", body)
        except Exception:
            logger.exception("Feishu webhook send failed")
