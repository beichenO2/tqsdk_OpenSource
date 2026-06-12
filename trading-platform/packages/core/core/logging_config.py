"""统一日志配置 — 控制台、可选 JSON lines、可选滚动文件，复用脱敏逻辑。"""
from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from typing import Any

from security.sanitizer import sanitize

_LOGRECORD_STANDARD_KEYS = frozenset(
    {
        "name", "msg", "args", "created", "filename", "funcName",
        "levelname", "levelno", "lineno", "module", "msecs", "pathname",
        "process", "processName", "relativeCreated", "thread", "threadName",
        "exc_info", "exc_text", "stack_info", "message", "taskName",
    }
)

_THIRD_PARTY_QUIET = (
    "urllib3", "urllib3.connectionpool",
    "httpx", "httpcore", "httpcore.connection", "httpcore.http11",
    "watchfiles.main",
)


def _resolve_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), logging.INFO)


def _record_extra(record: logging.LogRecord) -> dict[str, Any]:
    return {
        k: v for k, v in record.__dict__.items()
        if k not in _LOGRECORD_STANDARD_KEYS
    }


def _sanitize_obj(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize(value)
    if isinstance(value, dict):
        return {k: _sanitize_obj(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_obj(v) for v in value]
    return value


class SanitizingFormatter(logging.Formatter):
    """自动脱敏的日志格式化器"""

    def format(self, record: logging.LogRecord) -> str:
        original = super().format(record)
        return sanitize(original)


class JSONFormatter(SanitizingFormatter):
    """结构化 JSON lines 输出，整行经 sanitize 脱敏。"""

    def __init__(self) -> None:
        super().__init__(fmt="%(message)s")

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=UTC).isoformat()
        message = record.getMessage()
        if record.exc_info:
            message = message + "\n" + self.formatException(record.exc_info)
        payload = {
            "timestamp": ts,
            "level": record.levelname,
            "logger": record.name,
            "message": sanitize(message),
            "extra": _sanitize_obj(_record_extra(record)),
        }
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(
    level: str = "INFO",
    json_output: bool = False,
    log_file: str | None = None,
) -> None:
    """配置 root logger：stderr 必连；可选 JSON；可选滚动文件；压低第三方库噪声。"""
    log_level = _resolve_level(level)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(log_level)

    if json_output:
        formatter: logging.Formatter = JSONFormatter()
    else:
        formatter = SanitizingFormatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )

    stream = logging.StreamHandler(sys.stderr)
    stream.setLevel(log_level)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if log_file:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    for name in _THIRD_PARTY_QUIET:
        logging.getLogger(name).setLevel(logging.WARNING)
