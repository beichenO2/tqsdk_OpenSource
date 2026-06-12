"""Unit tests for ``core.logging_config.setup_logging`` and formatters."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

_repo = Path(__file__).resolve().parents[2]
for p in [_repo, _repo / "apps" / "api", _repo / "packages" / "core", _repo / "packages" / "security" / "src", _repo / "packages"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytest

import core.logging_config as lc
from core.logging_config import JSONFormatter, SanitizingFormatter, setup_logging


@pytest.fixture(autouse=True)
def _restore_root_logging() -> None:
    root = logging.getLogger()
    before_handlers = list(root.handlers)
    before_level = root.level
    yield
    root.handlers.clear()
    for h in before_handlers:
        root.addHandler(h)
    root.setLevel(before_level)


def test_setup_logging_configures_root_logger_level() -> None:
    setup_logging(level="ERROR")
    root = logging.getLogger()
    assert root.level == logging.ERROR


def test_setup_logging_default_root_level_is_info() -> None:
    setup_logging()
    assert logging.getLogger().level == logging.INFO


def test_setup_logging_adds_stream_handler_to_root() -> None:
    setup_logging()
    root = logging.getLogger()
    assert len(root.handlers) >= 1
    assert any(isinstance(h, logging.StreamHandler) for h in root.handlers)


def test_setup_logging_stream_handler_targets_stderr() -> None:
    setup_logging()
    root = logging.getLogger()
    stream_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler)]
    assert stream_handlers
    assert stream_handlers[0].stream is sys.stderr


def test_setup_logging_json_output_uses_json_formatter_on_stream() -> None:
    setup_logging(json_output=True)
    root = logging.getLogger()
    sh = next(h for h in root.handlers if isinstance(h, logging.StreamHandler))
    assert isinstance(sh.formatter, JSONFormatter)


def test_setup_logging_plain_uses_sanitizing_formatter_on_stream() -> None:
    setup_logging(json_output=False)
    root = logging.getLogger()
    sh = next(h for h in root.handlers if isinstance(h, logging.StreamHandler))
    assert isinstance(sh.formatter, SanitizingFormatter)
    assert not isinstance(sh.formatter, JSONFormatter)


def test_setup_logging_log_file_adds_rotating_file_handler(tmp_path: Path) -> None:
    log_path = tmp_path / "app.log"
    setup_logging(log_file=str(log_path))
    root = logging.getLogger()
    from logging.handlers import RotatingFileHandler

    assert any(isinstance(h, RotatingFileHandler) for h in root.handlers)


def test_setup_logging_log_file_writes_rotated_path(tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"
    setup_logging(level="INFO", log_file=str(log_path))
    logging.getLogger("tp-test").info("hello-file")
    text = log_path.read_text(encoding="utf-8")
    assert "hello-file" in text


def test_rotating_file_handler_uses_utf8_encoding(tmp_path: Path) -> None:
    log_path = tmp_path / "utf8.log"
    setup_logging(log_file=str(log_path))
    from logging.handlers import RotatingFileHandler

    fh = next(h for h in logging.getLogger().handlers if isinstance(h, RotatingFileHandler))
    assert fh.encoding == "utf-8"


def test_sanitizing_formatter_sanitizes_output() -> None:
    fmt = SanitizingFormatter("%(message)s")
    record = logging.LogRecord(
        name="n",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="password: secret123",
        args=(),
        exc_info=None,
    )
    with patch("core.logging_config.sanitize", side_effect=lambda s: f"SAN({s})"):
        out = fmt.format(record)
    assert out.startswith("SAN(")


def test_json_formatter_produces_valid_json() -> None:
    fmt = JSONFormatter()
    record = logging.LogRecord(
        name="my.logger",
        level=logging.WARNING,
        pathname="x.py",
        lineno=10,
        msg="line",
        args=(),
        exc_info=None,
    )
    line = fmt.format(record)
    data = json.loads(line)
    assert data["level"] == "WARNING"
    assert data["logger"] == "my.logger"
    assert data["message"] == "line"
    assert "timestamp" in data
    assert "extra" in data


def test_json_formatter_expected_field_keys() -> None:
    fmt = JSONFormatter()
    record = logging.LogRecord(
        name="n", level=logging.INFO, pathname="", lineno=0, msg="m", args=(), exc_info=None
    )
    data = json.loads(fmt.format(record))
    assert set(data.keys()) == {"timestamp", "level", "logger", "message", "extra"}


def test_json_formatter_extra_from_log_record() -> None:
    fmt = JSONFormatter()
    record = logging.LogRecord(
        name="n", level=logging.INFO, pathname="", lineno=0, msg="m", args=(), exc_info=None
    )
    record.correlation_id = "abc-123"
    payload = json.loads(fmt.format(record))
    assert payload["extra"].get("correlation_id") == "abc-123"


def test_json_formatter_timestamp_is_iso8601_utc() -> None:
    fmt = JSONFormatter()
    record = logging.LogRecord(
        name="n", level=logging.INFO, pathname="", lineno=0, msg="m", args=(), exc_info=None
    )
    record.created = datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC).timestamp()
    ts = json.loads(fmt.format(record))["timestamp"]
    assert ts.endswith("+00:00") or ts.endswith("Z")


def test_json_formatter_sanitizes_message_via_security_sanitizer() -> None:
    fmt = JSONFormatter()
    record = logging.LogRecord(
        name="n", level=logging.INFO, pathname="", lineno=0, msg="token abc", args=(), exc_info=None
    )
    with patch("core.logging_config.sanitize", return_value="REDACTED_LINE"):
        line = fmt.format(record)
    assert json.loads(line)["message"] == "REDACTED_LINE"


def test_resolve_level_accepts_int() -> None:
    assert lc._resolve_level(logging.DEBUG) == logging.DEBUG


def test_resolve_level_accepts_string_info() -> None:
    assert lc._resolve_level("info") == logging.INFO


def test_resolve_level_accepts_string_upper_debug() -> None:
    assert lc._resolve_level("DEBUG") == logging.DEBUG


def test_resolve_level_unknown_string_falls_back_to_info() -> None:
    assert lc._resolve_level("NOT_A_REAL_LEVEL_NAME_XYZ") == logging.INFO


def test_third_party_loggers_quieted_to_warning() -> None:
    setup_logging()
    for name in lc._THIRD_PARTY_QUIET:
        assert logging.getLogger(name).level == logging.WARNING


def test_multiple_setup_logging_calls_clear_previous_handlers() -> None:
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    try:
        root.handlers.clear()
        setup_logging()
        first_handlers = root.handlers[:]
        first_count = len(first_handlers)
        setup_logging()
        second_handlers = root.handlers[:]
        assert len(second_handlers) == first_count
        assert all(h not in first_handlers for h in second_handlers)
    finally:
        root.handlers[:] = saved_handlers
        root.level = saved_level


def test_setup_logging_with_file_clears_handlers_between_calls(tmp_path: Path) -> None:
    p1 = tmp_path / "a.log"
    p2 = tmp_path / "b.log"
    setup_logging(log_file=str(p1))
    n1 = len(logging.getLogger().handlers)
    setup_logging(log_file=str(p2))
    n2 = len(logging.getLogger().handlers)
    assert n1 == n2


def test_json_formatter_with_exc_info_contains_trace() -> None:
    fmt = JSONFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.LogRecord(
            name="n",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="failed",
            args=(),
            exc_info=sys.exc_info(),
        )
    line = fmt.format(record)
    data = json.loads(line)
    assert "boom" in data["message"]


def test_record_extra_helper_filters_standard_keys() -> None:
    record = logging.LogRecord(
        name="n", level=logging.INFO, pathname="", lineno=0, msg="m", args=(), exc_info=None
    )
    record.custom_key = 42
    extra = lc._record_extra(record)
    assert extra.get("custom_key") == 42
    assert "name" not in extra


def test_sanitize_obj_passes_through_int_unchanged() -> None:
    assert lc._sanitize_obj(3) == 3


def test_sanitize_obj_string_calls_sanitize() -> None:
    with patch("core.logging_config.sanitize", return_value="Z"):
        assert lc._sanitize_obj("x") == "Z"


def test_sanitize_obj_dict_values_sanitized_strings() -> None:
    with patch("core.logging_config.sanitize", side_effect=lambda s: "X"):
        out = lc._sanitize_obj({"k": "v"})
    assert out == {"k": "X"}


def test_setup_logging_json_and_file_handlers(tmp_path: Path) -> None:
    log_path = tmp_path / "j.log"
    setup_logging(json_output=True, log_file=str(log_path))
    root = logging.getLogger()
    from logging.handlers import RotatingFileHandler

    assert any(isinstance(h, logging.StreamHandler) for h in root.handlers)
    assert any(isinstance(h, RotatingFileHandler) for h in root.handlers)


def test_stream_handler_level_matches_resolved_level() -> None:
    setup_logging(level="CRITICAL")
    root = logging.getLogger()
    sh = next(h for h in root.handlers if isinstance(h, logging.StreamHandler))
    assert sh.level == logging.CRITICAL


def test_stream_and_file_handlers_share_formatter_type(tmp_path: Path) -> None:
    log_path = tmp_path / "shared.log"
    setup_logging(json_output=True, log_file=str(log_path))
    root = logging.getLogger()
    types_ = {type(h.formatter) for h in root.handlers}
    assert types_ == {JSONFormatter}
