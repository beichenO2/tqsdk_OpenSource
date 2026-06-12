"""Log sanitizer — redacts sensitive tokens, keys, and PII from text."""

from __future__ import annotations

import re
from typing import Sequence

_BUILTIN_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)(api[_-]?key|apikey|api[_-]?secret|secret[_-]?key)\s*[:=]\s*\S+"),
    re.compile(r"(?i)(password|passwd|pwd)\s*[:=]\s*\S+"),
    re.compile(r"(?i)(token|bearer)\s+[A-Za-z0-9_\-\.]+"),
    re.compile(r"(?i)(authorization)\s*:\s*\S+"),
    re.compile(r"\bsk-[a-zA-Z0-9]{20,}\b"),  # OpenAI-style API keys
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key id
    re.compile(r"\bASIA[0-9A-Z]{16}\b"),  # AWS STS temporary key id
    re.compile(r"(?i)(account|acct)\s*[#:]?\s*\d{6,}\b"),  # labeled account numbers
    re.compile(r"\b\d{3}[- ]?\d{3}[- ]?\d{4}\b"),  # US phone / short account-like runs
    re.compile(r"\b[A-Za-z0-9]{32,}\b"),  # long opaque tokens
    re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"),  # card-like numbers
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # email
]

_REDACTED = "[REDACTED]"


class LogSanitizer:
    """Strips sensitive patterns from arbitrary text.

    Initialise with optional extra regex strings to extend the builtin
    pattern set.  Call ``sanitize(text)`` to return a safe version.
    """

    def __init__(self, extra_patterns: Sequence[str] = ()) -> None:
        self._patterns = list(_BUILTIN_PATTERNS)
        for p in extra_patterns:
            self._patterns.append(re.compile(p))

    def sanitize(self, text: str) -> str:
        for pat in self._patterns:
            text = pat.sub(_REDACTED, text)
        return text


_default_sanitizer = LogSanitizer()


def sanitize(text: str) -> str:
    """Module-level convenience using default patterns."""
    return _default_sanitizer.sanitize(text)


def sanitize_log(text: str) -> str:
    """Redact secrets from *text* for safe logging (same rules as :func:`sanitize`)."""
    return sanitize(text)
