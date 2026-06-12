"""Security configuration via environment variables and .env files."""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

# Patterns that should appear in a repo .gitignore to avoid committing secrets.
_RECOMMENDED_GITIGNORE_FRAGMENTS: tuple[str, ...] = (
    ".env",
    ".env.local",
    ".env.*.local",
    "*.pem",
    "*.key",
    "*.p12",
    "secrets/",
    ".trading/",
    "*.enc",
    "vault.enc",
)

# Env var name substrings that usually carry secrets; values are heuristically checked.
_SENSITIVE_KEY_MARKERS: tuple[str, ...] = (
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "API_KEY",
    "APIKEY",
    "TOKEN",
    "PRIVATE_KEY",
    "BEARER",
    "CREDENTIAL",
)

_PLACEHOLDER_VALUES: frozenset[str] = frozenset(
    {
        "",
        "changeme",
        "change_me",
        "placeholder",
        "your_key_here",
        "xxx",
        "todo",
        "example",
        "test",
    }
)


class EncryptionBackend(str, Enum):
    FERNET = "fernet"
    AES_GCM = "aes-gcm"


class SecuritySettings(BaseSettings):
    """Central configuration for the security package.

    All values can be overridden via environment variables prefixed with
    ``TRADING_SECURITY_``.  A ``.env`` file in the project root is loaded
    automatically when present.
    """

    model_config = {"env_prefix": "TRADING_SECURITY_", "env_file": ".env"}

    encryption_backend: EncryptionBackend = EncryptionBackend.FERNET
    master_key_env_var: str = Field(
        default="TRADING_MASTER_KEY",
        description="Name of the env-var that holds the hex-encoded master key",
    )
    vault_path: Path = Field(
        default=Path.home() / ".trading" / "vault.enc",
        description="Path to the encrypted credential vault file",
    )
    key_rotation_days: int = Field(default=90, ge=1)
    max_api_keys_per_user: int = Field(default=10, ge=1)
    sanitize_patterns_extra: list[str] = Field(
        default_factory=list,
        description="Additional regex patterns to redact in logs",
    )


def verify_gitignore_covers_sensitive_files(gitignore_path: Path) -> tuple[bool, list[str]]:
    """Return ``(ok, missing_patterns)`` for recommended secret-related .gitignore rules."""
    if not gitignore_path.is_file():
        return False, list(_RECOMMENDED_GITIGNORE_FRAGMENTS)

    content = gitignore_path.read_text(encoding="utf-8", errors="replace")
    lines = {ln.strip() for ln in content.splitlines() if ln.strip() and not ln.strip().startswith("#")}
    missing: list[str] = []
    for frag in _RECOMMENDED_GITIGNORE_FRAGMENTS:
        if frag.endswith("/"):
            if not any(
                ln == frag
                or ln.rstrip("/") == frag.rstrip("/")
                or frag.rstrip("/") in ln
                for ln in lines
            ):
                missing.append(frag)
        elif frag.startswith("*"):
            stem = frag.lstrip("*")
            if not any(ln == frag or ln.endswith(stem) for ln in lines):
                missing.append(frag)
        else:
            if frag not in lines:
                missing.append(frag)
    return (len(missing) == 0, missing)


def scan_dotenv_for_exposed_secrets(
    env_path: Path,
    *,
    min_suspicious_value_len: int = 12,
) -> list[str]:
    """Heuristic scan of a ``.env`` file for lines that look like real secrets.

    Returns human-readable issue strings (empty if nothing suspicious). Does not
    prove a file is safe; intended for local / CI guardrails.
    """
    issues: list[str] = []
    if not env_path.is_file():
        return issues

    for raw_line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not value:
            continue
        upper_key = key.upper()
        if not any(marker in upper_key for marker in _SENSITIVE_KEY_MARKERS):
            continue
        if value.startswith("${") and "}" in value:
            continue  # defer to runtime env
        if value.lower() in _PLACEHOLDER_VALUES:
            continue
        if len(value) < min_suspicious_value_len:
            continue
        # Short obvious placeholders
        if re.match(r"^<[^>]+>$", value):
            continue
        issues.append(
            f"Sensitive-looking key {key!r} has a non-placeholder value (length {len(value)}); "
            "prefer env indirection or a secrets manager."
        )
    return issues


def validate_env_file_for_secrets(env_path: Path) -> list[str]:
    """Alias for :func:`scan_dotenv_for_exposed_secrets` (API clarity)."""
    return scan_dotenv_for_exposed_secrets(env_path)
