"""API key lifecycle — generation, validation, rotation, and revocation."""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_PREFIX = "tpk"  # trading-platform-key
_KEY_BYTES = 32


class KeyStatus(str, Enum):
    ACTIVE = "active"
    REVOKED = "revoked"
    EXPIRED = "expired"


class APIKeyRecord(BaseModel):
    """Metadata for an issued API key (the raw key is never stored)."""

    key_id: str
    prefix: str
    key_hash: str
    owner: str
    label: str = ""
    scopes: list[str] = Field(default_factory=lambda: ["read"])
    status: KeyStatus = KeyStatus.ACTIVE
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    last_used_at: datetime | None = None


class APIKeyManager:
    """Issue and verify API keys for platform consumers.

    Keys follow the format ``tpk_<random-hex>``.  Only the SHA-256 hash
    is persisted; the raw secret is returned exactly once on creation.
    """

    def __init__(self) -> None:
        self._store: dict[str, APIKeyRecord] = {}

    @staticmethod
    def _hash(raw: str) -> str:
        return hashlib.sha256(raw.encode()).hexdigest()

    def issue(self, owner: str, label: str = "", scopes: list[str] | None = None) -> tuple[str, APIKeyRecord]:
        """Generate a new API key.  Returns ``(raw_key, record)``."""
        raw = f"{_PREFIX}_{secrets.token_hex(_KEY_BYTES)}"
        key_id = secrets.token_hex(8)
        record = APIKeyRecord(
            key_id=key_id,
            prefix=raw[:8],
            key_hash=self._hash(raw),
            owner=owner,
            label=label,
            scopes=scopes or ["read"],
        )
        self._store[key_id] = record
        logger.info("Issued API key %s for owner=%s", key_id, owner)
        return raw, record

    def verify(self, raw_key: str) -> APIKeyRecord | None:
        """Validate a raw key and return its record if valid."""
        h = self._hash(raw_key)
        for record in self._store.values():
            if hmac.compare_digest(record.key_hash, h):
                if record.status != KeyStatus.ACTIVE:
                    return None
                if record.expires_at and datetime.now(timezone.utc) > record.expires_at:
                    record.status = KeyStatus.EXPIRED
                    return None
                record.last_used_at = datetime.now(timezone.utc)
                return record
        return None

    def revoke(self, key_id: str) -> None:
        if key_id not in self._store:
            raise KeyError(f"Key ID '{key_id}' not found")
        self._store[key_id].status = KeyStatus.REVOKED
        logger.info("Revoked API key %s", key_id)

    def list_keys(self, owner: str | None = None) -> list[APIKeyRecord]:
        keys = list(self._store.values())
        if owner:
            keys = [k for k in keys if k.owner == owner]
        return keys
