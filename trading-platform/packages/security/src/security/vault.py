"""Encrypted credential vault — stores broker API keys and secrets on disk."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from security.encryption import EncryptionService

logger = logging.getLogger(__name__)


class CredentialEntry(BaseModel):
    """A single credential stored in the vault."""

    name: str
    broker: str
    api_key: str
    api_secret: str
    extra: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    rotated_at: datetime | None = None

    def masked(self) -> dict[str, Any]:
        """Return a representation with secrets partially masked."""
        return {
            "name": self.name,
            "broker": self.broker,
            "api_key": self.api_key[:4] + "****" + self.api_key[-4:] if len(self.api_key) > 8 else "****",
            "api_secret": "****",
            "created_at": self.created_at.isoformat(),
        }


class CredentialVault:
    """File-backed encrypted credential store.

    The vault serialises all entries to JSON, encrypts the blob with
    ``EncryptionService``, and writes it to ``vault_path``.
    """

    def __init__(self, encryption: EncryptionService, vault_path: Path) -> None:
        self._enc = encryption
        self._path = vault_path
        self._entries: dict[str, CredentialEntry] = {}
        if self._path.exists():
            self._load()

    def _load(self) -> None:
        raw = self._path.read_bytes()
        plaintext = self._enc.decrypt(raw)
        data = json.loads(plaintext)
        self._entries = {k: CredentialEntry.model_validate(v) for k, v in data.items()}
        logger.info("Vault loaded with %d entries", len(self._entries))

    def _persist(self) -> None:
        payload = {k: v.model_dump(mode="json") for k, v in self._entries.items()}
        raw = json.dumps(payload, default=str).encode("utf-8")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_bytes(self._enc.encrypt(raw))

    def add(self, entry: CredentialEntry) -> None:
        if entry.name in self._entries:
            raise KeyError(f"Credential '{entry.name}' already exists; use update()")
        self._entries[entry.name] = entry
        self._persist()
        logger.info("Added credential '%s'", entry.name)

    def update(self, entry: CredentialEntry) -> None:
        entry.rotated_at = datetime.now(timezone.utc)
        self._entries[entry.name] = entry
        self._persist()
        logger.info("Updated credential '%s'", entry.name)

    def get(self, name: str) -> CredentialEntry:
        if name not in self._entries:
            raise KeyError(f"Credential '{name}' not found")
        return self._entries[name]

    def remove(self, name: str) -> None:
        if name not in self._entries:
            raise KeyError(f"Credential '{name}' not found")
        del self._entries[name]
        self._persist()
        logger.info("Removed credential '%s'", name)

    def list_names(self) -> list[str]:
        return list(self._entries.keys())

    def list_masked(self) -> list[dict[str, Any]]:
        return [e.masked() for e in self._entries.values()]
