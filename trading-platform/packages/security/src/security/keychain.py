"""macOS / system keychain credential storage via the ``keyring`` library."""

from __future__ import annotations

import keyring
from keyring.errors import PasswordDeleteError


def store_credential(service: str, username: str, password: str) -> None:
    """Persist *password* for *username* under application *service* in the OS keychain."""
    keyring.set_password(service, username, password)


def get_credential(service: str, username: str) -> str:
    """Return the stored password for *username* under *service*.

    Raises ``KeyError`` if no credential exists.
    """
    value = keyring.get_password(service, username)
    if value is None:
        raise KeyError(f"No credential stored for service={service!r} username={username!r}")
    return value


def delete_credential(service: str, username: str) -> None:
    """Remove the credential for *username* under *service*.

    Missing entries are ignored (idempotent).
    """
    try:
        keyring.delete_password(service, username)
    except PasswordDeleteError:
        pass
