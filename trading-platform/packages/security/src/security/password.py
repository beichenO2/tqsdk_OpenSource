"""Password hashing and verification using bcrypt via passlib."""

from __future__ import annotations

import hashlib
import hmac
import os


_SALT_LEN = 32
_ITERATIONS = 600_000  # OWASP 2024 recommendation for PBKDF2-SHA256


def hash_password(password: str) -> str:
    """Hash a password with PBKDF2-HMAC-SHA256 and a random salt.

    Returns a string in the format ``<hex-salt>$<hex-hash>`` suitable for
    database storage.
    """
    salt = os.urandom(_SALT_LEN)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ITERATIONS)
    return f"{salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Verify *password* against a previously hashed value.

    Uses constant-time comparison to prevent timing attacks.
    """
    try:
        salt_hex, hash_hex = stored.split("$", 1)
    except ValueError:
        return False
    salt = bytes.fromhex(salt_hex)
    expected = bytes.fromhex(hash_hex)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ITERATIONS)
    return hmac.compare_digest(dk, expected)
