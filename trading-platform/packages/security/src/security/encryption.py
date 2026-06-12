"""Symmetric encryption utilities (Fernet & AES-256-GCM)."""

from __future__ import annotations

import os
import secrets
from typing import Protocol

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

_NONCE_LEN = 12
_AES_KEY_LEN = 32
_DEFAULT_PBKDF2_ITERATIONS = 600_000


def derive_key_from_password(
    password: str,
    *,
    salt: bytes | None = None,
    iterations: int = _DEFAULT_PBKDF2_ITERATIONS,
) -> tuple[bytes, bytes]:
    """Derive a 32-byte AES key from *password* using PBKDF2-HMAC-SHA256.

    Returns ``(key, salt)``. If *salt* is omitted, a random 16-byte salt is generated.
    """
    if salt is None:
        salt = os.urandom(16)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=_AES_KEY_LEN,
        salt=salt,
        iterations=iterations,
    )
    key = kdf.derive(password.encode("utf-8"))
    return key, salt


def encrypt(plaintext: str | bytes, key: bytes) -> bytes:
    """Encrypt with AES-256-GCM. Output format: ``nonce (12 bytes) || ciphertext``."""
    if len(key) != _AES_KEY_LEN:
        raise ValueError("AES-256-GCM requires a 32-byte key")
    raw = plaintext.encode("utf-8") if isinstance(plaintext, str) else plaintext
    gcm = AESGCM(key)
    nonce = os.urandom(_NONCE_LEN)
    return nonce + gcm.encrypt(nonce, raw, None)


def decrypt(encrypted_bytes: bytes, key: bytes) -> bytes:
    """Decrypt a blob produced by :func:`encrypt`."""
    if len(key) != _AES_KEY_LEN:
        raise ValueError("AES-256-GCM requires a 32-byte key")
    if len(encrypted_bytes) < _NONCE_LEN:
        raise ValueError("Ciphertext too short")
    nonce = encrypted_bytes[:_NONCE_LEN]
    ct = encrypted_bytes[_NONCE_LEN:]
    return AESGCM(key).decrypt(nonce, ct, None)


class _Cipher(Protocol):
    def encrypt(self, plaintext: bytes) -> bytes: ...
    def decrypt(self, ciphertext: bytes) -> bytes: ...


class _FernetCipher:
    def __init__(self, key: bytes) -> None:
        self._f = Fernet(key)

    def encrypt(self, plaintext: bytes) -> bytes:
        return self._f.encrypt(plaintext)

    def decrypt(self, ciphertext: bytes) -> bytes:
        return self._f.decrypt(ciphertext)


class _AesGcmCipher:
    def __init__(self, key: bytes) -> None:
        if len(key) != _AES_KEY_LEN:
            raise ValueError("AES-256-GCM requires a 32-byte key")
        self._gcm = AESGCM(key)

    def encrypt(self, plaintext: bytes) -> bytes:
        nonce = os.urandom(_NONCE_LEN)
        ct = self._gcm.encrypt(nonce, plaintext, None)
        return nonce + ct

    def decrypt(self, ciphertext: bytes) -> bytes:
        nonce = ciphertext[:_NONCE_LEN]
        ct = ciphertext[_NONCE_LEN:]
        return self._gcm.decrypt(nonce, ct, None)


class EncryptionService:
    """High-level encryption service supporting multiple backends."""

    def __init__(self, *, backend: str = "fernet", key: bytes | None = None) -> None:
        if backend == "fernet":
            self._key = key or Fernet.generate_key()
            self._cipher: _Cipher = _FernetCipher(self._key)
        elif backend == "aes-gcm":
            self._key = key or secrets.token_bytes(32)
            self._cipher = _AesGcmCipher(self._key)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    @property
    def key(self) -> bytes:
        return self._key

    def encrypt(self, plaintext: str | bytes) -> bytes:
        raw = plaintext.encode("utf-8") if isinstance(plaintext, str) else plaintext
        return self._cipher.encrypt(raw)

    def decrypt(self, ciphertext: bytes) -> bytes:
        return self._cipher.decrypt(ciphertext)

    def decrypt_str(self, ciphertext: bytes) -> str:
        return self.decrypt(ciphertext).decode("utf-8")


class FieldEncryptor:
    """Encrypt / decrypt individual dict fields by name.

    Useful for selectively encrypting PII or credential fields inside
    a larger payload before persisting.
    """

    def __init__(self, service: EncryptionService, fields: set[str]) -> None:
        self._svc = service
        self._fields = fields

    def encrypt_dict(self, data: dict) -> dict:
        out = dict(data)
        for f in self._fields:
            if f in out and isinstance(out[f], (str, bytes)):
                out[f] = self._svc.encrypt(out[f])
        return out

    def decrypt_dict(self, data: dict) -> dict:
        out = dict(data)
        for f in self._fields:
            if f in out and isinstance(out[f], bytes):
                out[f] = self._svc.decrypt_str(out[f])
        return out
