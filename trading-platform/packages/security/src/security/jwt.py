"""JWT token issuance and verification (HS256 / RS256)."""

from __future__ import annotations

import hashlib
import hmac
import json
import base64
import time
from typing import Any


class JWTError(Exception):
    """Raised on token verification failures."""


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    padding = 4 - len(s) % 4
    if padding != 4:
        s += "=" * padding
    return base64.urlsafe_b64decode(s)


class JWTService:
    """Minimal HS256 JWT implementation with no third-party dependency.

    For production RS256 with key rotation, swap in PyJWT or
    python-jose under the same interface.
    """

    def __init__(self, secret: str, *, algorithm: str = "HS256", default_ttl: int = 3600) -> None:
        if algorithm != "HS256":
            raise NotImplementedError(f"Only HS256 is built-in; got {algorithm}")
        self._secret = secret.encode("utf-8")
        self._default_ttl = default_ttl

    def _sign(self, payload_b64: str, header_b64: str) -> str:
        msg = f"{header_b64}.{payload_b64}".encode("ascii")
        sig = hmac.new(self._secret, msg, hashlib.sha256).digest()
        return _b64url_encode(sig)

    def create_token(
        self,
        subject: str,
        scopes: list[str] | None = None,
        extra: dict[str, Any] | None = None,
        ttl: int | None = None,
    ) -> str:
        """Create a signed JWT for *subject*."""
        now = int(time.time())
        payload: dict[str, Any] = {
            "sub": subject,
            "iat": now,
            "exp": now + (ttl or self._default_ttl),
        }
        if scopes:
            payload["scopes"] = scopes
        if extra:
            payload.update(extra)

        header_b64 = _b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
        payload_b64 = _b64url_encode(json.dumps(payload).encode())
        signature = self._sign(payload_b64, header_b64)
        return f"{header_b64}.{payload_b64}.{signature}"

    def verify_token(self, token: str) -> dict[str, Any]:
        """Verify signature and expiry.  Returns the payload dict."""
        parts = token.split(".")
        if len(parts) != 3:
            raise JWTError("Malformed token")

        header_b64, payload_b64, sig_b64 = parts
        expected_sig = self._sign(payload_b64, header_b64)
        if not hmac.compare_digest(sig_b64, expected_sig):
            raise JWTError("Invalid signature")

        payload = json.loads(_b64url_decode(payload_b64))
        if "exp" in payload and payload["exp"] < int(time.time()):
            raise JWTError("Token expired")

        return payload

    def decode_unverified(self, token: str) -> dict[str, Any]:
        """Decode payload without verifying — useful for debugging only."""
        parts = token.split(".")
        if len(parts) != 3:
            raise JWTError("Malformed token")
        return json.loads(_b64url_decode(parts[1]))
