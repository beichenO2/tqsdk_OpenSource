"""PolarPrivate (PrivPortal) integration — sign-based interface (260505 batch).

After the 260505 plaintext-export-ban refactor, the legacy reveal-based
get_exchange_keys()/get_tqsdk_keys() interfaces are removed. Callers must use
one of the new typed interfaces:

  - sign(provider, action, *, method, path, query, body) → headers dict (B-class)
  - grant_d_class(service_name, caller_executable_sha256) → secrets dict (D-class)

Exchange credentials are stored as PolarPrivate Secrets with hierarchical keys:
    exchange.weex.api_key, exchange.weex.api_secret, exchange.weex.passphrase
    exchange.tqsdk.auth_user, exchange.tqsdk.auth_password, ...
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = os.environ.get("POLARPRIVATE_URL", "http://127.0.0.1:12790")


@dataclass
class ExchangeKeys:
    """Resolved exchange credentials — DEPRECATED. Use sign() instead."""

    exchange: str
    api_key: str
    api_secret: str
    passphrase: str | None = None
    testnet: bool = False
    extra: dict[str, str] = field(default_factory=dict)


@dataclass
class TqSdkKeys:
    """TqSdk credentials — populated via D-class grant for futures login."""

    auth_user: str
    auth_password: str
    broker: str
    account: str
    password: str


class PrivPortalClient:
    """Synchronous client for PolarPrivate sign + D-class APIs.

    Usage::

        client = PrivPortalClient()
        # B-class signing (preferred for HMAC-based protocols):
        headers = client.sign("weex", "rest", method="POST", path="/api/v3/order", body=body_str)
        # D-class controlled grant (for third-party SDK protocols only):
        keys = client.grant_d_class("tqsdk-login", caller_sha256)
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._base_url = (
            base_url or os.getenv("PRIVPORTAL_URL", _DEFAULT_BASE_URL)
        ).rstrip("/")
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=timeout,
            transport=httpx.HTTPTransport(),
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "PrivPortalClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def vault_status(self) -> dict[str, Any]:
        resp = self._client.get("/api/vault/status")
        resp.raise_for_status()
        return resp.json()

    def is_unlocked(self) -> bool:
        return not self.vault_status().get("locked", True)

    def sign(
        self,
        provider: str,
        action: str,
        *,
        binding: str | None = None,
        method: str = "GET",
        path: str = "/",
        query: str = "",
        body: str = "",
        timestamp: str | None = None,
    ) -> dict[str, str]:
        """Sign a request via PolarPrivate B-class HMAC signing service.

        Args:
            provider: weex / feishu-webhook / aliyun-sigv1 / etc.
            action: provider-specific action label (e.g. "rest", "sign").
            binding: Vault binding name resolving to the secret triple.
                     Defaults to provider name when None.
            method/path/query/body: request elements to include in the signature.
            timestamp: optional explicit timestamp; server fills if None.

        Returns:
            Dict of HTTP headers (or signature parameters) to attach to the
            outbound request. Secret material never leaves PolarPrivate.
        """
        payload = {
            "binding": binding or provider,
            "method": method,
            "path": path,
            "query": query,
            "body": body,
        }
        if timestamp is not None:
            payload["timestamp"] = timestamp
        resp = self._client.post(f"/sign/{provider}/{action}", json=payload)
        resp.raise_for_status()
        return resp.json()["headers"]

    def grant_d_class(
        self,
        service_name: str,
        caller_executable_sha256: str | None = None,
    ) -> dict[str, str]:
        """Request one-time plaintext for a third-party SDK via D-class channel.

        Restricted to ~/.privportal/d-class-allowlist.json entries that match
        both service_name and caller_executable_sha256. Agent processes are
        explicitly excluded.

        Args:
            service_name: Allowlist entry name (e.g. "tqsdk-login").
            caller_executable_sha256: SHA256 of the caller's binary. When None,
                computed from sys.executable.

        Returns:
            Dict mapping secret keys to plaintext values. Use immediately,
            do not persist.
        """
        if caller_executable_sha256 is None:
            caller_executable_sha256 = _hash_executable(sys.executable)

        resp = self._client.post(
            "/api/d-class/grant",
            json={
                "service_name": service_name,
                "caller_executable_sha256": caller_executable_sha256,
            },
            headers={"X-Caller-PID": str(os.getpid())},
        )
        if resp.status_code == 403:
            raise PermissionError(
                f"D-class grant denied for service={service_name}. "
                f"Add this binary's SHA256 to ~/.privportal/d-class-allowlist.json"
            )
        resp.raise_for_status()
        return resp.json()["secrets"]

    def get_exchange_keys(self, exchange: str, **_: Any) -> ExchangeKeys:
        """DEPRECATED — reveal-based credential fetch was removed in 260505.

        Migration: use sign() to request signed headers per call.
        Example::

            client.sign("weex", "rest", method="POST", path=path, body=body)
        """
        raise NotImplementedError(
            f"get_exchange_keys('{exchange}') is removed. "
            f"Plaintext exchange secrets no longer leave PolarPrivate. "
            f"Use client.sign('{exchange}', 'rest', method=..., path=..., body=...) "
            f"to obtain signed headers instead."
        )

    def get_tqsdk_keys(self) -> TqSdkKeys:
        """DEPRECATED for trading-platform — use TqSdk Gateway instead.

        Only the gateway process (tqsdk-gateway/) should call grant_d_class().
        """
        secrets = self.grant_d_class("tqsdk-login")

        def _get(key: str) -> str:
            val = secrets.get(key)
            if val is None:
                raise KeyError(
                    f"D-class allowlist entry for tqsdk-login is missing key: {key}. "
                    f"Update allowed_secret_keys in ~/.privportal/d-class-allowlist.json."
                )
            return val

        return TqSdkKeys(
            auth_user=_get("exchange.tqsdk.auth_user"),
            auth_password=_get("exchange.tqsdk.auth_password"),
            broker=_get("exchange.tqsdk.broker"),
            account=_get("exchange.tqsdk.account"),
            password=_get("exchange.tqsdk.password"),
        )


def _hash_executable(path: str) -> str:
    """Compute SHA256 of an executable file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
