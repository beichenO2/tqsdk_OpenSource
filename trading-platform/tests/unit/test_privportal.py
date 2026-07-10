"""PolarPrivate credential integration tests (mocked HTTP).

Targets the post-260505 sign/grant interface: plaintext reveal APIs are
removed; B-class sign() and D-class grant_d_class() are the supported paths.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from security.privportal import (
    ExchangeKeys,
    PrivPortalClient,
    TqSdkKeys,
)


def _make_client(mock_http) -> PrivPortalClient:
    client = PrivPortalClient.__new__(PrivPortalClient)
    client._base_url = "http://127.0.0.1:12790"
    client._client = mock_http
    return client


def _mock_response(json_data, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


@pytest.fixture
def mock_httpx_client():
    return MagicMock()


class TestVaultStatus:
    def test_vault_status(self, mock_httpx_client):
        mock_httpx_client.get.return_value = _mock_response({"locked": False})
        client = _make_client(mock_httpx_client)

        assert client.is_unlocked() is True
        mock_httpx_client.get.assert_called_with("/api/vault/status")

    def test_locked_vault(self, mock_httpx_client):
        mock_httpx_client.get.return_value = _mock_response({"locked": True})
        client = _make_client(mock_httpx_client)

        assert client.is_unlocked() is False


class TestSign:
    def test_sign_posts_payload_and_returns_headers(self, mock_httpx_client):
        mock_httpx_client.post.return_value = _mock_response(
            {"headers": {"X-Signature": "abc123", "X-Timestamp": "1700000000"}}
        )
        client = _make_client(mock_httpx_client)

        headers = client.sign(
            "weex", "rest", method="POST", path="/api/v3/order", body='{"qty":1}'
        )

        assert headers == {"X-Signature": "abc123", "X-Timestamp": "1700000000"}
        mock_httpx_client.post.assert_called_once_with(
            "/sign/weex/rest",
            json={
                "binding": "weex",
                "method": "POST",
                "path": "/api/v3/order",
                "query": "",
                "body": '{"qty":1}',
            },
        )

    def test_sign_custom_binding_and_timestamp(self, mock_httpx_client):
        mock_httpx_client.post.return_value = _mock_response({"headers": {}})
        client = _make_client(mock_httpx_client)

        client.sign("weex", "rest", binding="weex-sub1", timestamp="123")

        payload = mock_httpx_client.post.call_args.kwargs["json"]
        assert payload["binding"] == "weex-sub1"
        assert payload["timestamp"] == "123"


class TestGrantDClass:
    def test_grant_returns_secrets(self, mock_httpx_client):
        mock_httpx_client.post.return_value = _mock_response(
            {"secrets": {"exchange.tqsdk.account": "320102"}}
        )
        client = _make_client(mock_httpx_client)

        secrets = client.grant_d_class("tqsdk-login", "deadbeef")

        assert secrets == {"exchange.tqsdk.account": "320102"}
        call = mock_httpx_client.post.call_args
        assert call.args[0] == "/api/d-class/grant"
        assert call.kwargs["json"] == {
            "service_name": "tqsdk-login",
            "caller_executable_sha256": "deadbeef",
        }
        assert "X-Caller-PID" in call.kwargs["headers"]

    def test_grant_denied_raises_permission_error(self, mock_httpx_client):
        mock_httpx_client.post.return_value = _mock_response({}, status_code=403)
        client = _make_client(mock_httpx_client)

        with pytest.raises(PermissionError, match="D-class grant denied"):
            client.grant_d_class("tqsdk-login", "deadbeef")


class TestDeprecatedRevealAPI:
    def test_get_exchange_keys_removed(self, mock_httpx_client):
        client = _make_client(mock_httpx_client)
        with pytest.raises(NotImplementedError, match="removed"):
            client.get_exchange_keys("weex")

    def test_get_tqsdk_keys_via_grant(self, mock_httpx_client):
        mock_httpx_client.post.return_value = _mock_response(
            {
                "secrets": {
                    "exchange.tqsdk.auth_user": "user@example.com",
                    "exchange.tqsdk.auth_password": "authpwd",
                    "exchange.tqsdk.broker": "H海通期货",
                    "exchange.tqsdk.account": "320102",
                    "exchange.tqsdk.password": "123456",
                }
            }
        )
        client = _make_client(mock_httpx_client)

        keys = client.get_tqsdk_keys()
        assert keys.auth_user == "user@example.com"
        assert keys.broker == "H海通期货"
        assert keys.account == "320102"

    def test_get_tqsdk_keys_missing_key_raises(self, mock_httpx_client):
        mock_httpx_client.post.return_value = _mock_response(
            {"secrets": {"exchange.tqsdk.auth_user": "user@example.com"}}
        )
        client = _make_client(mock_httpx_client)

        with pytest.raises(KeyError, match="missing key"):
            client.get_tqsdk_keys()


class TestExchangeKeys:
    def test_dataclass_fields(self):
        keys = ExchangeKeys(
            exchange="weex",
            api_key="key123",
            api_secret="secret456",
            passphrase="pass789",
            testnet=True,
        )
        assert keys.exchange == "weex"
        assert keys.api_key == "key123"
        assert keys.passphrase == "pass789"
        assert keys.testnet is True

    def test_defaults(self):
        keys = ExchangeKeys(exchange="binance", api_key="k", api_secret="s")
        assert keys.passphrase is None
        assert keys.testnet is False
        assert keys.extra == {}


class TestTqSdkKeys:
    def test_dataclass_fields(self):
        keys = TqSdkKeys(
            auth_user="user@example.com",
            auth_password="authpwd",
            broker="H海通期货",
            account="320102",
            password="123456",
        )
        assert keys.auth_user == "user@example.com"
        assert keys.broker == "H海通期货"
        assert keys.account == "320102"
