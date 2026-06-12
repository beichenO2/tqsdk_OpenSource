"""PolarPrivate credential integration tests (mocked HTTP)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from security.privportal import (
    AsyncPrivPortalClient,
    ExchangeKeys,
    PrivPortalClient,
    TqSdkKeys,
    create_exchange_credentials,
)


@pytest.fixture
def mock_httpx_client():
    """Patch httpx.Client to avoid real HTTP calls."""
    with patch("security.privportal.httpx.Client") as mock_cls:
        client_instance = MagicMock()
        mock_cls.return_value = client_instance
        yield client_instance


def _mock_response(json_data, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    resp.cookies = MagicMock()
    resp.cookies.jar = []
    return resp


class TestPrivPortalClient:
    def test_vault_status(self, mock_httpx_client):
        mock_httpx_client.get.return_value = _mock_response({"locked": False})
        client = PrivPortalClient.__new__(PrivPortalClient)
        client._base_url = "http://127.0.0.1:12790"
        client._project_id = None
        client._timeout = 10.0
        client._client = mock_httpx_client
        client._cookie = None

        assert client.is_unlocked() is True
        mock_httpx_client.get.assert_called_with("/api/vault/status")

    def test_unlock_calls_api(self, mock_httpx_client):
        mock_httpx_client.get.return_value = _mock_response({"locked": True})
        mock_httpx_client.post.return_value = _mock_response({"status": "unlocked"})

        client = PrivPortalClient.__new__(PrivPortalClient)
        client._base_url = "http://127.0.0.1:12790"
        client._project_id = None
        client._timeout = 10.0
        client._client = mock_httpx_client
        client._cookie = None

        client.unlock("my_password")
        mock_httpx_client.post.assert_called_with(
            "/api/vault/unlock",
            json={"master_password": "my_password"},
        )

    def test_get_secret_value(self, mock_httpx_client):
        mock_httpx_client.get.return_value = _mock_response({
            "items": [
                {"id": "s1", "key": "exchange.weex.api_key", "enabled": True},
                {"id": "s2", "key": "exchange.weex.api_secret", "enabled": True},
            ],
            "total": 2,
        })
        mock_httpx_client.post.return_value = _mock_response({"value": "my-api-key-123"})

        client = PrivPortalClient.__new__(PrivPortalClient)
        client._base_url = "http://127.0.0.1:12790"
        client._project_id = None
        client._timeout = 10.0
        client._client = mock_httpx_client
        client._cookie = None

        val = client._get_secret_value("exchange.weex.api_key")
        assert val == "my-api-key-123"

    def test_get_exchange_keys_raises_on_missing(self, mock_httpx_client):
        mock_httpx_client.get.return_value = _mock_response({"items": [], "total": 0})

        client = PrivPortalClient.__new__(PrivPortalClient)
        client._base_url = "http://127.0.0.1:12790"
        client._project_id = None
        client._timeout = 10.0
        client._client = mock_httpx_client
        client._cookie = None

        with pytest.raises(KeyError, match="Exchange credentials not found"):
            client.get_exchange_keys("weex")


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


class TestCreateExchangeCredentials:
    def test_weex_conversion(self):
        keys = ExchangeKeys(
            exchange="weex",
            api_key="wk",
            api_secret="ws",
            passphrase="wp",
            testnet=False,
        )
        creds = create_exchange_credentials(keys)
        assert creds.exchange.value == "WEEX"
        assert creds.api_key == "wk"
        assert creds.api_secret == "ws"
        assert creds.passphrase == "wp"

    def test_binance_conversion(self):
        keys = ExchangeKeys(exchange="binance", api_key="bk", api_secret="bs")
        creds = create_exchange_credentials(keys)
        assert creds.exchange.value == "BINANCE"

    def test_unknown_exchange_raises(self):
        keys = ExchangeKeys(exchange="unknown_xyz", api_key="k", api_secret="s")
        with pytest.raises(ValueError, match="Unknown exchange"):
            create_exchange_credentials(keys)

    def test_list_configured_exchanges(self, mock_httpx_client):
        mock_httpx_client.get.return_value = _mock_response({
            "items": [
                {"id": "1", "key": "exchange.weex.api_key"},
                {"id": "2", "key": "exchange.weex.api_secret"},
                {"id": "3", "key": "exchange.binance.api_key"},
                {"id": "4", "key": "exchange.tqsdk.account"},
            ],
            "total": 4,
        })

        client = PrivPortalClient.__new__(PrivPortalClient)
        client._base_url = "http://127.0.0.1:12790"
        client._project_id = None
        client._timeout = 10.0
        client._client = mock_httpx_client
        client._cookie = None

        exchanges = client.list_configured_exchanges()
        assert exchanges == ["binance", "tqsdk", "weex"]
