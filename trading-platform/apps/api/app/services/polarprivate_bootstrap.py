"""PolarPrivate integration for API startup — WEEX via B-class sign() only."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from broker_crypto import BTCBrokerManager
    from security.privportal import PrivPortalClient

logger = logging.getLogger(__name__)

_SIGN_BINDINGS: dict[str, tuple[str, str]] = {
    "weex": ("weex", os.getenv("WEEX_SIGN_BINDING", "exchange.weex")),
}


def open_privportal_client() -> PrivPortalClient | None:
    """Return a PolarPrivate client when vault is unlocked."""
    try:
        from security.privportal import PrivPortalClient

        client = PrivPortalClient()
        if not client.is_unlocked():
            logger.warning("PolarPrivate vault locked — WEEX sign() unavailable")
            client.close()
            return None
        return client
    except Exception as exc:
        logger.warning("PolarPrivate unavailable (%s)", exc)
        return None


async def _try_register_weex_via_sign(manager, pp: PrivPortalClient, testnet: bool) -> bool:
    from broker_crypto.models import Exchange, ExchangeCredentials
    from broker_crypto.weex import WEEXAdapter

    provider, binding = _SIGN_BINDINGS["weex"]
    try:
        pp.sign(
            provider,
            "rest",
            binding=binding,
            method="GET",
            path="/api/v3/account/",
        )
    except Exception as exc:
        logger.debug("WEEX B-class sign probe failed (%s)", exc)
        return False

    placeholder = ExchangeCredentials(
        exchange=Exchange.WEEX,
        api_key="polarprivate-sign",
        api_secret="polarprivate-sign",
        passphrase="",
        testnet=testnet,
    )
    adapter = WEEXAdapter(placeholder, sign_client=pp, sign_binding=binding)
    try:
        await manager.register_adapter(Exchange.WEEX, adapter)
    except Exception as exc:
        logger.debug("WEEX register via PolarPrivate sign failed (%s)", exc)
        return False
    logger.info("BTCBrokerManager: WEEX via PolarPrivate B-class sign (binding=%s)", binding)
    return True


async def init_btc_broker() -> BTCBrokerManager | None:
    """Initialize WEEX via PolarPrivate B-class sign()."""
    try:
        from broker_crypto import BTCBrokerManager

        manager = BTCBrokerManager()
        is_testnet = os.getenv("CRYPTO_TESTNET", "false").lower() == "true"
        pp = open_privportal_client()

        if pp and await _try_register_weex_via_sign(manager, pp, is_testnet):
            logger.info("BTCBrokerManager: WEEX connected")
            return manager

        logger.info("BTCBrokerManager: WEEX not connected (PolarPrivate sign unavailable)")
        return manager
    except Exception:
        logger.warning("BTCBrokerManager init failed, BTC routes will return 503", exc_info=True)
        return None
