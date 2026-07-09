#!/usr/bin/env python3
"""Verify TqSdk connectivity via the gateway (no plaintext credentials in this process)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import urllib.error
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GATEWAY_URL = os.getenv("TQSDK_GATEWAY_URL", "http://127.0.0.1:12890").rstrip("/")


def _get(path: str) -> dict:
    with urllib.request.urlopen(f"{GATEWAY_URL}{path}", timeout=30) as resp:
        return json.loads(resp.read().decode())


def main() -> None:
    parser = argparse.ArgumentParser(description="Test TqSdk via gateway (zero local credentials)")
    parser.add_argument(
        "--mode",
        choices=["health", "live", "sim"],
        default="live",
        help="health=gateway only; live/sim=account+quote via gateway session",
    )
    args = parser.parse_args()

    try:
        health = _get("/health")
        logger.info("Gateway health: %s", health)
        if not health.get("connected"):
            raise RuntimeError("Gateway is up but TqSdk session is not connected")

        if args.mode == "health":
            logger.info("✓ Gateway health check passed")
            return

        account = _get("/api/v1/account")
        logger.info("Account: balance=%.2f available=%.2f", account["balance"], account["available"])

        quote = _get("/api/v1/market/quote/KQ.m@SHFE.rb")
        logger.info("Quote RB: last=%s", quote.get("last_price"))

        klines = _get("/api/v1/market/klines/KQ.m@SHFE.rb?duration=300&length=3")
        logger.info("Klines: %d bars", len(klines.get("items", [])))
        logger.info("✓ TqSdk gateway live check passed (mode=%s, account_mode=%s)", args.mode, health.get("account_mode"))
    except urllib.error.HTTPError as e:
        body = e.read().decode() if hasattr(e, "read") else ""
        logger.error("HTTP %s: %s", e.code, body)
        sys.exit(1)
    except Exception as e:
        logger.error("Test failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
