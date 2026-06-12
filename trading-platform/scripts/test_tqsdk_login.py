#!/usr/bin/env python3
"""TqSdk 登录测试 — 验证 Simnow / 华安期货 连接是否正常。

从 PolarPrivate 读取凭证，支持两种模式：
  1. Simnow 模拟 (broker_id="simnow")
  2. 华安期货实盘 (broker_id="H华安期货")

Usage:
    python scripts/test_tqsdk_login.py --mode simnow   # Simnow 模拟
    python scripts/test_tqsdk_login.py --mode huaan     # 华安期货实盘
    python scripts/test_tqsdk_login.py --mode sim       # TqSim 本地模拟（无需账号）
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "security" / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# PolarPrivate 端口（SOTAgent 注册的端口）
PRIVPORTAL_URL = os.getenv("PRIVPORTAL_URL", "http://127.0.0.1:12790")


def _make_pp_client():
    """创建带会话管理的 PolarPrivate 客户端。"""
    import urllib.request
    import http.cookiejar
    cj = http.cookiejar.CookieJar()
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPCookieProcessor(cj),
    )


_opener = _make_pp_client()


def _pp_request(method: str, path: str, body: bytes | None = None) -> dict:
    """向 PolarPrivate API 发送请求。"""
    import urllib.request
    import json as _json
    url = f"{PRIVPORTAL_URL}{path}"
    headers = {"Content-Type": "application/json"} if body else {}
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    with _opener.open(req, timeout=10) as resp:
        return _json.loads(resp.read().decode())


def _grant_d_class(service_name: str) -> dict[str, str]:
    """Request a one-time plaintext grant via PolarPrivate D-class controlled channel.

    Requires this script's executable SHA256 to be in
    ~/.privportal/d-class-allowlist.json under `service_name` with
    `allowed_secret_keys` covering the keys we need.
    """
    import hashlib
    import json as _json

    sha = hashlib.sha256()
    with open(sys.executable, "rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            sha.update(chunk)
    payload = _json.dumps({
        "service_name": service_name,
        "caller_executable_sha256": sha.hexdigest(),
    }).encode()
    resp = _pp_request("POST", "/api/d-class/grant", payload)
    return resp.get("secrets", {}) if isinstance(resp, dict) else {}


def _require(secrets: dict[str, str], key: str, label: str) -> str:
    val = secrets.get(key)
    if not val or val in ("FILL_ME", "PENDING"):
        raise ValueError(
            f"{label}未配置或不在 D-class 白名单 allowed_secret_keys 中：{key}"
        )
    return val


def fetch_credentials(mode: str) -> dict[str, str]:
    """通过 D-class 受控信道一次性拿到 TqSdk 凭证。"""
    service_map = {
        "simnow": "tqsdk-login-simnow",
        "huaan": "tqsdk-login-huaan",
        "sim": "tqsdk-login",
    }
    service_name = service_map[mode]
    secrets = _grant_d_class(service_name)
    if not secrets:
        raise RuntimeError(
            f"D-class grant denied. Add this binary's SHA256 to "
            f"~/.privportal/d-class-allowlist.json under service_name "
            f"'{service_name}'."
        )

    auth_user = _require(secrets, "exchange.tqsdk.auth_user", "快期账号")
    auth_password = _require(secrets, "exchange.tqsdk.auth_password", "快期密码")
    result = {"auth_user": auth_user, "auth_password": auth_password}

    if mode == "simnow":
        result["broker_id"] = "simnow"
        result["account_id"] = _require(secrets, "exchange.tqsdk.simnow_user", "Simnow 账号")
        result["password"] = _require(secrets, "exchange.tqsdk.simnow_password", "Simnow 密码")
    elif mode == "huaan":
        result["broker_id"] = _require(secrets, "exchange.tqsdk.broker", "期货公司")
        result["account_id"] = _require(secrets, "exchange.tqsdk.account", "华安期货账号")
        result["password"] = _require(secrets, "exchange.tqsdk.password", "华安期货密码")

    return result


def test_login(mode: str) -> None:
    """执行 TqSdk 登录测试。"""
    from tqsdk import TqApi, TqAuth

    if mode == "sim":
        # TqSim 本地模拟，仍需快期认证
        from tqsdk import TqSim
        creds = fetch_credentials("sim")
        logger.info("=" * 60)
        logger.info("模式: TqSim 本地模拟")
        logger.info("快期账号: %s", creds["auth_user"])
        logger.info("=" * 60)
        auth = TqAuth(creds["auth_user"], creds["auth_password"])
        api = TqApi(account=TqSim(init_balance=1_000_000), auth=auth)
    else:
        creds = fetch_credentials(mode)

        from tqsdk import TqAccount
        logger.info("=" * 60)
        logger.info("模式: %s", "Simnow 模拟" if mode == "simnow" else "华安期货实盘")
        logger.info("快期账号: %s", creds["auth_user"])
        logger.info("Broker: %s", creds["broker_id"])
        logger.info("交易账号: %s", creds["account_id"])
        logger.info("=" * 60)

        auth = TqAuth(creds["auth_user"], creds["auth_password"])
        account = TqAccount(
            broker_id=creds["broker_id"],
            account_id=creds["account_id"],
            password=creds["password"],
        )
        api = TqApi(account=account, auth=auth)

    try:
        logger.info("✓ TqApi 创建成功，连接已建立！")

        # 获取账户信息
        acc = api.get_account()
        logger.info("── 账户信息 ──")
        logger.info("  可用资金: %.2f", acc.available)
        logger.info("  账户权益: %.2f", acc.balance)
        logger.info("  保证金:   %.2f", acc.margin)
        logger.info("  浮动盈亏: %.2f", acc.float_profit)

        # 测试行情 — 获取螺纹钢主力合约行情
        quote = api.get_quote("KQ.m@SHFE.rb")
        logger.info("── 行情测试 (螺纹钢主力) ──")
        logger.info("  合约: KQ.m@SHFE.rb")
        logger.info("  最新价: %s", quote.last_price)
        logger.info("  买一价: %s / 量: %s", quote.bid_price1, quote.bid_volume1)
        logger.info("  卖一价: %s / 量: %s", quote.ask_price1, quote.ask_volume1)

        # 获取5分钟K线
        klines = api.get_kline_serial("KQ.m@SHFE.rb", 300, 5)
        logger.info("── K线测试 (5分钟, 最近5根) ──")
        for i in range(len(klines)):
            row = klines.iloc[i]
            logger.info(
                "  %s  O=%.0f H=%.0f L=%.0f C=%.0f V=%d",
                row.name, row["open"], row["high"], row["low"], row["close"], int(row["volume"]),
            )

        logger.info("=" * 60)
        logger.info("✓ 全部测试通过！TqSdk 登录和行情获取正常。")
        logger.info("=" * 60)

    finally:
        api.close()
        logger.info("TqApi 已关闭。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TqSdk 登录测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
PolarPrivate 字段说明:
  exchange.tqsdk.auth_user      快期账号（邮箱/手机号）
  exchange.tqsdk.auth_password  快期密码
  exchange.tqsdk.broker         期货公司（已设为 "H华安期货"）
  exchange.tqsdk.account        华安期货资金账号
  exchange.tqsdk.password       华安期货交易密码
  exchange.tqsdk.simnow_user    Simnow 投资者账号
  exchange.tqsdk.simnow_password Simnow 密码
""",
    )
    parser.add_argument(
        "--mode",
        choices=["simnow", "huaan", "sim"],
        default="simnow",
        help="登录模式: simnow=Simnow模拟, huaan=华安期货实盘, sim=TqSim本地模拟",
    )
    args = parser.parse_args()

    try:
        test_login(args.mode)
    except Exception as e:
        logger.error("登录失败: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
