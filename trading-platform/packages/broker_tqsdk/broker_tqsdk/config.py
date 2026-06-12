"""TqSdk 连接配置."""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class TqSdkSettings(BaseSettings):
    """从环境变量加载 TqSdk 配置."""

    tq_auth_email: str = ""
    tq_auth_password: str = ""
    tq_broker_id: str = ""
    tq_account_id: str = ""
    tq_td_url: str = ""
    tq_use_sim: bool = True

    model_config = {"env_prefix": "TQ_", "case_sensitive": False}


class BrokerConfig(BaseModel):
    """Broker 运行时配置."""

    reconnect_interval: float = Field(5.0, ge=1.0)
    max_reconnect_attempts: int = Field(10, ge=1)
    heartbeat_interval: float = Field(30.0, ge=5.0)
