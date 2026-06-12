"""应用全局配置."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    """从环境变量加载应用配置."""

    app_name: str = "TqTrader API"
    debug: bool = False
    database_url: str = "postgresql+asyncpg://localhost:5432/tqtrader"
    redis_url: str = "redis://localhost:6379/0"
    log_level: str = "INFO"

    model_config = {"env_prefix": "APP_", "case_sensitive": False}


settings = AppSettings()
