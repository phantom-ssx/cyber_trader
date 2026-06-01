from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── OKX ──────────────────────────────────────────────────────────────────
    okx_api_key: str = ""
    okx_api_secret: str = ""
    okx_passphrase: str = ""
    okx_is_demo: bool = False

    # ── Feishu ────────────────────────────────────────────────────────────────
    feishu_webhook_url: str = ""
    feishu_secret: str = ""

    # ── Data ─────────────────────────────────────────────────────────────────
    data_catalog_path: Path = Path("./data/catalog")
    database_url: str = "postgresql+asyncpg://trader:password@localhost:5432/cyber_trader"

    # ── Runtime ──────────────────────────────────────────────────────────────
    trading_mode: Literal["backtest", "paper", "live"] = "paper"
    trader_id: str = "CYBER-TRADER-001"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("data_catalog_path", mode="before")
    @classmethod
    def expand_path(cls, v: str | Path) -> Path:
        return Path(v).expanduser().resolve()

    def ensure_catalog_dir(self) -> None:
        self.data_catalog_path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
