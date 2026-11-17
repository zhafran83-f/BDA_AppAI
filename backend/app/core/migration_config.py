"""Explicit migration-only configuration, separate from runtime settings.

Online migration requires BDA_MIGRATION_APPROVED=1 in that process only.
Offline SQL rendering never instantiates these settings or reads a password.
"""

import os
from typing import Literal

from pydantic import Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.exceptions import SettingsError
from sqlalchemy import URL


class MigrationConfigurationError(ValueError):
    """Sanitized approval/configuration failure."""


class MigrationSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BDA_MIGRATION_DB_",
        env_file=None,
        secrets_dir=None,
        extra="forbid",
        hide_input_in_errors=True,
        frozen=True,
    )

    host: Literal["127.0.0.1"] = "127.0.0.1"
    port: int = Field(default=5432, ge=5432, le=5432)
    name: Literal["bda_ecommerce_ai"] = "bda_ecommerce_ai"
    user: Literal["bda_app_migrator"] = "bda_app_migrator"
    password: SecretStr = Field(repr=False, exclude=True, min_length=1)


def load_migration_settings() -> MigrationSettings:
    # Check approval BEFORE initializing any environment settings source.
    if os.environ.get("BDA_MIGRATION_APPROVED") != "1":
        raise MigrationConfigurationError(
            "Online migration requires explicit process approval."
        )
    try:
        return MigrationSettings()
    except (ValidationError, SettingsError):
        raise MigrationConfigurationError(
            "Invalid migration configuration."
        ) from None


def build_migration_url(settings: MigrationSettings) -> URL:
    return URL.create(
        "postgresql+psycopg",
        username=settings.user,
        password=settings.password.get_secret_value(),
        host=settings.host,
        port=settings.port,
        database=settings.name,
    )
