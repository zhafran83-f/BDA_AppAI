"""Explicit runtime database settings; no configuration is loaded at import."""

from typing import Literal

from pydantic import Field, SecretStr, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.exceptions import SettingsError


class DatabaseConfigurationError(ValueError):
    """Safe configuration failure without input values."""


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BDA_DB_",
        env_file=None,
        secrets_dir=None,
        extra="forbid",
        hide_input_in_errors=True,
        frozen=True,
    )

    enabled: bool = False
    host: Literal["127.0.0.1"] = "127.0.0.1"
    port: int = Field(default=5432, ge=1, le=65535)
    name: Literal["bda_ecommerce_ai"] = "bda_ecommerce_ai"
    user: Literal["bda_app_runtime"] = "bda_app_runtime"
    password: SecretStr = Field(
        default_factory=lambda: SecretStr(""), repr=False, exclude=True
    )

    @model_validator(mode="after")
    def validate_enabled_configuration(self):
        if self.enabled and not self.password.get_secret_value():
            raise ValueError("Password is required when database is enabled.")
        return self


def load_database_settings() -> DatabaseSettings:
    """Read process environment explicitly, without automatically reading .env."""
    try:
        return DatabaseSettings()
    except (ValidationError, SettingsError):
        raise DatabaseConfigurationError(
            "Invalid database configuration."
        ) from None
