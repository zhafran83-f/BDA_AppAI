"""Lazy engine factory and explicit lifecycle; no connection probes or DDL."""

from contextlib import contextmanager

from sqlalchemy import URL, create_engine
from sqlalchemy.engine import Engine

from ..core.config import DatabaseSettings
from . import DatabaseCleanupError, DatabaseOperationError, record_cleanup_failure


def create_database_engine(settings: DatabaseSettings) -> Engine | None:
    if not settings.enabled:
        return None
    try:
        url = URL.create(
            drivername="postgresql+psycopg",
            username=settings.user,
            password=settings.password.get_secret_value(),
            host=settings.host,
            port=settings.port,
            database=settings.name,
        )
        return create_engine(
            url,
            pool_size=2,
            max_overflow=0,
            pool_timeout=3,
            pool_pre_ping=True,
            echo=False,
            echo_pool=False,
            hide_parameters=True,
            connect_args={
                "connect_timeout": 3,
                "application_name": "bda_appai_runtime",
                "options": "-c statement_timeout=5000",
            },
        )
    except Exception:
        raise DatabaseOperationError("engine_initialization") from None


@contextmanager
def database_engine(settings: DatabaseSettings):
    """Own one engine per future application lifespan, not per request."""
    engine = create_database_engine(settings)
    primary = None
    try:
        yield engine
    except BaseException as exc:
        primary = exc
        raise
    finally:
        if engine is not None:
            try:
                engine.dispose()
            except Exception:
                if primary is not None:
                    record_cleanup_failure(primary, "engine_dispose")
                else:
                    raise DatabaseCleanupError("engine_dispose") from None
