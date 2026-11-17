"""Offline SQL rendering or explicitly approved migration execution.

No runtime configuration, .env, model artifact, or credential URL is logged.
The target guard is executed INSIDE Alembic's transaction, BEFORE run_migrations,
which is also responsible for creating/stamping public.alembic_version.
"""

from alembic import context
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

from backend.app.core.migration_config import (
    build_migration_url,
    load_migration_settings,
)
from backend.app.db import record_cleanup_failure
from backend.app.db.base import Base
from backend.app.db.models import FashionPrediction, PurchasePrediction, ReviewCluster

target_metadata = Base.metadata
APPLICATION_TABLES = frozenset({
    "fashion_predictions", "review_clusters", "purchase_predictions",
})

TARGET_GUARD = r"""-- BDA target guard: must run before any migration DDL.
DO $bda_target$
BEGIN
    IF current_database() <> 'bda_ecommerce_ai'
       OR current_user <> 'bda_app_migrator'
       OR session_user <> 'bda_app_migrator'
       OR inet_server_addr() IS DISTINCT FROM '127.0.0.1'::inet
       OR inet_server_port() IS DISTINCT FROM 5432
       OR current_setting('server_version_num')::integer / 10000 <> 15
       OR (
           SELECT pg_get_userbyid(datdba)
           FROM pg_database WHERE datname = current_database()
       ) IS DISTINCT FROM 'postgres'
    THEN
        RAISE EXCEPTION 'Migration target verification failed';
    END IF;

    IF (
        SELECT count(*) FROM pg_roles
        WHERE rolname IN ('bda_app_runtime', 'bda_app_migrator')
    ) <> 2 OR EXISTS (
        SELECT 1 FROM pg_roles
        WHERE rolname IN ('bda_app_runtime', 'bda_app_migrator')
          AND (NOT rolcanlogin OR rolsuper OR rolcreatedb OR rolcreaterole
               OR rolreplication OR rolbypassrls)
    ) OR EXISTS (
        SELECT 1 FROM pg_auth_members m
        JOIN pg_roles r ON r.oid = m.member
        WHERE r.rolname IN ('bda_app_runtime', 'bda_app_migrator')
    ) THEN
        RAISE EXCEPTION 'Migration role verification failed';
    END IF;

    IF NOT has_schema_privilege('bda_app_migrator', 'public', 'USAGE')
       OR NOT has_schema_privilege('bda_app_migrator', 'public', 'CREATE')
       OR NOT has_schema_privilege('bda_app_runtime', 'public', 'USAGE')
       OR has_schema_privilege('bda_app_runtime', 'public', 'CREATE')
    THEN
        RAISE EXCEPTION 'Migration schema privilege verification failed';
    END IF;

    IF EXISTS (
        SELECT 1 FROM pg_roles r
        WHERE r.rolname IN ('bda_app_runtime', 'bda_app_migrator')
          AND (
              NOT has_database_privilege(r.oid, current_database(), 'CONNECT')
              OR has_database_privilege(r.oid, current_database(), 'TEMP')
              OR has_database_privilege(r.oid, current_database(), 'CREATE')
          )
    ) THEN
        RAISE EXCEPTION 'Migration database privilege verification failed';
    END IF;
END
$bda_target$
"""


class MigrationExecutionError(RuntimeError):
    """Sanitized migration failure without SQL, parameters or driver text."""


def include_name(name, type_, parent_names):
    if type_ == "schema":
        return name in (None, "public")
    if type_ == "table":
        return (
            name in APPLICATION_TABLES
            and parent_names.get("schema_name") in (None, "public")
        )
    return True


def configuration():
    return {
        "target_metadata": target_metadata,
        "version_table": "alembic_version",
        "version_table_schema": "public",
        "include_schemas": True,
        "include_name": include_name,
        "compare_type": True,
        "compare_server_default": True,
        "transactional_ddl": True,
        "transaction_per_migration": False,
    }


def run_migrations_offline():
    # Deliberately do not load settings, check approval, or create an engine.
    context.configure(
        dialect_name="postgresql",
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        **configuration(),
    )
    with context.begin_transaction():
        context.execute(text(TARGET_GUARD))
        context.run_migrations()


def run_migrations_online():
    engine = None
    primary = None
    try:
        settings = load_migration_settings()
        engine = create_engine(
            build_migration_url(settings),
            poolclass=NullPool,
            echo=False,
            echo_pool=False,
            hide_parameters=True,
            connect_args={
                "connect_timeout": 3,
                "application_name": "bda_appai_migrator",
                "options": "-c statement_timeout=30000 -c lock_timeout=3000",
            },
        )
        with engine.connect() as connection:
            # No preflight SELECT or guard may cause autobegin before configure.
            if connection.in_transaction():
                raise MigrationExecutionError(
                    "Migration requires a connection without an active transaction."
                )
            context.configure(connection=connection, **configuration())
            with context.begin_transaction():
                context.execute(text(TARGET_GUARD))
                context.run_migrations()
    except Exception:
        primary = MigrationExecutionError("Migration execution failed.")
        raise primary from None
    except BaseException as exc:
        primary = exc
        raise
    finally:
        if engine is not None:
            try:
                engine.dispose()
            except Exception:
                if primary is not None:
                    record_cleanup_failure(primary, "migration_engine_dispose")
                else:
                    raise MigrationExecutionError(
                        "Migration engine cleanup failed."
                    ) from None


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
