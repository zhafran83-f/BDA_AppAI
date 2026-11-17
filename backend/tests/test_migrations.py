"""Complete offline SQL and simulated transaction tests: no PostgreSQL access."""

from contextlib import contextmanager
from importlib.util import module_from_spec, spec_from_file_location
from io import StringIO
import os
from pathlib import Path
import re
import traceback
from types import SimpleNamespace
from unittest.mock import Mock, patch

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from mako.template import Template
import pytest
from sqlalchemy.dialects.postgresql import dialect
from sqlalchemy.schema import CreateTable

from backend.app.core import migration_config
from backend.app.core.migration_config import (
    MigrationConfigurationError, MigrationSettings,
    build_migration_url, load_migration_settings,
)
from backend.app.db.base import Base
from backend.app.db import models  # Register exactly the application tables.

ROOT = Path(__file__).resolve().parents[2]
INI = ROOT / "backend/alembic.ini"
REVISION = ROOT / "backend/alembic/versions/0001_prediction_history.py"
SECRET = "dummy-only@:/?#-never-real"
TABLES = ("fashion_predictions", "review_clusters", "purchase_predictions")


@pytest.fixture(autouse=True)
def isolated_environment_and_no_connections(monkeypatch):
    import psycopg
    forbidden = Mock(side_effect=AssertionError("PostgreSQL access forbidden."))
    engine = Mock(side_effect=AssertionError("Real engine creation forbidden."))
    monkeypatch.setattr(psycopg, "connect", forbidden)
    monkeypatch.setattr("sqlalchemy.create_engine", engine)
    with patch.dict(os.environ, {}, clear=True):
        yield
    forbidden.assert_not_called()


def config_for(output=None):
    return Config(str(INI), output_buffer=output)


def render_sql():
    output = StringIO()
    command.upgrade(config_for(output), "head", sql=True)
    return output.getvalue()


@pytest.fixture
def revision_module():
    spec = spec_from_file_location("bda_frozen_initial_revision", REVISION)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize(sql):
    return re.sub(r"\s+", " ", sql.strip()).rstrip(";")


def test_entire_offline_sql_without_settings_or_engine(monkeypatch):
    reader = Mock(side_effect=AssertionError("No settings/password in offline mode."))
    monkeypatch.setattr(migration_config, "load_migration_settings", reader)
    sql = render_sql()
    reader.assert_not_called()
    assert sql.lstrip().startswith("BEGIN;")
    assert sql.rstrip().endswith("COMMIT;")
    guard = sql.index("DO $bda_target$")
    version = sql.index("CREATE TABLE public.alembic_version")
    first_table = sql.index("CREATE TABLE public.fashion_predictions")
    stamp = sql.index("INSERT INTO public.alembic_version")
    assert guard < version < first_table < stamp < sql.rindex("COMMIT;")
    assert sql.count("CREATE TABLE public.") == 4
    assert sql.count("CREATE INDEX ix_") == 3
    assert "'0001_prediction_history'" in sql
    assert "BDA_MIGRATION_DB_PASSWORD" not in sql
    assert "postgresql+psycopg://" not in sql
    assert "CREATE DATABASE" not in sql and "CREATE ROLE" not in sql


def test_offline_ignores_even_invalid_dummy_credentials():
    with patch.dict(os.environ, {
        "BDA_MIGRATION_APPROVED": "not-approved",
        "BDA_MIGRATION_DB_USER": "postgres",
        "BDA_MIGRATION_DB_PASSWORD": SECRET,
        "BDA_MIGRATION_DB_PORT": "invalid-" + SECRET,
    }, clear=True):
        sql = render_sql()
    assert SECRET not in sql
    assert "CREATE TABLE public.purchase_predictions" in sql


def test_frozen_migration_matches_every_orm_table(revision_module):
    assert revision_module.revision == "0001_prediction_history"
    assert revision_module.down_revision is None
    expected = {
        normalize(str(CreateTable(table).compile(dialect=dialect())))
        for table in Base.metadata.tables.values()
    }
    assert {normalize(sql) for sql in revision_module.TABLE_DDL} == expected
    assert set(revision_module.TABLES) == set(TABLES)


def test_exact_runtime_privileges_and_no_version_access():
    sql = render_sql()
    grants = re.findall(r"GRANT\s+(.+?)\s+ON TABLE\s+(\S+)\s+TO\s+(\S+);", sql)
    assert set(grants) == {
        ("SELECT, INSERT", "public." + table, "bda_app_runtime")
        for table in TABLES
    }
    assert len(grants) == 3
    for table in (*TABLES, "alembic_version"):
        assert (
            f"REVOKE ALL PRIVILEGES ON TABLE public.{table} "
            "FROM PUBLIC, bda_app_runtime;"
        ) in sql
    assert "ON ALL TABLES" not in sql
    assert "ALTER DEFAULT PRIVILEGES" not in sql


@pytest.mark.parametrize("approval", [None, "", "true", "0", "yes"])
def test_approval_checked_before_settings(monkeypatch, approval):
    if approval is not None:
        monkeypatch.setenv("BDA_MIGRATION_APPROVED", approval)
    reader = Mock(side_effect=AssertionError("Must not read settings."))
    monkeypatch.setattr(migration_config, "MigrationSettings", reader)
    with pytest.raises(MigrationConfigurationError, match="explicit process approval"):
        load_migration_settings()
    reader.assert_not_called()


def test_dummy_migrator_settings_and_password_safe_url():
    dummy = {
        "BDA_MIGRATION_APPROVED": "1",
        "BDA_MIGRATION_DB_HOST": "127.0.0.1",
        "BDA_MIGRATION_DB_PORT": "5432",
        "BDA_MIGRATION_DB_NAME": "bda_ecommerce_ai",
        "BDA_MIGRATION_DB_USER": "bda_app_migrator",
        "BDA_MIGRATION_DB_PASSWORD": SECRET,
        "BDA_DB_USER": "postgres",
        "BDA_DB_PASSWORD": "ignored-runtime-dummy",
    }
    with patch.dict(os.environ, dummy, clear=True):
        settings = load_migration_settings()
    assert settings.user == "bda_app_migrator"
    assert settings.password.get_secret_value() == SECRET
    assert SECRET not in repr(settings)
    assert "password" not in settings.model_dump()
    assert SECRET not in settings.model_dump_json()
    url = build_migration_url(settings)
    assert url.drivername == "postgresql+psycopg"
    assert url.username == "bda_app_migrator"
    assert url.password == SECRET
    assert url.database == "bda_ecommerce_ai"
    assert SECRET not in str(url)


@pytest.mark.parametrize("key,value", [
    ("USER", "postgres"), ("USER", "bda_app_runtime"),
    ("NAME", "other_database"), ("HOST", "::1"), ("PORT", "5433"),
    ("PASSWORD", ""),
])
def test_invalid_migration_configuration_is_sanitized(key, value):
    dummy = {
        "BDA_MIGRATION_APPROVED": "1",
        "BDA_MIGRATION_DB_PASSWORD": SECRET,
        "BDA_MIGRATION_DB_" + key: value,
    }
    with patch.dict(os.environ, dummy, clear=True):
        with pytest.raises(MigrationConfigurationError) as caught:
            load_migration_settings()
    output = "".join(traceback.format_exception(caught.value))
    assert SECRET not in output
    assert "Invalid migration configuration." in output


def test_dotenv_not_read(monkeypatch):
    from pydantic_settings.sources.providers.dotenv import DotEnvSettingsSource
    reader = Mock(side_effect=AssertionError("No .env reads."))
    monkeypatch.setattr(DotEnvSettingsSource, "_read_env_file", reader)
    with patch.dict(os.environ, {
        "BDA_MIGRATION_APPROVED": "1", "BDA_MIGRATION_DB_PASSWORD": SECRET,
    }, clear=True):
        load_migration_settings()
    reader.assert_not_called()


def test_ini_has_no_url_or_credentials():
    config = config_for()
    assert not config.get_main_option("sqlalchemy.url")
    text = INI.read_text(encoding="utf-8")
    assert "password" not in text.lower()
    assert "postgresql://" not in text


def test_revision_template_compiles_without_files():
    source = Template(filename=str(ROOT / "backend/alembic/script.py.mako")).render(
        message="Unit test revision", imports="", up_revision="dummy",
        down_revision=None, branch_labels=None, depends_on=None,
        upgrades="", downgrades="",
    )
    compile(source, "<mako-test>", "exec")


class FakeTransaction:
    def __init__(self, connection):
        self.connection = connection

    def commit(self):
        self.connection.events.append("COMMIT")
        self.connection.active = False

    def rollback(self):
        self.connection.events.append("ROLLBACK")
        self.connection.active = False

    def __exit__(self, kind, value, traceback):
        self.rollback() if kind is not None else self.commit()


class FakeConnection:
    """Implements only the SQLAlchemy connection protocol Alembic needs."""

    def __init__(self, events, fail_guard=False, fail_stamp=False, active=False):
        self.events = events
        self.dialect = dialect()
        self.active = active
        self.transaction = None
        self.fail_guard = fail_guard
        self.fail_stamp = fail_stamp

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if self.active:
            self.events.append("CLOSE_WITH_UNCOMMITTED_TRANSACTION")
            self.active = False
        self.events.append("CLOSE")

    def in_transaction(self):
        return self.active

    def get_transaction(self):
        return self.transaction if self.active else None

    def begin(self):
        assert not self.active, "Unexpected autobegin/external transaction."
        self.events.append("BEGIN")
        self.active = True
        self.transaction = FakeTransaction(self)
        return self.transaction

    def execute(self, statement, parameters=None):
        assert self.active, "SQL must run inside Alembic-owned transaction."
        sql = str(statement.compile(
            dialect=self.dialect, compile_kwargs={"literal_binds": True}
        ))
        self.events.append(sql)
        if self.fail_guard and "DO $bda_target$" in sql:
            raise RuntimeError(SECRET)
        if self.fail_stamp and "INSERT INTO public.alembic_version" in sql:
            raise RuntimeError(SECRET)
        return SimpleNamespace(rowcount=1)


@pytest.fixture
def online_simulator(monkeypatch):
    """Real Alembic transaction manager; fake engine/connection, never PostgreSQL."""

    def run(*, fail_guard=False, fail_stamp=False, fail_dispose=False, active=False):
        events = []
        connection = FakeConnection(events, fail_guard, fail_stamp, active)
        engine = Mock()
        engine.connect.return_value = connection

        def dispose():
            events.append("DISPOSE")
            if fail_dispose:
                raise RuntimeError(SECRET)
        engine.dispose.side_effect = dispose
        creator = Mock(return_value=engine)
        monkeypatch.setattr("sqlalchemy.create_engine", creator)
        monkeypatch.setattr(MigrationContext, "get_current_heads", lambda self: ())

        def ensure_version(context, purge=False):
            assert context.connection.active
            context.connection.execute(
                CreateTable(context._version)
            )
        monkeypatch.setattr(MigrationContext, "_ensure_version_table", ensure_version)

        captured = []
        original_configure = MigrationContext.configure.__func__

        def configure(cls, *args, **kwargs):
            connection = kwargs.get("connection")
            if connection is not None:
                assert not connection.in_transaction()
                events.append("CONFIGURE_WITHOUT_TRANSACTION")
            context = original_configure(cls, *args, **kwargs)
            captured.append(context)
            return context
        monkeypatch.setattr(MigrationContext, "configure", classmethod(configure))

        error = None
        with patch.dict(os.environ, {
            "BDA_MIGRATION_APPROVED": "1",
            "BDA_MIGRATION_DB_PASSWORD": SECRET,
        }, clear=True):
            try:
                # sql=False is simulated ONLY: all DB calls go to FakeConnection.
                command.upgrade(config_for(), "head", sql=False)
            except Exception as exc:
                error = exc
        return events, error, creator, captured

    return run


def test_simulated_online_commits_schema_and_revision_together(online_simulator):
    events, error, creator, contexts = online_simulator()
    assert error is None
    assert events.count("BEGIN") == events.count("COMMIT") == 1
    assert "ROLLBACK" not in events
    assert "CLOSE_WITH_UNCOMMITTED_TRANSACTION" not in events
    guard = next(i for i, event in enumerate(events) if "DO $bda_target$" in event)
    version = next(i for i, event in enumerate(events)
                   if "CREATE TABLE public.alembic_version" in event)
    stamp = next(i for i, event in enumerate(events)
                 if "INSERT INTO public.alembic_version" in event)
    assert events.index("CONFIGURE_WITHOUT_TRANSACTION") < events.index("BEGIN")
    assert events.index("BEGIN") < guard < version < stamp < events.index("COMMIT")
    assert events.index("COMMIT") < events.index("CLOSE") < events.index("DISPOSE")
    assert contexts[0]._in_external_transaction is False
    options = creator.call_args.kwargs
    assert options["hide_parameters"] is True
    assert options["echo"] is False
    assert options["connect_args"]["connect_timeout"] == 3


@pytest.mark.parametrize("failure", ["guard", "stamp"])
def test_simulated_failure_rolls_back_without_commit(online_simulator, failure):
    events, error, _, _ = online_simulator(
        fail_guard=failure == "guard", fail_stamp=failure == "stamp",
    )
    assert error is not None
    assert "COMMIT" not in events
    assert events.count("ROLLBACK") == 1
    assert events[-2:] == ["CLOSE", "DISPOSE"]
    assert SECRET not in "".join(traceback.format_exception(error))
    if failure == "guard":
        assert not any("CREATE TABLE" in event for event in events)


def test_active_transaction_rejected_before_ddl(online_simulator):
    events, error, _, _ = online_simulator(active=True)
    assert error is not None
    assert not any("CREATE TABLE" in event for event in events)
    assert "BEGIN" not in events
    assert SECRET not in "".join(traceback.format_exception(error))


def test_dispose_does_not_mask_primary_migration_failure(online_simulator):
    events, error, _, _ = online_simulator(fail_guard=True, fail_dispose=True)
    assert str(error) == "Migration execution failed."
    assert "migration_engine_dispose" in " ".join(error.__notes__)
    assert SECRET not in "".join(traceback.format_exception(error))
    assert "ROLLBACK" in events


def test_online_without_approval_does_not_create_engine(monkeypatch):
    creator = Mock(side_effect=AssertionError("Must not create engine."))
    monkeypatch.setattr("sqlalchemy.create_engine", creator)
    with pytest.raises(RuntimeError) as caught:
        command.upgrade(config_for(), "head", sql=False)
    creator.assert_not_called()
    assert str(caught.value) == "Migration execution failed."
