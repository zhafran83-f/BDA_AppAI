"""Database foundation tests: dummy environment and fakes, no PostgreSQL."""

import importlib
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError
from sqlalchemy.exc import DBAPIError, SQLAlchemyError

from backend.app.core import config
from backend.app.core.config import DatabaseSettings
from backend.app.db import (
    DatabaseCleanupError,
    DatabaseCommitError,
    DatabaseOperationError,
    TransactionOutcome,
    TransactionReport,
)
from backend.app.db import engine as engine_module
from backend.app.db.session import create_session_factory, session_scope


FAKE_PASSWORD = "unit-test-only@:/?#"
SENSITIVE_ERROR = "unit-test-driver-secret"


class IsolatedSettings(DatabaseSettings):
    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings,
        dotenv_settings, file_secret_settings,
    ):
        return (init_settings,)


@pytest.fixture(autouse=True)
def no_database_connections(monkeypatch):
    import psycopg

    connect = Mock(side_effect=AssertionError("Real connections forbidden."))
    monkeypatch.setattr(psycopg, "connect", connect)
    yield
    connect.assert_not_called()


def enabled_settings():
    return IsolatedSettings(enabled=True, password=FAKE_PASSWORD)


def formatted(exc):
    return "".join(traceback.format_exception(exc))


def driver_error(sqlstate=None):
    original = Exception(SENSITIVE_ERROR)
    original.sqlstate = sqlstate
    return DBAPIError("unit-test SQL", {}, original, False)


def test_disabled_by_default():
    settings = IsolatedSettings()
    assert settings.enabled is False
    assert settings.password.get_secret_value() == ""
    assert settings.name == "bda_ecommerce_ai"
    assert settings.user == "bda_app_runtime"


def test_process_environment_disabled_by_default():
    with patch.dict(os.environ, {}, clear=True):
        settings = config.load_database_settings()
    assert settings.enabled is False
    assert settings.password.get_secret_value() == ""


def test_reads_all_six_dummy_environment_variables():
    dummy = {
        "BDA_DB_ENABLED": "true",
        "BDA_DB_HOST": "127.0.0.1",
        "BDA_DB_PORT": "5433",
        "BDA_DB_NAME": "bda_ecommerce_ai",
        "BDA_DB_USER": "bda_app_runtime",
        "BDA_DB_PASSWORD": FAKE_PASSWORD,
    }
    with patch.dict(os.environ, dummy, clear=True):
        settings = config.load_database_settings()
    assert settings.enabled is True
    assert settings.host == "127.0.0.1"
    assert settings.port == 5433
    assert settings.name == "bda_ecommerce_ai"
    assert settings.user == "bda_app_runtime"
    assert settings.password.get_secret_value() == FAKE_PASSWORD


def test_does_not_read_dotenv(monkeypatch):
    from pydantic_settings.sources.providers.dotenv import DotEnvSettingsSource

    reader = Mock(side_effect=AssertionError("Dotenv must not be read."))
    monkeypatch.setattr(DotEnvSettingsSource, "_read_env_file", reader)
    with patch.dict(os.environ, {}, clear=True):
        assert config.load_database_settings().enabled is False
    reader.assert_not_called()


def test_password_not_in_repr_or_dump():
    settings = enabled_settings()
    assert FAKE_PASSWORD not in repr(settings)
    assert "password" not in settings.model_dump()
    assert FAKE_PASSWORD not in settings.model_dump_json()


@pytest.mark.parametrize("values", [
    {"enabled": True},
    {"port": 0},
    {"port": 65536},
    {"host": "other-host"},
    {"name": "bda_appai"},
    {"user": "postgres"},
    {"user": "bda_app_migrator"},
    {"unexpected": True},
])
def test_invalid_configuration(values):
    with pytest.raises(ValidationError):
        IsolatedSettings(**values)


def test_environment_validation_error_is_sanitized():
    with patch.dict(os.environ, {
        "BDA_DB_ENABLED": "invalid-" + SENSITIVE_ERROR,
        "BDA_DB_PASSWORD": FAKE_PASSWORD,
    }, clear=True):
        with pytest.raises(config.DatabaseConfigurationError) as caught:
            config.load_database_settings()
    assert FAKE_PASSWORD not in formatted(caught.value)
    assert SENSITIVE_ERROR not in formatted(caught.value)


def test_import_does_not_load_settings(monkeypatch):
    import pydantic_settings

    init = Mock(side_effect=AssertionError("No settings at import."))
    with monkeypatch.context() as context:
        context.setattr(pydantic_settings.BaseSettings, "__init__", init)
        importlib.reload(config)
    init.assert_not_called()


def test_import_does_not_create_engine(monkeypatch):
    creator = Mock()
    with monkeypatch.context() as context:
        context.setattr("sqlalchemy.create_engine", creator)
        importlib.reload(engine_module)
    creator.assert_not_called()
    importlib.reload(engine_module)


def test_disabled_does_not_create_engine(monkeypatch):
    creator = Mock()
    monkeypatch.setattr(engine_module, "create_engine", creator)
    with engine_module.database_engine(IsolatedSettings()) as engine:
        assert engine is None
    creator.assert_not_called()


def test_engine_configuration_and_lazy_lifecycle(monkeypatch):
    engine = Mock()
    creator = Mock(return_value=engine)
    monkeypatch.setattr(engine_module, "create_engine", creator)
    with engine_module.database_engine(enabled_settings()) as current:
        assert current is engine
        engine.connect.assert_not_called()
    engine.dispose.assert_called_once_with()
    url = creator.call_args.args[0]
    options = creator.call_args.kwargs
    assert url.drivername == "postgresql+psycopg"
    assert url.host == "127.0.0.1"
    assert url.database == "bda_ecommerce_ai"
    assert url.username == "bda_app_runtime"
    assert url.password == FAKE_PASSWORD
    assert options["pool_size"] == 2
    assert options["max_overflow"] == 0
    assert options["pool_timeout"] == 3
    assert options["pool_pre_ping"] is True
    assert options["hide_parameters"] is True
    assert options["echo"] is False
    assert options["echo_pool"] is False
    assert options["connect_args"]["connect_timeout"] == 3
    assert "statement_timeout=5000" in options["connect_args"]["options"]


def test_engine_initialization_error_is_sanitized(monkeypatch, capsys, caplog):
    monkeypatch.setattr(
        engine_module, "create_engine",
        Mock(side_effect=RuntimeError(SENSITIVE_ERROR)),
    )
    with pytest.raises(DatabaseOperationError) as caught:
        engine_module.create_database_engine(enabled_settings())
    assert caught.value.stage == "engine_initialization"
    assert SENSITIVE_ERROR not in formatted(caught.value)
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    assert caplog.text == ""


@pytest.mark.parametrize("primary", [
    ValueError("Application failure"),
    DatabaseCommitError("commit", TransactionOutcome.UNKNOWN),
    KeyboardInterrupt(),
])
def test_engine_cleanup_preserves_primary(monkeypatch, primary):
    engine = Mock()
    engine.dispose.side_effect = RuntimeError(SENSITIVE_ERROR)
    monkeypatch.setattr(engine_module, "create_engine", Mock(return_value=engine))
    with pytest.raises(type(primary)) as caught:
        with engine_module.database_engine(enabled_settings()):
            raise primary
    assert caught.value is primary
    assert "engine_dispose" in " ".join(primary.__notes__)
    assert SENSITIVE_ERROR not in formatted(caught.value)
    engine.dispose.assert_called_once_with()


def test_engine_cleanup_failure_without_primary(monkeypatch):
    engine = Mock()
    engine.dispose.side_effect = RuntimeError(SENSITIVE_ERROR)
    monkeypatch.setattr(engine_module, "create_engine", Mock(return_value=engine))
    with pytest.raises(DatabaseCleanupError) as caught:
        with engine_module.database_engine(enabled_settings()):
            pass
    assert caught.value.stage == "engine_dispose"
    assert caught.value.outcome is None
    assert SENSITIVE_ERROR not in formatted(caught.value)


def test_session_factory_is_lazy_and_sessions_are_distinct():
    factory = create_session_factory(Mock())
    first, second = factory(), factory()
    try:
        assert first is not second
        assert first.autoflush is False
        assert first.expire_on_commit is False
    finally:
        first.close()
        second.close()


def test_session_commits_and_closes():
    session = Mock()
    report = TransactionReport()
    with session_scope(Mock(return_value=session), report=report) as current:
        assert current is session
    assert report.outcome is TransactionOutcome.COMMITTED
    assert report.failed_stage is None
    session.commit.assert_called_once_with()
    session.rollback.assert_not_called()
    session.close.assert_called_once_with()


def test_operation_failure_rolls_back_and_preserves_identity():
    session = Mock()
    primary = ValueError("Application failure")
    report = TransactionReport()
    with pytest.raises(ValueError) as caught:
        with session_scope(Mock(return_value=session), report=report):
            raise primary
    assert caught.value is primary
    assert report.outcome is TransactionOutcome.NOT_COMMITTED
    session.commit.assert_not_called()
    session.rollback.assert_called_once_with()
    session.close.assert_called_once_with()


def test_query_failure_is_sanitized():
    session = Mock()
    report = TransactionReport()
    with pytest.raises(DatabaseOperationError) as caught:
        with session_scope(Mock(return_value=session), report=report):
            raise SQLAlchemyError(SENSITIVE_ERROR)
    assert caught.value.stage == "execute"
    assert caught.value.outcome is TransactionOutcome.NOT_COMMITTED
    assert SENSITIVE_ERROR not in formatted(caught.value)
    session.rollback.assert_called_once_with()
    session.close.assert_called_once_with()


@pytest.mark.parametrize("sqlstate,outcome", [
    ("23505", TransactionOutcome.NOT_COMMITTED),
    ("40001", TransactionOutcome.NOT_COMMITTED),
    ("40P01", TransactionOutcome.NOT_COMMITTED),
    ("08006", TransactionOutcome.UNKNOWN),
    ("08007", TransactionOutcome.UNKNOWN),
    (None, TransactionOutcome.UNKNOWN),
])
def test_commit_failure_classification(sqlstate, outcome):
    session = Mock()
    session.commit.side_effect = driver_error(sqlstate)
    report = TransactionReport()
    with pytest.raises(DatabaseCommitError) as caught:
        with session_scope(Mock(return_value=session), report=report):
            pass
    assert caught.value.stage == "commit"
    assert caught.value.outcome is outcome
    assert report.outcome is outcome
    assert report.failed_stage == "commit"
    assert SENSITIVE_ERROR not in formatted(caught.value)
    session.commit.assert_called_once_with()
    session.rollback.assert_called_once_with()
    session.close.assert_called_once_with()


def test_cleanup_after_successful_commit_retains_committed_outcome():
    session = Mock()
    session.close.side_effect = RuntimeError(SENSITIVE_ERROR)
    report = TransactionReport()
    with pytest.raises(DatabaseCleanupError) as caught:
        with session_scope(Mock(return_value=session), report=report):
            pass
    assert report.outcome is TransactionOutcome.COMMITTED
    assert caught.value.outcome is TransactionOutcome.COMMITTED
    assert caught.value.stage == "session_close"
    assert report.cleanup_failures == ["session_close"]
    session.commit.assert_called_once_with()
    session.rollback.assert_not_called()
    assert SENSITIVE_ERROR not in formatted(caught.value)


def test_commit_error_not_masked_by_rollback_and_close():
    session = Mock()
    session.commit.side_effect = driver_error("08006")
    session.rollback.side_effect = RuntimeError(SENSITIVE_ERROR)
    session.close.side_effect = RuntimeError(SENSITIVE_ERROR)
    report = TransactionReport()
    with pytest.raises(DatabaseCommitError) as caught:
        with session_scope(Mock(return_value=session), report=report):
            pass
    assert caught.value.stage == "commit"
    assert caught.value.outcome is TransactionOutcome.UNKNOWN
    assert report.cleanup_failures == ["rollback", "session_close"]
    assert caught.value.cleanup_failures == report.cleanup_failures
    assert SENSITIVE_ERROR not in formatted(caught.value)


def test_cleanup_preserves_application_error():
    session = Mock()
    session.rollback.side_effect = RuntimeError(SENSITIVE_ERROR)
    session.close.side_effect = RuntimeError(SENSITIVE_ERROR)
    primary = ValueError("Application failure")
    with pytest.raises(ValueError) as caught:
        with session_scope(Mock(return_value=session)):
            raise primary
    assert caught.value is primary
    assert len(primary.__notes__) == 2
    assert SENSITIVE_ERROR not in formatted(caught.value)


def test_interrupted_commit_remains_unknown_and_preserves_interrupt():
    session = Mock()
    primary = KeyboardInterrupt()
    session.commit.side_effect = primary
    session.close.side_effect = RuntimeError(SENSITIVE_ERROR)
    report = TransactionReport()
    with pytest.raises(KeyboardInterrupt) as caught:
        with session_scope(Mock(return_value=session), report=report):
            pass
    assert caught.value is primary
    assert report.outcome is TransactionOutcome.UNKNOWN
    assert report.failed_stage == "commit"
    assert SENSITIVE_ERROR not in formatted(caught.value)


def test_session_open_failure_is_sanitized():
    factory = Mock(side_effect=RuntimeError(SENSITIVE_ERROR))
    with pytest.raises(DatabaseOperationError) as caught:
        with session_scope(factory):
            pytest.fail("Must not yield a session.")
    assert caught.value.stage == "session_open"
    assert SENSITIVE_ERROR not in formatted(caught.value)


def test_each_worker_owns_its_session():
    sessions = []
    lock = Lock()

    def worker(_):
        session = Mock()
        report = TransactionReport()
        with session_scope(Mock(return_value=session), report=report):
            with lock:
                sessions.append(session)
        return report.outcome

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(worker, range(4)))
    assert len({id(session) for session in sessions}) == 4
    assert results == [TransactionOutcome.COMMITTED] * 4
    for session in sessions:
        session.close.assert_called_once_with()


def test_used_report_is_rejected_without_opening_session():
    factory = Mock()
    report = TransactionReport(outcome=TransactionOutcome.COMMITTED)
    with pytest.raises(ValueError, match="fresh transaction report"):
        with session_scope(factory, report=report):
            pass
    factory.assert_not_called()
