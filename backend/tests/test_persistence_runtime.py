"""Runtime lifecycle tests with fake engines/sessions, never PostgreSQL."""

from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
from pathlib import Path
import os
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from backend.app.core.config import DatabaseSettings
from backend.app.persistence import runtime as runtime_module
from backend.app.persistence.contracts import PersistenceReason, PersistenceStatus
from backend.app.persistence.runtime import PersistenceRuntime, RuntimeState
from backend.tests.test_persistence_contracts import review_record
from backend.tests.test_persistence_service import fake_provider


class IsolatedSettings(DatabaseSettings):
    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings,
    ):
        return (init_settings,)


@pytest.fixture(autouse=True)
def forbid_real_connections(monkeypatch):
    import psycopg
    forbidden = Mock(side_effect=AssertionError("Real PostgreSQL connections forbidden."))
    monkeypatch.setattr(psycopg, "connect", forbidden)
    yield
    forbidden.assert_not_called()


def setup_runtime(**changes):
    engine = Mock()
    session = Mock()
    factory = Mock(return_value=session)
    provider, bundles = fake_provider()
    settings = IsolatedSettings(enabled=True, password="dummy-only")
    settings_reader = Mock(return_value=settings)
    engine_builder = Mock(return_value=engine)
    session_builder = Mock(return_value=factory)
    metadata_builder = Mock(return_value=provider)
    options = dict(
        settings_provider=settings_reader, engine_factory=engine_builder,
        session_factory_builder=session_builder, metadata_factory=metadata_builder,
    )
    options.update(changes)
    runtime = PersistenceRuntime(**options)
    return SimpleNamespace(
        runtime=runtime, engine=engine, session=session, factory=factory,
        provider=provider, bundles=bundles, settings_reader=settings_reader,
        engine_builder=engine_builder, session_builder=session_builder,
        metadata_builder=metadata_builder,
    )


def save(case):
    return case.runtime.save_review(review_record(), bundle=case.bundles["reviews"])


def test_import_and_construction_do_not_read_settings_or_create_engine():
    code = """
import sys
from unittest.mock import Mock
sys.path.insert(0, sys.argv[1])
import psycopg
from backend.app.core import config
from backend.app.db import engine
blocked = Mock(side_effect=AssertionError("No IO at import or construction."))
psycopg.connect = blocked
config.load_database_settings = blocked
engine.create_database_engine = blocked
from backend.app.persistence.runtime import PersistenceRuntime
runtime = PersistenceRuntime()
assert runtime.state.value == "new"
blocked.assert_not_called()
"""
    env = {k: v for k, v in os.environ.items()
           if k.upper() in {"SYSTEMROOT", "WINDIR", "TEMP", "TMP"}}
    env.update({"SystemRoot": r"C:\Windows", "WINDIR": r"C:\Windows",
                "PYTHONDONTWRITEBYTECODE": "1"})
    result = subprocess.run(
        [sys.executable, "-B", "-I", "-c", code, str(Path(__file__).resolve().parents[2])],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_default_database_disabled_without_any_resource_creation():
    engine, session, metadata = Mock(), Mock(), Mock()
    runtime = PersistenceRuntime(
        engine_factory=engine, session_factory_builder=session, metadata_factory=metadata,
    )
    with patch.dict(os.environ, {}, clear=True):
        runtime.start()
    assert runtime.state is RuntimeState.DISABLED
    result = runtime.save_review(review_record(), bundle=object())
    assert result.status is PersistenceStatus.DISABLED
    assert runtime.bundle_provider is None
    engine.assert_not_called()
    session.assert_not_called()
    metadata.assert_not_called()
    assert runtime.close()
    assert runtime.close()


def test_explicit_start_is_lazy_configured_and_idempotent():
    case = setup_runtime()
    assert case.runtime.state is RuntimeState.NEW
    assert save(case).reason is PersistenceReason.NOT_STARTED
    case.settings_reader.assert_not_called()
    case.runtime.start()
    case.runtime.start()
    assert case.runtime.state is RuntimeState.CONFIGURED
    assert case.runtime.bundle_provider is case.provider
    case.engine_builder.assert_called_once_with(case.settings_reader.return_value)
    case.session_builder.assert_called_once_with(case.engine)
    case.metadata_builder.assert_called_once_with()
    case.engine.connect.assert_not_called()
    case.factory.assert_not_called()
    assert save(case).status is PersistenceStatus.COMMITTED
    case.runtime.close()
    case.runtime.close()
    case.engine.dispose.assert_called_once_with()
    assert case.runtime.state is RuntimeState.CLOSED
    assert save(case).reason is PersistenceReason.SHUTDOWN


@pytest.mark.parametrize("failure", ["settings", "engine", "session_factory", "metadata"])
def test_initialization_failure_is_nonfatal_sanitized_and_cleans_partial_engine(failure, capsys, caplog):
    case = setup_runtime()
    selected = {
        "settings": case.settings_reader, "engine": case.engine_builder,
        "session_factory": case.session_builder, "metadata": case.metadata_builder,
    }[failure]
    selected.side_effect = RuntimeError("dummy-secret-never-log")
    case.runtime.start()
    assert case.runtime.state is RuntimeState.UNAVAILABLE
    result = save(case)
    assert result.reason is PersistenceReason.UNAVAILABLE
    assert "dummy-secret" not in repr(result)
    if failure in ("session_factory", "metadata"):
        case.engine.dispose.assert_called_once()
    else:
        case.engine.dispose.assert_not_called()
    assert case.runtime.close()
    captured = capsys.readouterr()
    assert captured.out == captured.err == caplog.text == ""


@pytest.mark.parametrize("user", ["postgres", "bda_app_migrator"])
def test_runtime_cannot_use_privileged_or_migration_role(user):
    settings = SimpleNamespace(enabled=True, user=user, name="bda_ecommerce_ai", host="127.0.0.1")
    case = setup_runtime(settings_provider=lambda: settings)
    case.runtime.start()
    assert case.runtime.state is RuntimeState.UNAVAILABLE
    case.engine_builder.assert_not_called()


def test_disposal_error_does_not_change_committed_receipt_and_retains_ownership():
    case = setup_runtime()
    case.runtime.start()
    result = save(case)
    case.engine.dispose.side_effect = RuntimeError("dummy-secret")
    assert case.runtime.close() is False
    assert case.runtime.state is RuntimeState.CLOSING
    assert case.runtime.cleanup_failed is True
    assert result.status is PersistenceStatus.COMMITTED
    assert save(case).reason is PersistenceReason.SHUTDOWN
    case.engine.dispose.side_effect = None
    assert case.runtime.close()
    assert case.runtime.state is RuntimeState.CLOSED
    assert case.engine.dispose.call_count == 2  # Explicit cleanup call, not transaction retry.
    case.factory.assert_called_once()


def test_failed_partial_initialization_retains_engine_if_cleanup_fails():
    case = setup_runtime()
    case.session_builder.side_effect = RuntimeError("dummy-startup-error")
    case.engine.dispose.side_effect = RuntimeError("dummy-dispose-error")
    case.runtime.start()
    assert case.runtime.state is RuntimeState.UNAVAILABLE
    assert case.runtime.cleanup_failed
    case.engine.dispose.side_effect = None
    assert case.runtime.close()
    assert case.engine.dispose.call_count == 2


def test_shutdown_does_not_dispose_engine_before_active_commit_and_close_finish():
    case = setup_runtime()
    case.runtime.start()
    service = case.runtime._service
    entered, release = Event(), Event()

    def commit():
        entered.set()
        assert release.wait(5)
    case.session.commit.side_effect = commit

    def dispose():
        assert service.active_operations == 0
        case.session.close.assert_called_once()
    case.engine.dispose.side_effect = dispose

    with ThreadPoolExecutor(max_workers=2) as executor:
        active = executor.submit(save, case)
        try:
            assert entered.wait(5)
            closing = executor.submit(case.runtime.close)
            condition = service._gate._condition
            with condition:
                assert condition.wait_for(lambda: not service._gate._accepting, timeout=5)
            assert not closing.done()
            case.engine.dispose.assert_not_called()
            assert save(case).reason is PersistenceReason.SHUTDOWN
        finally:
            release.set()
        assert active.result(timeout=5).status is PersistenceStatus.COMMITTED
        assert closing.result(timeout=5)
    case.engine.dispose.assert_called_once()


def test_interrupted_shutdown_never_disposes_active_engine(monkeypatch):
    case = setup_runtime()
    case.runtime.start()
    service = case.runtime._service
    entered, release = Event(), Event()

    def commit():
        entered.set()
        assert release.wait(5)
    case.session.commit.side_effect = commit

    with ThreadPoolExecutor(max_workers=1) as executor:
        active = executor.submit(save, case)
        try:
            assert entered.wait(5)
            with monkeypatch.context() as patcher:
                patcher.setattr(service._gate._condition, "wait_for",
                                Mock(side_effect=KeyboardInterrupt()))
                with pytest.raises(KeyboardInterrupt):
                    case.runtime.close()
            assert case.runtime.state is RuntimeState.CLOSING
            case.engine.dispose.assert_not_called()
            assert save(case).reason is PersistenceReason.SHUTDOWN
        finally:
            release.set()
        assert active.result(timeout=5).status is PersistenceStatus.COMMITTED
    assert case.runtime.close()
    case.engine.dispose.assert_called_once()


def test_self_shutdown_refuses_before_lifecycle_lock():
    repository = Mock()
    case = setup_runtime(repository=repository)
    case.runtime.start()
    repository.add_review.side_effect = lambda *args: case.runtime.close()
    result = save(case)
    assert result.status is PersistenceStatus.NOT_COMMITTED
    assert case.runtime.state is RuntimeState.CONFIGURED
    case.engine.dispose.assert_not_called()
    assert case.runtime._service.active_operations == 0
    assert case.runtime.close()


def test_start_interruption_cleans_partial_resources_and_preserves_interrupt():
    case = setup_runtime()
    primary = KeyboardInterrupt()
    case.session_builder.side_effect = primary
    with pytest.raises(KeyboardInterrupt) as caught:
        case.runtime.start()
    assert caught.value is primary
    assert case.runtime.state is RuntimeState.UNAVAILABLE
    case.engine.dispose.assert_called_once()
    assert case.runtime.close()
