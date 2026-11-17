"""Transaction outcomes/admission tests use fake sessions and fake bundles."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sqlalchemy.exc import DBAPIError, SQLAlchemyError

from backend.app.db import TransactionOutcome, TransactionReport
from backend.app.persistence import service as service_module
from backend.app.persistence.contracts import PersistenceReason, PersistenceStatus
from backend.app.persistence.service import PersistenceService
from backend.tests.test_persistence_contracts import (
    fashion_record, purchase_record, review_record, snapshot,
)


SECRET = "dummy-driver-secret"
BUILDERS = {"fashion": fashion_record, "reviews": review_record, "purchase": purchase_record}
METHODS = {"fashion": "save_fashion", "reviews": "save_review", "purchase": "save_purchase"}


def fake_provider():
    bundles = {module: object() for module in BUILDERS}
    expected = dict(bundles)  # Keep the provider's identity registry independent of test inputs.
    provider = SimpleNamespace(snapshot_for=Mock(side_effect=lambda module, bundle:
        snapshot(module) if expected.get(module) is bundle else None))
    return provider, bundles


def make_service(session=None, repository=None, factory=None):
    provider, bundles = fake_provider()
    session = Mock() if session is None else session
    factory = Mock(return_value=session) if factory is None else factory
    service = PersistenceService(factory, provider, repository=repository)
    return service, factory, provider, bundles, session


def save(service, bundles, module="reviews"):
    return getattr(service, METHODS[module])(BUILDERS[module](), bundle=bundles[module])


def db_error(state):
    original = Exception(SECRET)
    original.sqlstate = state
    return DBAPIError("dummy SQL must not leak", {}, original, False)


@pytest.mark.parametrize("module", BUILDERS)
def test_one_session_one_transaction_and_no_orm_return(module):
    service, factory, _, bundles, session = make_service()
    result = save(service, bundles, module)
    assert result.status is PersistenceStatus.COMMITTED
    assert result.outcome is TransactionOutcome.COMMITTED
    assert result.reason is PersistenceReason.SAVED
    factory.assert_called_once_with()
    session.add.assert_called_once()
    assert session.add.call_args.args[0].id == result.prediction_id
    session.commit.assert_called_once_with()
    session.rollback.assert_not_called()
    session.close.assert_called_once_with()
    assert service.active_operations == 0


@pytest.mark.parametrize("state,outcome", [
    ("23505", TransactionOutcome.NOT_COMMITTED),
    ("40001", TransactionOutcome.NOT_COMMITTED),
    ("40P01", TransactionOutcome.NOT_COMMITTED),
    ("40003", TransactionOutcome.UNKNOWN),
    ("08006", TransactionOutcome.UNKNOWN),
    (None, TransactionOutcome.UNKNOWN),
])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_commit_outcome_unchanged_by_rollback_cleanup_and_no_retry(state, outcome, cleanup_fails):
    service, factory, _, bundles, session = make_service()
    session.commit.side_effect = db_error(state)
    if cleanup_fails:
        session.rollback.side_effect = RuntimeError(SECRET)
        session.close.side_effect = RuntimeError(SECRET)
    result = save(service, bundles)
    assert result.outcome is outcome
    assert result.reason is PersistenceReason.TRANSACTION_ERROR
    assert result.failed_stage == "commit"
    assert result.cleanup_failures == (("rollback", "session_close") if cleanup_fails else ())
    factory.assert_called_once()
    session.commit.assert_called_once()
    session.rollback.assert_called_once()
    session.close.assert_called_once()
    assert service.active_operations == 0


def test_successful_commit_with_close_error_is_still_committed():
    service, factory, _, bundles, session = make_service()
    session.close.side_effect = RuntimeError(SECRET)
    result = save(service, bundles)
    assert result.status is PersistenceStatus.COMMITTED
    assert result.reason is PersistenceReason.CLEANUP_FAILED
    assert result.cleanup_failures == ("session_close",)
    session.rollback.assert_not_called()
    factory.assert_called_once()


@pytest.mark.parametrize("failure", ["factory", "repository"])
def test_precommit_failure_does_not_start_commit_or_leak_slot(failure):
    repository = Mock()
    service, factory, _, bundles, session = make_service(repository=repository)
    if failure == "factory":
        factory.side_effect = RuntimeError(SECRET)
    else:
        repository.add_review.side_effect = SQLAlchemyError(SECRET)
    result = save(service, bundles)
    assert result.status is PersistenceStatus.NOT_COMMITTED
    session.commit.assert_not_called()
    assert service.active_operations == 0
    factory.side_effect = None
    repository.add_review.side_effect = None
    assert save(service, bundles).status is PersistenceStatus.COMMITTED


@pytest.mark.parametrize("failure", ["missing", "wrong_bundle", "wrong_module", "provider_error"])
def test_unverified_metadata_never_opens_session(failure):
    service, factory, provider, bundles, _ = make_service()
    if failure == "wrong_bundle":
        bundles["reviews"] = object()
    elif failure == "provider_error":
        provider.snapshot_for.side_effect = OSError(SECRET)
    else:
        provider.snapshot_for.side_effect = None
        provider.snapshot_for.return_value = None if failure == "missing" else snapshot("fashion")
    assert save(service, bundles).reason is PersistenceReason.METADATA_UNVERIFIED
    factory.assert_not_called()
    assert service.active_operations == 0


def test_errors_emit_no_sensitive_output_or_log(capsys, caplog):
    service, _, _, bundles, session = make_service()
    session.commit.side_effect = db_error("40003")
    session.rollback.side_effect = RuntimeError(SECRET)
    session.close.side_effect = RuntimeError(SECRET)
    result = save(service, bundles)
    assert SECRET not in repr(result)
    assert "dummy SQL" not in repr(result)
    captured = capsys.readouterr()
    assert captured.out == captured.err == caplog.text == ""


@pytest.mark.parametrize("stage", ["repository", "commit", "rollback", "close"])
def test_interruption_releases_slot_and_does_not_rewrite_report(monkeypatch, stage):
    repository = Mock()
    session = Mock()
    observed = TransactionReport()
    monkeypatch.setattr(service_module, "TransactionReport",
                        Mock(side_effect=[observed, TransactionReport()]))
    service, factory, _, bundles, _ = make_service(session=session, repository=repository)
    if stage == "repository":
        repository.add_review.side_effect = KeyboardInterrupt()
    elif stage == "commit":
        session.commit.side_effect = KeyboardInterrupt()
    else:
        session.commit.side_effect = db_error("40003")
        getattr(session, stage).side_effect = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        save(service, bundles)
    assert service.active_operations == 0
    assert observed.outcome is (
        TransactionOutcome.NOT_COMMITTED if stage == "repository"
        else TransactionOutcome.UNKNOWN
    )
    factory.assert_called_once()
    session.close.assert_called_once()
    repository.add_review.side_effect = None
    session.commit.side_effect = None
    session.rollback.side_effect = None
    session.close.side_effect = None
    assert save(service, bundles).outcome is TransactionOutcome.COMMITTED


def test_shared_capacity_two_across_all_modules_and_no_busy_session():
    provider, bundles = fake_provider()
    entered = [Event(), Event()]
    release = Event()
    lock = Lock()
    sessions = []

    def make_session():
        with lock:
            index = len(sessions)
            session = Mock()
            sessions.append(session)

        def commit():
            if index < 2:
                entered[index].set()
            assert release.wait(5)
        session.commit.side_effect = commit
        return session

    factory = Mock(side_effect=make_session)
    service = PersistenceService(factory, provider)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(save, service, bundles, "fashion")
        second = executor.submit(save, service, bundles, "reviews")
        try:
            assert all(event.wait(5) for event in entered)
            assert service.active_operations == 2
            rejected = save(service, bundles, "purchase")
            assert rejected.reason is PersistenceReason.BUSY
            assert factory.call_count == 2
        finally:
            release.set()
        assert first.result(timeout=5).status is PersistenceStatus.COMMITTED
        assert second.result(timeout=5).status is PersistenceStatus.COMMITTED
    assert service.active_operations == 0
    assert save(service, bundles, "purchase").status is PersistenceStatus.COMMITTED
    for session in sessions:
        session.close.assert_called_once()


def test_quiesce_waits_for_active_operation_and_rejects_new_work():
    service, factory, _, bundles, session = make_service()
    entered, release = Event(), Event()

    def commit():
        entered.set()
        assert release.wait(5)
    session.commit.side_effect = commit
    with ThreadPoolExecutor(max_workers=2) as executor:
        active = executor.submit(save, service, bundles)
        try:
            assert entered.wait(5)
            closing = executor.submit(service.quiesce)
            condition = service._gate._condition
            with condition:
                assert condition.wait_for(lambda: not service._gate._accepting, timeout=5)
            assert not closing.done()
            assert save(service, bundles).reason is PersistenceReason.SHUTDOWN
            factory.assert_called_once()
        finally:
            release.set()
        assert active.result(timeout=5).status is PersistenceStatus.COMMITTED
        closing.result(timeout=5)
    assert service.active_operations == 0
    service.quiesce()  # Idempotent.


def test_self_quiesce_is_rejected_without_deadlock_or_gate_leak():
    repository = Mock()
    service, _, _, bundles, _ = make_service(repository=repository)
    repository.add_review.side_effect = lambda *args: service.quiesce()
    assert save(service, bundles).status is PersistenceStatus.NOT_COMMITTED
    assert service.active_operations == 0
    repository.add_review.side_effect = None
    assert save(service, bundles).status is PersistenceStatus.COMMITTED


def test_wrong_record_type_never_obtains_slot_or_session():
    service, factory, _, bundles, _ = make_service()
    with pytest.raises(TypeError):
        service.save_review(purchase_record(), bundle=bundles["reviews"])
    factory.assert_not_called()
    assert service.active_operations == 0
