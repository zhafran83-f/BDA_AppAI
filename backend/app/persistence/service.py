"""Synchronous best-effort persistence. One fresh session/transaction per call."""

from contextlib import contextmanager
from threading import BoundedSemaphore, Condition, get_ident

from ..core.model_metadata import BundleSnapshot
from ..db import TransactionOutcome, TransactionReport
from ..db.session import session_scope
from ..repositories.predictions import PredictionRepository
from .contracts import (
    FashionRecord, PersistenceReason, PersistenceReceipt, PersistenceStatus,
    PurchaseRecord, ReviewRecord,
)


class _AdmissionGate:
    def __init__(self):
        self._condition = Condition()
        self._semaphore = BoundedSemaphore(2)
        self._accepting = True
        self._active = 0
        self._owners = {}

    @property
    def active(self):
        with self._condition:
            return self._active

    @property
    def accepting(self):
        with self._condition:
            return self._accepting

    @contextmanager
    def try_enter(self):
        acquired = False
        owner = get_ident()
        try:
            with self._condition:
                if not self._accepting:
                    rejection = PersistenceReason.SHUTDOWN
                elif not self._semaphore.acquire(blocking=False):
                    rejection = PersistenceReason.BUSY
                else:
                    acquired = True
                    self._active += 1
                    self._owners[owner] = self._owners.get(owner, 0) + 1
                    rejection = None
            yield rejection
        finally:
            if acquired:
                with self._condition:
                    self._active -= 1
                    self._owners[owner] -= 1
                    if not self._owners[owner]:
                        del self._owners[owner]
                    self._semaphore.release()
                    self._condition.notify_all()

    def assert_can_close(self):
        with self._condition:
            if self._owners.get(get_ident(), 0):
                raise RuntimeError("Cannot close persistence from an active operation.")

    def close_and_wait(self):
        with self._condition:
            if self._owners.get(get_ident(), 0):
                raise RuntimeError("Cannot close persistence from an active operation.")
            self._accepting = False
            self._condition.notify_all()
            self._condition.wait_for(lambda: self._active == 0)


class PersistenceService:
    def __init__(self, session_factory, metadata_provider, *, repository=None):
        self._factory = session_factory
        self._metadata = metadata_provider
        self._repository = PredictionRepository() if repository is None else repository
        self._gate = _AdmissionGate()

    @property
    def active_operations(self):
        return self._gate.active

    def assert_can_close(self):
        self._gate.assert_can_close()

    def quiesce(self):
        """Stop admission, wait for active operations; never dispose an engine."""
        self._gate.close_and_wait()

    @staticmethod
    def _rejected(record, reason):
        return PersistenceReceipt(
            record.id, PersistenceStatus.NOT_COMMITTED, reason
        )

    def _save(self, module, method, expected_type, record, bundle):
        if type(record) is not expected_type:
            raise TypeError("Persistence record type does not match the operation.")
        with self._gate.try_enter() as rejection:
            if rejection is not None:
                return self._rejected(record, rejection)
            try:
                snapshot = self._metadata.snapshot_for(module, bundle)
            except Exception:
                snapshot = None
            if not isinstance(snapshot, BundleSnapshot) or snapshot.module != module:
                return self._rejected(record, PersistenceReason.METADATA_UNVERIFIED)

            report = TransactionReport()
            try:
                with session_scope(self._factory, report=report) as session:
                    getattr(self._repository, method)(session, record, snapshot)
            except Exception:
                # report, not exception type or rollback success, determines outcome.
                pass

            if report.outcome is TransactionOutcome.COMMITTED:
                reason = (
                    PersistenceReason.CLEANUP_FAILED if report.cleanup_failures
                    else PersistenceReason.SAVED
                )
            else:
                reason = PersistenceReason.TRANSACTION_ERROR
            return PersistenceReceipt(
                record.id, PersistenceStatus(report.outcome.value), reason,
                tuple(report.cleanup_failures), report.failed_stage,
            )

    def save_fashion(self, record, *, bundle):
        return self._save("fashion", "add_fashion", FashionRecord, record, bundle)

    def save_review(self, record, *, bundle):
        return self._save("reviews", "add_review", ReviewRecord, record, bundle)

    def save_purchase(self, record, *, bundle):
        return self._save("purchase", "add_purchase", PurchaseRecord, record, bundle)
