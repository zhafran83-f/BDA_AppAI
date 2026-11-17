"""One session and report per synchronous repository operation in one thread."""

from contextlib import contextmanager

from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from . import (
    DatabaseCleanupError,
    DatabaseCommitError,
    DatabaseOperationError,
    TransactionOutcome,
    TransactionReport,
    record_cleanup_failure,
)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create a reusable factory; this does not create a session or connect."""
    return sessionmaker(
        bind=engine,
        class_=Session,
        autoflush=False,
        expire_on_commit=False,
    )


def _commit_failure_outcome(exc: Exception) -> TransactionOutcome:
    original = getattr(exc, "orig", None)
    driver_error = original if original is not None else exc
    sqlstate = getattr(driver_error, "sqlstate", None)
    # Statement completion is uncertain despite belonging to SQLSTATE class 40.
    if sqlstate == "40003":
        return TransactionOutcome.UNKNOWN
    # Other server integrity/transaction-rollback errors confirm rejection.
    # Transport failures, missing SQLSTATE and other failures stay uncertain.
    if isinstance(sqlstate, str) and sqlstate[:2] in ("23", "40"):
        return TransactionOutcome.NOT_COMMITTED
    return TransactionOutcome.UNKNOWN


@contextmanager
def session_scope(factory, *, report: TransactionReport | None = None):
    """Commit on success, rollback on failure, always close.

    Create/use/close this context in the same synchronous worker function.
    Never share sessions or reports between requests/threads. No automatic retry.
    A successful rollback after a commit error does NOT prove commit failed.
    """
    state = report if report is not None else TransactionReport()
    if (
        state.outcome != TransactionOutcome.NOT_COMMITTED
        or state.failed_stage is not None
        or state.cleanup_failures
    ):
        raise ValueError("A fresh transaction report is required.")

    session = None
    primary = None
    stage = "session_open"
    try:
        try:
            session = factory()
        except Exception:
            raise DatabaseOperationError(stage, state.outcome) from None

        stage = "execute"
        try:
            yield session
        except SQLAlchemyError:
            raise DatabaseOperationError(stage, state.outcome) from None

        stage = "commit"
        state.outcome = TransactionOutcome.UNKNOWN
        try:
            session.commit()
        except Exception as exc:
            state.outcome = _commit_failure_outcome(exc)
            raise DatabaseCommitError(stage, state.outcome) from None
        state.outcome = TransactionOutcome.COMMITTED

    except BaseException as exc:
        primary = exc
        state.failed_stage = stage
        if session is not None:
            try:
                session.rollback()
            except Exception:
                state.cleanup_failures.append("rollback")
                record_cleanup_failure(primary, "rollback")
        raise
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                state.cleanup_failures.append("session_close")
                if primary is not None:
                    record_cleanup_failure(primary, "session_close")
                else:
                    state.failed_stage = "session_close"
                    raise DatabaseCleanupError(
                        "session_close", state.outcome
                    ) from None
