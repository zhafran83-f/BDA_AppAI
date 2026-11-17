"""Optional database infrastructure with safe transaction outcome reporting."""

from dataclasses import dataclass, field
from enum import Enum


class TransactionOutcome(str, Enum):
    NOT_COMMITTED = "not_committed"
    COMMITTED = "committed"
    UNKNOWN = "unknown"


@dataclass
class TransactionReport:
    """Use a fresh report per operation; never share it between threads.

    COMMITTED means commit returned successfully, even if subsequent cleanup
    fails. UNKNOWN must not be treated as failed persistence or retried blindly.
    """

    outcome: TransactionOutcome = TransactionOutcome.NOT_COMMITTED
    failed_stage: str | None = None
    cleanup_failures: list[str] = field(default_factory=list)


class DatabaseOperationError(RuntimeError):
    """Safe error metadata; never retain raw SQL, parameters, or driver text."""

    def __init__(self, stage: str, outcome: TransactionOutcome | None = None):
        super().__init__("Database operation failed.")
        self.stage = stage
        self.outcome = outcome
        self.cleanup_failures: list[str] = []


class DatabaseCommitError(DatabaseOperationError):
    """Commit failed; inspect outcome to distinguish rejection from uncertainty."""


class DatabaseCleanupError(DatabaseOperationError):
    """Cleanup failed with no earlier exception; inspect outcome before retrying."""


def record_cleanup_failure(primary: BaseException, stage: str) -> None:
    """Attach only controlled metadata while preserving the primary exception."""
    if isinstance(primary, DatabaseOperationError):
        primary.cleanup_failures.append(stage)
    primary.add_note("Database cleanup failed at " + stage + ".")
