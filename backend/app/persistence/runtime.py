"""Explicit lifecycle foundation; deliberately not integrated with FastAPI.

CONFIGURED means local factories exist, not that PostgreSQL is reachable.
Run each save in one synchronous worker. close() blocks while saves are active.
An interrupted/failed close retains ownership for a later explicit close call.
"""

from enum import Enum
from threading import Lock

from ..core.config import load_database_settings
from ..core.model_metadata import VerifiedBundleProvider
from ..db.engine import create_database_engine
from ..db.session import create_session_factory
from .contracts import PersistenceReason, PersistenceReceipt, PersistenceStatus
from .service import PersistenceService


class RuntimeState(str, Enum):
    NEW = "new"
    DISABLED = "disabled"
    CONFIGURED = "configured"
    UNAVAILABLE = "unavailable"
    CLOSING = "closing"
    CLOSED = "closed"


class PersistenceRuntime:
    def __init__(self, *, settings_provider=None, engine_factory=None,
                 session_factory_builder=None, metadata_factory=None, repository=None):
        self._settings_provider = settings_provider or load_database_settings
        self._engine_factory = engine_factory or create_database_engine
        self._session_builder = session_factory_builder or create_session_factory
        self._metadata_factory = metadata_factory or VerifiedBundleProvider
        self._repository = repository
        self._lifecycle_lock = Lock()
        self._state_lock = Lock()
        self._state = RuntimeState.NEW
        self._engine = None
        self._service = None
        self._metadata = None
        self._cleanup_failed = False

    @property
    def state(self):
        with self._state_lock:
            return self._state

    @property
    def cleanup_failed(self):
        with self._state_lock:
            return self._cleanup_failed

    @property
    def bundle_provider(self):
        """Future inference must use this exact provider, not a second loader."""
        with self._state_lock:
            return self._metadata

    def start(self):
        with self._lifecycle_lock:
            if self.state is not RuntimeState.NEW:
                return
            engine = None
            try:
                settings = self._settings_provider()
                if not settings.enabled:
                    with self._state_lock:
                        self._state = RuntimeState.DISABLED
                    return
                if (
                    settings.user != "bda_app_runtime"
                    or settings.name != "bda_ecommerce_ai"
                    or settings.host != "127.0.0.1"
                ):
                    raise ValueError("Unexpected runtime database target.")
                engine = self._engine_factory(settings)
                if engine is None:
                    raise ValueError("Enabled database requires an engine.")
                factory = self._session_builder(engine)
                provider = self._metadata_factory()
                service = PersistenceService(factory, provider, repository=self._repository)
                with self._state_lock:
                    self._engine = engine
                    self._metadata = provider
                    self._service = service
                    self._state = RuntimeState.CONFIGURED
            except BaseException as primary:
                disposed = engine is None
                if engine is not None:
                    try:
                        engine.dispose()
                        disposed = True
                    except BaseException:
                        # Preserve the initialization failure/interruption.
                        pass
                with self._state_lock:
                    self._engine = None if disposed else engine
                    self._cleanup_failed = not disposed
                    self._state = RuntimeState.UNAVAILABLE
                if not isinstance(primary, Exception):
                    raise

    def _save(self, method, record, bundle):
        with self._state_lock:
            state, service = self._state, self._service
        if state is RuntimeState.CONFIGURED:
            return getattr(service, method)(record, bundle=bundle)
        if state is RuntimeState.DISABLED:
            return PersistenceReceipt(
                record.id, PersistenceStatus.DISABLED, PersistenceReason.DISABLED
            )
        reason = {
            RuntimeState.NEW: PersistenceReason.NOT_STARTED,
            RuntimeState.UNAVAILABLE: PersistenceReason.UNAVAILABLE,
            RuntimeState.CLOSING: PersistenceReason.SHUTDOWN,
            RuntimeState.CLOSED: PersistenceReason.SHUTDOWN,
        }[state]
        return PersistenceReceipt(record.id, PersistenceStatus.NOT_COMMITTED, reason)

    def save_fashion(self, record, *, bundle):
        return self._save("save_fashion", record, bundle)

    def save_review(self, record, *, bundle):
        return self._save("save_review", record, bundle)

    def save_purchase(self, record, *, bundle):
        return self._save("save_purchase", record, bundle)

    def close(self):
        """Return cleanup success. No disposal before the service has drained."""
        # Check before taking the lifecycle lock: another closer may be waiting
        # for this very worker. Refuse self-close instead of deadlocking.
        with self._state_lock:
            service = self._service
        if service is not None:
            service.assert_can_close()
        with self._lifecycle_lock:
            with self._state_lock:
                if self._state is RuntimeState.CLOSED:
                    return True
                self._state = RuntimeState.CLOSING
                service, engine = self._service, self._engine
            if service is not None:
                service.quiesce()
            if engine is not None:
                try:
                    engine.dispose()
                except Exception:
                    with self._state_lock:
                        self._cleanup_failed = True
                    return False
            with self._state_lock:
                self._engine = None
                self._service = None
                self._state = RuntimeState.CLOSED
            return True
