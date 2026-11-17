"""Lifespan-owned loading state around the existing, unchanged ModelLoader."""
import logging
from threading import Lock

from ..services.fashion_service import FashionService
from ..services.review_service import ReviewService
from ..services.purchase_service import PurchaseService

logger = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    pass


class InferenceRuntime:
    def __init__(self, loader):
        self._loader = loader
        self._bundles = {}
        self._states = {name: "not_loaded" for name in ("fashion", "reviews", "purchase")}
        self._locks = {name: Lock() for name in self._states}
        self._state_lock = Lock()
        self.active = True
        self.services = {
            "fashion": FashionService(self),
            "reviews": ReviewService(self),
            "purchase": PurchaseService(self),
        }

    def _state(self, name, value):
        with self._state_lock:
            self._states[name] = value

    def snapshot(self):
        with self._state_lock:
            return dict(self._states)

    def mark_ready(self, name):
        self._state(name, "ready")

    def mark_unavailable(self, name):
        self._state(name, "unavailable")

    def _bundle(self, name, method):
        with self._locks[name]:
            if not self.active:
                raise ModelUnavailableError("Runtime is not active.")
            if name in self._bundles:
                return self._bundles[name]
            self._state(name, "loading")
            try:
                bundle = getattr(self._loader, method)()
            except Exception as exc:
                self.mark_unavailable(name)
                logger.exception("Could not load the %s model bundle.", name)
                raise ModelUnavailableError("Required model is unavailable.") from exc
            self._bundles[name] = bundle
            self.mark_ready(name)
            return bundle

    def fashion(self):
        return self._bundle("fashion", "fashion")

    def review(self):
        return self._bundle("reviews", "review")

    def purchase(self):
        return self._bundle("purchase", "purchase")

    def warmup(self):
        # The smaller sklearn bundles are eager; VGG16 stays lazy.
        for load in (self.review, self.purchase):
            try:
                load()
            except ModelUnavailableError:
                pass  # Keep health available; readiness exposes the failure.

    def close(self):
        self.active = False
        self.services.clear()
        self._bundles.clear()
        self._loader = None
