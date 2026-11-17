"""Lazy, process-local caches for trusted artifacts; no training or HTTP."""
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Callable

from ..preprocessing.shopper_features import validate_feature_columns

MODEL_DIRECTORY = Path(__file__).resolve().parents[2] / "models"


@dataclass
class FashionBundle:
    model: Any
    lock: Any = field(default_factory=Lock, repr=False)


@dataclass
class ReviewBundle:
    vectorizer: Any
    kmeans: Any
    lock: Any = field(default_factory=Lock, repr=False)


@dataclass
class PurchaseBundle:
    models: dict[str, Any]
    feature_columns: tuple[str, ...]
    lock: Any = field(default_factory=Lock, repr=False)


def _load_joblib(path: Path) -> Any:
    import joblib
    return joblib.load(path)


def _load_keras(path: Path) -> Any:
    import keras
    return keras.models.load_model(path, compile=False, safe_mode=True)


class ModelLoader:
    """Reuse one bundle per journal per loader; serialize initial loading."""

    def __init__(self, model_directory: Path | None = None):
        self.model_directory = (
            MODEL_DIRECTORY if model_directory is None else Path(model_directory).resolve()
        )
        self._bundles: dict[str, Any] = {}
        self._load_lock = RLock()

    def _artifact(self, journal: str, filename: str) -> Path:
        path = self.model_directory / journal / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required local artifact is missing: {journal}/{filename}")
        return path

    def _get(self, key: str, create: Callable[[], Any]) -> Any:
        with self._load_lock:
            if key not in self._bundles:
                self._bundles[key] = create()
            return self._bundles[key]

    def fashion(self) -> FashionBundle:
        def create() -> FashionBundle:
            model = _load_keras(self._artifact("jurnal1", "vgg16_fashion_65_35.keras"))
            if tuple(model.input_shape) != (None, 224, 224, 3):
                raise RuntimeError("Fashion model input shape does not match training.")
            if tuple(model.output_shape) != (None, 4):
                raise RuntimeError("Fashion model output must contain four classes.")
            if str(model.inputs[0].dtype) != "float32":
                raise RuntimeError("Fashion model input dtype must be float32.")
            return FashionBundle(model)
        return self._get("fashion", create)

    def review(self) -> ReviewBundle:
        def create() -> ReviewBundle:
            vectorizer = _load_joblib(self._artifact("jurnal2", "tfidf_review.joblib"))
            kmeans = _load_joblib(self._artifact("jurnal2", "kmeans_review_k2.joblib"))
            size = len(vectorizer.vocabulary_)
            if (
                size != 5000
                or vectorizer.idf_.shape != (size,)
                or kmeans.n_clusters != 2
                or kmeans.n_features_in_ != size
                or kmeans.cluster_centers_.shape != (2, size)
            ):
                raise RuntimeError("Review bundle dimensions do not match training.")
            return ReviewBundle(vectorizer, kmeans)
        return self._get("review", create)

    def purchase(self) -> PurchaseBundle:
        def create() -> PurchaseBundle:
            columns = validate_feature_columns(
                _load_joblib(self._artifact("jurnal3", "feature_columns_jurnal3.joblib"))
            )
            filenames = {
                "random_forest": "random_forest_jurnal3.joblib",
                "svm": "svm_jurnal3.joblib",
                "mlp": "mlp_jurnal3.joblib",
            }
            models = {}
            for name, filename in filenames.items():
                model = _load_joblib(self._artifact("jurnal3", filename))
                if (
                    model.n_features_in_ != len(columns)
                    or tuple(model.feature_names_in_) != columns
                ):
                    raise RuntimeError(f"{name} feature order does not match the artifact.")
                if set(model.classes_) != {0, 1} or len(model.classes_) != 2:
                    raise RuntimeError(f"{name} must expose Revenue classes 0 and 1.")
                if name != "svm" and not callable(getattr(model, "predict_proba", None)):
                    raise RuntimeError(f"{name} must support predict_proba.")
                models[name] = model
            return PurchaseBundle(models, columns)
        return self._get("purchase", create)


# Constructing a loader does not open artifacts or import TensorFlow/Keras.
DEFAULT_LOADER = ModelLoader()
