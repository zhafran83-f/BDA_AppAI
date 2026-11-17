"""Owned lazy loader with immutable provenance for its exact cached bundles.

Use THIS provider as the inference loader in a future integration. Never attach
file hashes to a preloaded external loader. Production artifacts must remain
unchanged during loading; replacing files requires a new provider/deployment.
Metadata failure skips persistence but does not invalidate a successfully loaded
inference bundle. Constructors/imports perform no hashing or model loading.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from threading import Lock
from types import MappingProxyType

from .model_loader import MODEL_DIRECTORY, ModelLoader


@dataclass(frozen=True)
class ArtifactSpec:
    key: str
    relative_path: str
    sha256: str


@dataclass(frozen=True)
class BundleSpec:
    model_version: str
    preprocessing_version: str
    artifacts: tuple[ArtifactSpec, ...]


ARTIFACT_KEYS = MappingProxyType({
    "fashion": ("vgg16",),
    "reviews": ("tfidf", "kmeans"),
    "purchase": ("random_forest", "svm", "mlp", "feature_columns"),
})

BASELINE_MANIFEST = MappingProxyType({
    "fashion": BundleSpec("jurnal1-bundle-v1", "fashion-rgb224-div255-v1", (
        ArtifactSpec("vgg16", "jurnal1/vgg16_fashion_65_35.keras",
                     "dd6edc3347458c34a7a3342108c570074d29577e13f5b8a17ada0ea73ffb0a54"),
    )),
    "reviews": BundleSpec("jurnal2-bundle-v1", "reviews-clean-text-v1", (
        ArtifactSpec("tfidf", "jurnal2/tfidf_review.joblib",
                     "4d98ebce0673acf065cdb7ca69fde62ca0fb5726697d1d34768a100cee254629"),
        ArtifactSpec("kmeans", "jurnal2/kmeans_review_k2.joblib",
                     "0b864ba9b6466e509c376939dc986e2ce4351e825b139aedd155c51e170bca0e"),
    )),
    "purchase": BundleSpec("jurnal3-bundle-v1", "shopper-28-features-v1", (
        ArtifactSpec("random_forest", "jurnal3/random_forest_jurnal3.joblib",
                     "e9b2654c0e5b1728562e12b6fd7e9083d0d4efde63df7ebc0f238914c70a3973"),
        ArtifactSpec("svm", "jurnal3/svm_jurnal3.joblib",
                     "9320dd169f9a86cec5bf56f0751ae27c3a2559e15effc7489159bdcb6752f21b"),
        ArtifactSpec("mlp", "jurnal3/mlp_jurnal3.joblib",
                     "d71fe270522b879163cbc1750f7aa0b85ded34b7d60053bcfff384963d4a39a9"),
        ArtifactSpec("feature_columns", "jurnal3/feature_columns_jurnal3.joblib",
                     "e4780da145b5a5d8b79b44fcefcb963f39425a36cda9a1d2077071ed34787f8b"),
    )),
})


@dataclass(frozen=True)
class BundleSnapshot:
    module: str
    model_version: str
    preprocessing_version: str
    artifact_sha256: tuple[tuple[str, str], ...]

    def __post_init__(self):
        if self.module not in ARTIFACT_KEYS:
            raise ValueError("Unknown model module.")
        if (
            not isinstance(self.model_version, str) or not self.model_version.strip()
            or not isinstance(self.preprocessing_version, str)
            or not self.preprocessing_version.strip()
            or not isinstance(self.artifact_sha256, tuple)
            or any(not isinstance(pair, tuple) or len(pair) != 2
                   for pair in self.artifact_sha256)
        ):
            raise ValueError("Invalid model metadata.")
        if tuple(key for key, _ in self.artifact_sha256) != ARTIFACT_KEYS[self.module]:
            raise ValueError("Artifact keys do not match the model module.")
        if any(not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{64}", sha) is None
               for _, sha in self.artifact_sha256):
            raise ValueError("Invalid artifact checksum.")


def _hash_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class VerifiedBundleProvider:
    def __init__(self, model_directory=None, *, loader_factory=None,
                 hasher=None, manifest=None):
        self._directory = Path(
            MODEL_DIRECTORY if model_directory is None else model_directory
        ).resolve()
        self._loader_factory = ModelLoader if loader_factory is None else loader_factory
        self._hasher = _hash_file if hasher is None else hasher
        self._manifest = MappingProxyType(dict(
            BASELINE_MANIFEST if manifest is None else manifest
        ))
        self._loader = None
        self._loader_lock = Lock()
        self._locks = {name: Lock() for name in ARTIFACT_KEYS}
        self._bundles = {}
        self._snapshots = {}
        self._attempted = set()

    def _owned_loader(self):
        with self._loader_lock:
            if self._loader is None:
                self._loader = self._loader_factory(self._directory)
            return self._loader

    def _capture(self, module):
        try:
            spec = self._manifest[module]
            if tuple(item.key for item in spec.artifacts) != ARTIFACT_KEYS[module]:
                return None
            values = []
            for item in spec.artifacts:
                path = (self._directory / item.relative_path).resolve()
                if not path.is_relative_to(self._directory):
                    return None
                values.append((item.key, self._hasher(path)))
            return tuple(values)
        except Exception:
            return None

    def _get(self, module, method):
        with self._locks[module]:
            if module in self._bundles:
                return self._bundles[module]
            first_attempt = module not in self._attempted
            self._attempted.add(module)
            before = self._capture(module) if first_attempt else None
            # Preserve the loader's normal exceptions; never cache failed loads.
            bundle = getattr(self._owned_loader(), method)()
            after = self._capture(module) if before is not None else None
            snapshot = None
            if before is not None and before == after:
                try:
                    spec = self._manifest[module]
                    expected = tuple((item.key, item.sha256) for item in spec.artifacts)
                    if before == expected:
                        snapshot = BundleSnapshot(
                            module, spec.model_version, spec.preprocessing_version, before
                        )
                except Exception:
                    pass
            self._bundles[module] = bundle
            self._snapshots[module] = snapshot
            return bundle

    def fashion(self):
        return self._get("fashion", "fashion")

    def review(self):
        return self._get("reviews", "review")

    def purchase(self):
        return self._get("purchase", "purchase")

    def cached_bundle(self, module):
        lock = self._locks.get(module)
        if lock is None:
            return None
        with lock:
            return self._bundles.get(module)

    def snapshot_for(self, module, bundle):
        lock = self._locks.get(module)
        if lock is None:
            return None
        with lock:
            if module not in self._bundles or self._bundles[module] is not bundle:
                return None
            return self._snapshots.get(module)
