"""Review clustering without an unverified sentiment interpretation."""
import numpy as np

from ..core.model_loader import DEFAULT_LOADER, ModelLoader
from ..preprocessing.reviews import preprocess_review


class ReviewService:
    def __init__(self, loader: ModelLoader | None = None):
        self.loader = DEFAULT_LOADER if loader is None else loader

    def cluster(self, text: str) -> dict:
        cleaned = preprocess_review(text)
        bundle = self.loader.review()
        with bundle.lock:
            features = bundle.vectorizer.transform([cleaned])
            if features.shape != (1, bundle.kmeans.n_features_in_):
                raise RuntimeError("TF-IDF and K-Means feature dimensions differ.")
            if not np.isfinite(features.data).all():
                raise RuntimeError("TF-IDF returned non-finite features.")
            if features.nnz == 0:
                raise ValueError("Review has no terms in the model vocabulary.")
            prediction = np.asarray(bundle.kmeans.predict(features))
        if (
            prediction.shape != (1,)
            or not np.issubdtype(prediction.dtype, np.integer)
            or prediction[0] not in (0, 1)
        ):
            raise RuntimeError("K-Means returned an invalid cluster.")
        return {"cleaned_text": cleaned, "cluster": int(prediction[0])}
