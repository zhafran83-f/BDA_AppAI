"""Revenue class predictions; preserve saved pipelines and class ordering."""
from collections.abc import Mapping

import numpy as np

from ..core.model_loader import DEFAULT_LOADER, ModelLoader
from ..preprocessing.shopper_features import encode_shopper, EXPECTED_COLUMNS
from ._validation import probability_row


class PurchaseService:
    def __init__(self, loader: ModelLoader | None = None):
        self.loader = DEFAULT_LOADER if loader is None else loader

    def predict(self, data: Mapping[str, object]) -> dict:
        # Reject invalid requests before loading the model bundle.
        encode_shopper(data, EXPECTED_COLUMNS)
        bundle = self.loader.purchase()
        features = encode_shopper(data, bundle.feature_columns)
        results = {}
        with bundle.lock:
            for name in ("random_forest", "svm", "mlp"):
                model = bundle.models[name]
                prediction = np.asarray(model.predict(features))
                if (
                    prediction.shape != (1,)
                    or not np.issubdtype(prediction.dtype, np.integer)
                    or prediction[0] not in (0, 1)
                ):
                    raise RuntimeError(f"{name} returned an invalid Revenue class.")
                probabilities = None
                if name != "svm":
                    classes = np.asarray(model.classes_)
                    if classes.shape != (2,) or set(classes) != {0, 1}:
                        raise RuntimeError(f"{name} returned invalid target labels.")
                    values = probability_row(model.predict_proba(features), 2)
                    probabilities = {
                        str(int(label)): float(values[i])
                        for i, label in enumerate(classes)
                    }
                results[name] = {
                    "revenue_class": int(prediction[0]),
                    "probabilities": probabilities,
                }
        return {"predictions": results}
