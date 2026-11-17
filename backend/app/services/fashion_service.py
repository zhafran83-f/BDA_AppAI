"""Fashion classification using the verified four-class mapping."""
import numpy as np

from ..core.model_loader import DEFAULT_LOADER, ModelLoader
from ..preprocessing.images import preprocess_image
from ._validation import probability_row

FASHION_LABELS = ("Apparel", "Accessories", "Footwear", "Personal Care")


class FashionService:
    def __init__(self, loader: ModelLoader | None = None):
        self.loader = DEFAULT_LOADER if loader is None else loader

    def classify(self, image: bytes) -> dict:
        batch = preprocess_image(image)
        bundle = self.loader.fashion()
        with bundle.lock:
            output = bundle.model.predict(batch, verbose=0)
        probabilities = probability_row(output, len(FASHION_LABELS))
        index = int(np.argmax(probabilities))
        return {
            "class_index": index,
            "label": FASHION_LABELS[index],
            "probabilities": {
                label: float(probabilities[i]) for i, label in enumerate(FASHION_LABELS)
            },
        }
