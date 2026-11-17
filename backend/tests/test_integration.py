"""Opt-in real-artifact smoke tests. Default pytest runs do not load AI models.

Run explicitly from the repository root:
    $env:BDA_RUN_MODEL_TESTS = "1"
    backend/e-commerce/Scripts/python.exe -B -m pytest backend/tests/test_integration.py -q
These tests perform inference only and never fit or save artifacts.
"""
from io import BytesIO
import os

import pytest
from PIL import Image

from backend.app.core.model_loader import ModelLoader
from backend.app.services.fashion_service import FashionService
from backend.app.services.review_service import ReviewService
from backend.app.services.purchase_service import PurchaseService

pytestmark = pytest.mark.skipif(
    os.environ.get("BDA_RUN_MODEL_TESTS") != "1",
    reason="Real model loading/inference requires explicit BDA_RUN_MODEL_TESTS=1.",
)


@pytest.fixture(scope="module")
def loader(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("KERAS_HOME", str(tmp_path_factory.mktemp("keras_home")))
        patch.setenv("KERAS_BACKEND", "tensorflow")
        yield ModelLoader()


def test_real_fashion(loader):
    stream = BytesIO()
    Image.new("RGB", (224, 224), (128, 64, 192)).save(stream, format="PNG")
    result = FashionService(loader).classify(stream.getvalue())
    assert result["label"] in ("Apparel", "Accessories", "Footwear", "Personal Care")
    assert sum(result["probabilities"].values()) == pytest.approx(1, abs=1e-5)


def test_real_review(loader):
    result = ReviewService(loader).cluster(
        "The product quality is excellent and I love it."
    )
    assert result["cleaned_text"] == "product quality excellent love"
    assert result["cluster"] in (0, 1)


def test_real_purchase(loader):
    data = {
        "Administrative": 1, "Administrative_Duration": 20.0,
        "Informational": 0, "Informational_Duration": 0.0,
        "ProductRelated": 8, "ProductRelated_Duration": 180.0,
        "BounceRates": 0.02, "ExitRates": 0.04, "PageValues": 10.0,
        "SpecialDay": 0.0, "OperatingSystems": 2, "Browser": 2,
        "Region": 1, "TrafficType": 2, "Weekend": False,
        "Month": "Nov", "VisitorType": "Returning_Visitor",
    }
    results = PurchaseService(loader).predict(data)["predictions"]
    assert set(results) == {"random_forest", "svm", "mlp"}
    assert results["svm"]["probabilities"] is None
    for name in ("random_forest", "mlp"):
        assert results[name]["revenue_class"] in (0, 1)
        assert sum(results[name]["probabilities"].values()) == pytest.approx(1)
