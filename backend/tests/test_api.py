"""API tests use the real services with fake in-memory model bundles."""
import asyncio
from collections import Counter
from io import BytesIO
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from scipy.sparse import csr_matrix

from backend.app.core.model_loader import FashionBundle, PurchaseBundle, ReviewBundle
from backend.app.main import create_app
from backend.app.preprocessing.shopper_features import EXPECTED_COLUMNS


def rgb_image(size=(224, 224), mode="RGB", format="PNG"):
    stream = BytesIO()
    color = 128 if mode == "L" else (128, 64, 192)
    Image.new(mode, size, color).save(stream, format=format)
    return stream.getvalue()


def shopper():
    return {
        "Administrative": 1, "Administrative_Duration": 20.0,
        "Informational": 0, "Informational_Duration": 0.0,
        "ProductRelated": 8, "ProductRelated_Duration": 180.0,
        "BounceRates": 0.02, "ExitRates": 0.04, "PageValues": 10.0,
        "SpecialDay": 0.0, "OperatingSystems": 2, "Browser": 2,
        "Region": 1, "TrafficType": 2, "Weekend": False,
        "Month": "Nov", "VisitorType": "Returning_Visitor",
    }


class FakeFashion:
    def __init__(self):
        self.calls = 0
        self.thread_id = None
        self.error = None

    def predict(self, batch, verbose=0):
        self.calls += 1
        self.thread_id = threading.get_ident()
        if self.error:
            raise self.error
        assert batch.shape == (1, 224, 224, 3)
        assert batch.dtype == np.float32
        assert verbose == 0
        return np.array([[0.1, 0.2, 0.3, 0.4]])


class FakeLoader:
    def __init__(self):
        self.calls = Counter()
        self.failures = {}
        self.load_threads = []
        self.fashion_model = FakeFashion()
        self.fashion_bundle = FashionBundle(self.fashion_model)
        vocabulary = ("product", "quality", "excellent", "love")
        self.vectorizer = SimpleNamespace(
            transform=Mock(side_effect=lambda texts: csr_matrix([
                [float(word in text.split()) for word in vocabulary] for text in texts
            ]))
        )
        self.kmeans = SimpleNamespace(
            n_features_in_=4, predict=Mock(return_value=np.array([0]))
        )
        self.review_bundle = ReviewBundle(self.vectorizer, self.kmeans)
        models = {}
        for name in ("random_forest", "svm", "mlp"):
            models[name] = SimpleNamespace(
                classes_=np.array([0, 1]),
                predict=Mock(return_value=np.array([1 if name == "svm" else 0])),
                predict_proba=Mock(return_value=np.array([[0.7, 0.3]])),
            )
        models["svm"].predict_proba.side_effect = AssertionError(
            "SVM probabilities must not be requested"
        )
        self.purchase_bundle = PurchaseBundle(models, EXPECTED_COLUMNS)

    def _load(self, name, bundle):
        self.calls[name] += 1
        self.load_threads.append(threading.get_ident())
        if name in self.failures:
            raise self.failures[name]
        return bundle

    def fashion(self):
        return self._load("fashion", self.fashion_bundle)

    def review(self):
        return self._load("reviews", self.review_bundle)

    def purchase(self):
        return self._load("purchase", self.purchase_bundle)


@pytest.fixture
def api():
    loader = FakeLoader()
    app = create_app(loader_factory=lambda: loader)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, loader, app


def assert_error(response, status, code=None):
    assert response.status_code == status, response.text
    payload = response.json()
    assert set(payload) == {"error"}
    assert set(payload["error"]) == {"code", "message", "details"}
    if code:
        assert payload["error"]["code"] == code
    assert "Traceback" not in response.text
    return payload["error"]


def post_fashion(client, data=None, mime="image/png"):
    return client.post(
        "/api/v1/fashion/classify",
        files={"image": ("input.png", rgb_image() if data is None else data, mime)},
    )


def test_lifespan_warms_small_bundles_and_keeps_vgg_lazy(api):
    client, loader, app = api
    assert client.get("/health").json() == {"status": "ok"}
    assert loader.calls == {"reviews": 1, "purchase": 1}
    before = assert_error(client.get("/ready"), 503, "not_ready")
    assert before["details"]["models"] == {
        "fashion": "not_loaded", "reviews": "ready", "purchase": "ready"
    }
    assert loader.calls["fashion"] == 0
    result = post_fashion(client)
    assert result.status_code == 200, result.text
    assert result.json()["label"] == "Personal Care"
    assert result.json()["class_index"] == 3
    assert set(result.json()["probabilities"]) == {
        "Apparel", "Accessories", "Footwear", "Personal Care"
    }
    assert sum(result.json()["probabilities"].values()) == pytest.approx(1)
    assert client.get("/ready").json() == {
        "status": "ready",
        "models": {"fashion": "ready", "reviews": "ready", "purchase": "ready"},
    }


def test_bundles_reused_across_requests(api):
    client, loader, _ = api
    for _ in range(2):
        assert post_fashion(client).status_code == 200
        assert client.post("/api/v1/reviews/cluster", json={"text": "quality product"}).status_code == 200
        assert client.post("/api/v1/purchase/predict", json=shopper()).status_code == 200
    assert loader.calls == {"fashion": 1, "reviews": 1, "purchase": 1}


def test_lifespan_cleanup_and_no_loading_at_app_construction():
    loader = FakeLoader()
    app = create_app(loader_factory=lambda: loader)
    assert loader.calls == {}
    assert app.state.runtime is None
    with TestClient(app):
        runtime = app.state.runtime
        assert runtime.active
    assert app.state.runtime is None
    assert not runtime.active
    assert runtime.services == {}


def test_ready_without_lifespan_is_unavailable():
    app = create_app(loader_factory=FakeLoader)
    client = TestClient(app, raise_server_exceptions=False)
    try:
        assert_error(client.get("/ready"), 503, "model_unavailable")
    finally:
        client.close()


@pytest.mark.parametrize("endpoint", [
    "/api/v1/reviews/cluster", "/api/v1/purchase/predict"
])
def test_extra_json_fields_are_not_ignored(api, endpoint):
    client, _, _ = api
    body = {"text": "product"} if "reviews" in endpoint else shopper()
    body["unexpected"] = "sensitive-value-not-to-echo"
    response = client.post(endpoint, json=body)
    error = assert_error(response, 422, "validation_error")
    assert error["details"]
    assert "sensitive-value-not-to-echo" not in response.text


def test_review_response_and_original_cleaning(api):
    client, loader, _ = api
    response = client.post("/api/v1/reviews/cluster", json={
        "text": "The product quality is excellent and I love it."
    })
    assert response.status_code == 200, response.text
    assert response.json() == {
        "cleaned_text": "product quality excellent love", "cluster": 0
    }
    loader.vectorizer.transform.assert_called_once_with(
        ["product quality excellent love"]
    )


@pytest.mark.parametrize("text", ["", "the and it", "zzzzzzz", "x" * 10_001, 123, None])
def test_invalid_review(api, text):
    client, _, _ = api
    assert_error(
        client.post("/api/v1/reviews/cluster", json={"text": text}),
        422, "validation_error",
    )


def test_review_maximum_length_is_accepted(api):
    client, _, _ = api
    response = client.post("/api/v1/reviews/cluster", json={"text": "product " * 1250})
    assert response.status_code == 200, response.text


def test_invalid_json_uses_safe_error_format(api):
    client, _, _ = api
    response = client.post(
        "/api/v1/reviews/cluster", content="{",
        headers={"content-type": "application/json"},
    )
    assert_error(response, 422, "validation_error")


@pytest.mark.parametrize(("data", "mime"), [
    (b"not an image", "image/png"),
    (b"", "image/png"),
    (rgb_image(size=(225, 224)), "image/png"),
    (rgb_image(mode="L"), "image/png"),
    (rgb_image(format="BMP"), "image/png"),
    (rgb_image(), "application/pdf"),
])
def test_invalid_fashion_upload(api, data, mime):
    client, loader, _ = api
    error = assert_error(post_fashion(client, data, mime), 422, "validation_error")
    assert error["message"]
    assert loader.calls["fashion"] == 0


def test_jpeg_upload(api):
    client, _, _ = api
    assert post_fashion(client, rgb_image(format="JPEG"), "image/jpeg").status_code == 200


def test_oversize_file_is_413_before_loading(api):
    client, loader, _ = api
    assert_error(
        post_fashion(client, b"x" * (5 * 1024 * 1024 + 1)), 413, "file_too_large"
    )
    assert loader.calls["fashion"] == 0


def test_missing_or_wrong_upload_field(api):
    client, _, _ = api
    assert_error(client.post("/api/v1/fashion/classify"), 422, "validation_error")
    response = client.post(
        "/api/v1/fashion/classify",
        files={"photo": ("test.png", rgb_image(), "image/png")},
    )
    assert_error(response, 422, "validation_error")


def test_extra_and_duplicate_multipart_fields_are_rejected(api):
    client, loader, _ = api
    extra = client.post(
        "/api/v1/fashion/classify",
        files={"image": ("test.png", rgb_image(), "image/png")},
        data={"unexpected": "value"},
    )
    assert_error(extra, 422, "validation_error")
    duplicate = client.post("/api/v1/fashion/classify", files=[
        ("image", ("one.png", rgb_image(), "image/png")),
        ("image", ("two.png", rgb_image(), "image/png")),
    ])
    assert_error(duplicate, 422, "validation_error")
    assert loader.calls["fashion"] == 0


def test_purchase_uses_raw_fields_without_extra_scaling(api):
    client, loader, _ = api
    response = client.post("/api/v1/purchase/predict", json=shopper())
    assert response.status_code == 200, response.text
    results = response.json()["predictions"]
    assert set(results) == {"random_forest", "svm", "mlp"}
    assert results["svm"] == {"revenue_class": 1, "probabilities": None}
    loader.purchase_bundle.models["svm"].predict_proba.assert_not_called()
    for name in ("random_forest", "mlp"):
        assert results[name] == {
            "revenue_class": 0, "probabilities": {"0": 0.7, "1": 0.3}
        }
    frame = loader.purchase_bundle.models["svm"].predict.call_args.args[0]
    assert tuple(frame.columns) == EXPECTED_COLUMNS
    assert frame.loc[0, "ProductRelated_Duration"] == 180
    assert frame.loc[0, "Month_Nov"] == 1


@pytest.mark.parametrize(("field", "value"), [
    ("Month", "Jan"), ("Month", "Jun"), ("VisitorType", "Returning Visitor"),
    ("Weekend", 1), ("Weekend", "false"),
    ("Administrative", True), ("Administrative", "1"),
    ("Administrative", 1.0), ("Administrative", -1),
    ("ProductRelated_Duration", -1), ("ProductRelated_Duration", True),
    ("PageValues", "10"), ("BounceRates", 1.01), ("SpecialDay", -1),
])
def test_purchase_strict_validation(api, field, value):
    client, loader, _ = api
    data = shopper()
    data[field] = value
    assert_error(
        client.post("/api/v1/purchase/predict", json=data), 422, "validation_error"
    )
    loader.purchase_bundle.models["random_forest"].predict.assert_not_called()


def test_purchase_missing_field(api):
    client, _, _ = api
    data = shopper()
    del data["Weekend"]
    assert_error(
        client.post("/api/v1/purchase/predict", json=data), 422, "validation_error"
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_purchase_nonfinite_values(api, value):
    client, _, _ = api
    data = shopper()
    data["PageValues"] = value
    response = client.post(
        "/api/v1/purchase/predict", content=json.dumps(data),
        headers={"content-type": "application/json"},
    )
    assert_error(response, 422, "validation_error")


def test_failed_startup_model_is_503_and_does_not_leak_paths():
    loader = FakeLoader()
    secret = r"D:\private\model.joblib token=do-not-expose"
    loader.failures["reviews"] = FileNotFoundError(secret)
    app = create_app(loader_factory=lambda: loader)
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get("/health").status_code == 200
        readiness = assert_error(client.get("/ready"), 503, "not_ready")
        assert readiness["details"]["models"]["reviews"] == "unavailable"
        response = client.post("/api/v1/reviews/cluster", json={"text": "product"})
        assert_error(response, 503, "model_unavailable")
        assert secret not in response.text


def test_lazy_fashion_load_failure_is_503():
    loader = FakeLoader()
    loader.failures["fashion"] = RuntimeError(r"D:\secret\custom_layer.py")
    app = create_app(loader_factory=lambda: loader)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = post_fashion(client)
        assert_error(response, 503, "model_unavailable")
        assert "custom_layer.py" not in response.text


def test_internal_inference_failure_is_sanitized(api):
    client, loader, _ = api
    loader.fashion_model.error = RuntimeError(r"D:\private\password=secret")
    response = post_fashion(client)
    assert_error(response, 500, "inference_error")
    assert "private" not in response.text and "secret" not in response.text
    error = assert_error(client.get("/ready"), 503, "not_ready")
    assert error["details"]["models"]["fashion"] == "unavailable"


def test_response_schema_failure_is_sanitized(api):
    client, _, app = api
    app.state.runtime.services["reviews"].cluster = Mock(return_value={
        "cleaned_text": "product", "cluster": "private-path-secret"
    })
    response = client.post("/api/v1/reviews/cluster", json={"text": "product"})
    assert_error(response, 500, "inference_error")
    assert "private-path-secret" not in response.text


def test_swagger_and_openapi_contracts(api):
    client, _, _ = api
    assert client.get("/docs").status_code == 200
    document = client.get("/openapi.json").json()
    assert set(document["paths"]) == {
        "/health", "/ready", "/api/v1/fashion/classify",
        "/api/v1/reviews/cluster", "/api/v1/purchase/predict",
    }
    purchase = document["components"]["schemas"]["PurchaseRequest"]
    assert purchase["additionalProperties"] is False
    assert len(purchase["properties"]) == 17
    assert "June" in purchase["properties"]["Month"]["enum"]
    assert "Jun" not in purchase["properties"]["Month"]["enum"]
    fashion = document["paths"]["/api/v1/fashion/classify"]["post"]
    assert "multipart/form-data" in fashion["requestBody"]["content"]
    assert {"200", "413", "422", "503"}.issubset(fashion["responses"])
    assert_error(client.get("/does-not-exist"), 404, "http_error")


def test_inference_and_startup_run_outside_event_loop_thread():
    async def scenario():
        loop_thread = threading.get_ident()
        loader = FakeLoader()
        app = create_app(loader_factory=lambda: loader)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/api/v1/fashion/classify",
                    files={"image": ("image.png", rgb_image(), "image/png")},
                )
                assert response.status_code == 200, response.text
                assert loader.fashion_model.thread_id != loop_thread
                assert all(thread != loop_thread for thread in loader.load_threads)
    asyncio.run(scenario())
