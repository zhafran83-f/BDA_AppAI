from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image
from scipy.sparse import csr_matrix

from backend.app.core import model_loader
from backend.app.core.model_loader import (
    FashionBundle, ModelLoader, PurchaseBundle, ReviewBundle,
)
from backend.app.preprocessing.shopper_features import EXPECTED_COLUMNS
from backend.app.services.fashion_service import FashionService
from backend.app.services.purchase_service import PurchaseService
from backend.app.services.review_service import ReviewService


def rgb_image():
    stream = BytesIO()
    Image.new("RGB", (224, 224), (128, 64, 192)).save(stream, format="PNG")
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


def fashion_model(output=None):
    return SimpleNamespace(
        input_shape=(None, 224, 224, 3), output_shape=(None, 4),
        inputs=[SimpleNamespace(dtype="float32")],
        predict=Mock(return_value=np.array(
            [[0.1, 0.2, 0.6, 0.1]] if output is None else output
        )),
    )


def test_fashion_result_and_input_contract():
    model = fashion_model()
    service = FashionService(SimpleNamespace(fashion=lambda: FashionBundle(model)))
    result = service.classify(rgb_image())
    assert result["class_index"] == 2
    assert result["label"] == "Footwear"
    assert tuple(result["probabilities"]) == (
        "Apparel", "Accessories", "Footwear", "Personal Care"
    )
    assert sum(result["probabilities"].values()) == pytest.approx(1)
    args, kwargs = model.predict.call_args
    assert args[0].shape == (1, 224, 224, 3)
    assert args[0].dtype == np.float32
    assert kwargs == {"verbose": 0}


@pytest.mark.parametrize("output", [
    [[0.1, 0.2, 0.3, 0.1]],
    [[float("nan"), 0, 0, 1]],
    [[-0.1, 0.2, 0.3, 0.6]],
    [[0, 0, 0, 1.1]],
    [[0.5, 0.5]],
])
def test_fashion_rejects_invalid_model_outputs(output):
    model = fashion_model(output)
    with pytest.raises(RuntimeError, match="probability"):
        FashionService(SimpleNamespace(fashion=lambda: FashionBundle(model))).classify(
            rgb_image()
        )


def test_invalid_inputs_do_not_load_models():
    loader = SimpleNamespace(fashion=Mock(), review=Mock(), purchase=Mock())
    with pytest.raises(ValueError):
        FashionService(loader).classify(b"broken")
    with pytest.raises(ValueError):
        ReviewService(loader).cluster("the and it")
    with pytest.raises(ValueError):
        PurchaseService(loader).predict({})
    loader.fashion.assert_not_called()
    loader.review.assert_not_called()
    loader.purchase.assert_not_called()


def review_bundle(matrix=None, prediction=None):
    matrix = csr_matrix([[0.4, 0.0, 0.6]]) if matrix is None else matrix
    vectorizer = SimpleNamespace(transform=Mock(return_value=matrix))
    kmeans = SimpleNamespace(
        n_features_in_=3,
        predict=Mock(return_value=np.array([0] if prediction is None else prediction)),
    )
    return ReviewBundle(vectorizer, kmeans)


def test_review_uses_transform_and_passes_sparse_features_without_sentiment():
    bundle = review_bundle()
    result = ReviewService(SimpleNamespace(review=lambda: bundle)).cluster(
        "The product quality is excellent and I love it."
    )
    bundle.vectorizer.transform.assert_called_once_with(
        ["product quality excellent love"]
    )
    assert bundle.kmeans.predict.call_args.args[0] is bundle.vectorizer.transform.return_value
    assert result == {"cleaned_text": "product quality excellent love", "cluster": 0}


def test_review_rejects_out_of_vocabulary_text_before_prediction():
    bundle = review_bundle(csr_matrix((1, 3)))
    with pytest.raises(ValueError, match="vocabulary"):
        ReviewService(SimpleNamespace(review=lambda: bundle)).cluster("zzzzzz")
    bundle.kmeans.predict.assert_not_called()


@pytest.mark.parametrize("matrix", [
    csr_matrix([[1.0, 0.0]]),
    csr_matrix([[float("nan"), 0, 0]]),
])
def test_review_rejects_bad_vectorization(matrix):
    bundle = review_bundle(matrix)
    with pytest.raises(RuntimeError):
        ReviewService(SimpleNamespace(review=lambda: bundle)).cluster("product")
    bundle.kmeans.predict.assert_not_called()


def test_review_rejects_unexpected_cluster():
    bundle = review_bundle(prediction=[2])
    with pytest.raises(RuntimeError, match="cluster"):
        ReviewService(SimpleNamespace(review=lambda: bundle)).cluster("product")


def purchase_bundle():
    models = {}
    for name in ("random_forest", "svm", "mlp"):
        models[name] = SimpleNamespace(
            n_features_in_=28, feature_names_in_=np.array(EXPECTED_COLUMNS),
            classes_=np.array([0, 1]), predict=Mock(return_value=np.array([1])),
            predict_proba=Mock(return_value=np.array([[0.25, 0.75]])),
        )
    models["svm"].predict_proba.side_effect = AssertionError(
        "SVM probabilities must never be requested"
    )
    return PurchaseBundle(models, EXPECTED_COLUMNS)


def test_purchase_results_pipeline_input_and_no_svm_probability():
    bundle = purchase_bundle()
    result = PurchaseService(SimpleNamespace(purchase=lambda: bundle)).predict(shopper())
    assert set(result["predictions"]) == {"random_forest", "svm", "mlp"}
    for name, model in bundle.models.items():
        model.predict.assert_called_once()
        frame = model.predict.call_args.args[0]
        assert tuple(frame.columns) == EXPECTED_COLUMNS
        assert frame.loc[0, "Administrative_Duration"] == 20.0
        assert frame.loc[0, "ProductRelated_Duration"] == 180.0
        assert frame.loc[0, "Month_Nov"] == 1
        assert result["predictions"][name]["revenue_class"] == 1
        if name == "svm":
            model.predict_proba.assert_not_called()
            assert result["predictions"][name]["probabilities"] is None
        else:
            model.predict_proba.assert_called_once()
            assert result["predictions"][name]["probabilities"] == {"0": 0.25, "1": 0.75}


def test_purchase_probability_keys_follow_model_class_order():
    bundle = purchase_bundle()
    bundle.models["random_forest"].classes_ = np.array([1, 0])
    result = PurchaseService(SimpleNamespace(purchase=lambda: bundle)).predict(shopper())
    assert result["predictions"]["random_forest"]["probabilities"] == {
        "1": 0.25, "0": 0.75
    }


@pytest.mark.parametrize("output", [[[0.1, 0.1]], [[float("inf"), 0.0]], [[1.0]]])
def test_purchase_rejects_invalid_probabilities(output):
    bundle = purchase_bundle()
    bundle.models["random_forest"].predict_proba.return_value = np.array(output)
    with pytest.raises(RuntimeError, match="probability"):
        PurchaseService(SimpleNamespace(purchase=lambda: bundle)).predict(shopper())


def test_purchase_rejects_invalid_prediction():
    bundle = purchase_bundle()
    bundle.models["random_forest"].predict.return_value = np.array([9])
    with pytest.raises(RuntimeError, match="Revenue"):
        PurchaseService(SimpleNamespace(purchase=lambda: bundle)).predict(shopper())


def test_loader_is_lazy_thread_safe_and_relative_to_source(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    loader = ModelLoader()
    assert loader.model_directory == (
        Path(model_loader.__file__).resolve().parents[2] / "models"
    )
    model = fashion_model()
    load = Mock(return_value=model)
    monkeypatch.setattr(model_loader, "_load_keras", load)
    monkeypatch.setattr(loader, "_artifact", lambda journal, name: tmp_path / name)
    load.assert_not_called()
    with ThreadPoolExecutor(max_workers=4) as pool:
        bundles = list(pool.map(lambda _: loader.fashion(), range(8)))
    assert all(bundle is bundles[0] for bundle in bundles)
    load.assert_called_once()


def test_keras_loading_options_are_safe(monkeypatch):
    import sys
    load = Mock(return_value="sentinel")
    monkeypatch.setitem(sys.modules, "keras", SimpleNamespace(
        models=SimpleNamespace(load_model=load)
    ))
    assert model_loader._load_keras(model_loader.MODEL_DIRECTORY / "model.keras") == "sentinel"
    assert load.call_args.kwargs == {"compile": False, "safe_mode": True}


def test_missing_artifact_is_reported_without_loading(monkeypatch, tmp_path):
    loader = ModelLoader(tmp_path)
    load = Mock()
    monkeypatch.setattr(model_loader, "_load_keras", load)
    with pytest.raises(FileNotFoundError, match="jurnal1"):
        loader.fashion()
    load.assert_not_called()


def test_failed_model_load_does_not_poison_cache(monkeypatch, tmp_path):
    loader = ModelLoader(tmp_path)
    bad = fashion_model()
    bad.output_shape = (None, 8)
    load = Mock(side_effect=[bad, fashion_model()])
    monkeypatch.setattr(model_loader, "_load_keras", load)
    monkeypatch.setattr(loader, "_artifact", lambda journal, name: tmp_path / name)
    with pytest.raises(RuntimeError, match="four"):
        loader.fashion()
    assert loader.fashion().model.output_shape == (None, 4)
    assert load.call_count == 2


def test_review_bundle_is_loaded_once_and_dimensions_checked(monkeypatch, tmp_path):
    loader = ModelLoader(tmp_path)
    vectorizer = SimpleNamespace(vocabulary_={str(i): i for i in range(5000)},
                                 idf_=np.ones(5000))
    kmeans = SimpleNamespace(n_clusters=2, n_features_in_=5000,
                            cluster_centers_=np.zeros((2, 5000)))
    load = Mock(side_effect=[vectorizer, kmeans])
    monkeypatch.setattr(model_loader, "_load_joblib", load)
    monkeypatch.setattr(loader, "_artifact", lambda journal, name: tmp_path / name)
    assert loader.review() is loader.review()
    assert load.call_count == 2


def test_purchase_bundle_loaded_once_and_bad_feature_order_rejected(monkeypatch, tmp_path):
    loader = ModelLoader(tmp_path)
    models = purchase_bundle().models
    artifacts = {
        "feature_columns_jurnal3.joblib": list(EXPECTED_COLUMNS),
        "random_forest_jurnal3.joblib": models["random_forest"],
        "svm_jurnal3.joblib": models["svm"],
        "mlp_jurnal3.joblib": models["mlp"],
    }
    load = Mock(side_effect=lambda path: artifacts[path.name])
    monkeypatch.setattr(model_loader, "_load_joblib", load)
    monkeypatch.setattr(loader, "_artifact", lambda journal, name: tmp_path / name)
    assert loader.purchase() is loader.purchase()
    assert load.call_count == 4
    models["svm"].feature_names_in_ = np.array(tuple(reversed(EXPECTED_COLUMNS)))
    fresh = ModelLoader(tmp_path)
    monkeypatch.setattr(fresh, "_artifact", lambda journal, name: tmp_path / name)
    with pytest.raises(RuntimeError, match="feature order"):
        fresh.purchase()
