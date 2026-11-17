"""Repository tests inspect ORM objects passed to a fake session; no SQL."""

from unittest.mock import Mock

import pytest

from backend.app.db.models import FashionPrediction, PurchasePrediction, ReviewCluster
from backend.app.repositories.predictions import PredictionRepository
from backend.tests.test_persistence_contracts import (
    fashion_record, purchase_record, review_record, snapshot,
)


@pytest.mark.parametrize("module,method,builder,model", [
    ("fashion", "add_fashion", fashion_record, FashionPrediction),
    ("reviews", "add_review", review_record, ReviewCluster),
    ("purchase", "add_purchase", purchase_record, PurchasePrediction),
])
def test_exact_record_and_metadata_mapping_without_transaction_control(
    module, method, builder, model,
):
    record = builder()
    verified = snapshot(module)
    session = Mock(spec_set=["add"])
    getattr(PredictionRepository(), method)(session, record, verified)
    session.add.assert_called_once()
    row = session.add.call_args.args[0]
    assert type(row) is model
    expected = record.model_dump()
    for key, value in expected.items():
        assert getattr(row, key) == value
    assert row.model_version == verified.model_version
    assert row.preprocessing_version == verified.preprocessing_version
    assert row.artifact_sha256 == dict(verified.artifact_sha256)
    assert row.stored_at is None
    if module == "purchase":
        assert row.svm_probabilities is None
    fields = {key for key in vars(row) if key != "_sa_instance_state"}
    assert fields == (
        set(expected) | {"model_version", "preprocessing_version", "artifact_sha256"}
        | ({"svm_probabilities"} if module == "purchase" else set())
    )
    row.artifact_sha256.clear()
    assert verified.artifact_sha256  # Repository copied the immutable snapshot.


@pytest.mark.parametrize("method,builder,module", [
    ("add_fashion", fashion_record, "fashion"),
    ("add_review", review_record, "reviews"),
    ("add_purchase", purchase_record, "purchase"),
])
def test_wrong_snapshot_or_record_is_rejected_before_add(method, builder, module):
    session = Mock(spec_set=["add"])
    wrong = "reviews" if module != "reviews" else "fashion"
    with pytest.raises(ValueError):
        getattr(PredictionRepository(), method)(session, builder(), snapshot(wrong))
    with pytest.raises(TypeError):
        getattr(PredictionRepository(), method)(session, object(), snapshot(module))
    session.add.assert_not_called()
