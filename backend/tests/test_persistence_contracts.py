"""Dummy record builders and isolated contract validation tests."""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from backend.app.core.model_metadata import ARTIFACT_KEYS, BundleSnapshot
from backend.app.persistence.contracts import (
    FashionRecord, PersistenceReason, PersistenceReceipt, PersistenceStatus,
    PurchaseRecord, ReviewRecord,
)


def context():
    return dict(id=uuid4(), predicted_at=datetime.now(timezone.utc), inference_duration_ms=12)


def fashion_record(**changes):
    data = dict(**context(), class_index=0, label="Apparel", p_apparel=0.7,
                p_accessories=0.1, p_footwear=0.1, p_personal_care=0.1,
                image_size_bytes=256, image_format="PNG")
    data.update(changes)
    return FashionRecord(**data)


def review_record(**changes):
    data = dict(**context(), cluster=0, input_character_count=20, cleaned_character_count=12)
    data.update(changes)
    return ReviewRecord(**data)


def purchase_record(**changes):
    data = dict(**context(), rf_revenue_class=0, svm_revenue_class=1, mlp_revenue_class=0,
                rf_probability_0=0.6, rf_probability_1=0.4,
                mlp_probability_0=0.7, mlp_probability_1=0.3)
    data.update(changes)
    return PurchaseRecord(**data)


def snapshot(module):
    return BundleSnapshot(
        module, "dummy-bundle-v1", "dummy-preprocessing-v1",
        tuple((key, "a" * 64) for key in ARTIFACT_KEYS[module]),
    )


@pytest.mark.parametrize("builder", [fashion_record, review_record, purchase_record])
def test_valid_records_are_immutable_and_have_only_allowed_scalars(builder):
    record = builder()
    with pytest.raises(ValidationError):
        record.inference_duration_ms = 99
    assert record.predicted_at.utcoffset() == timedelta(0)
    assert record.model_config["extra"] == "forbid"


@pytest.mark.parametrize("builder", [fashion_record, review_record, purchase_record])
@pytest.mark.parametrize("field", [
    "image", "filename", "text", "cleaned_text", "visitor_profile",
    "artifact_sha256", "password",
])
def test_forbidden_fields_cannot_enter_records(builder, field):
    with pytest.raises(ValidationError) as caught:
        builder(**{field: "dummy-sensitive-input"})
    assert "dummy-sensitive-input" not in str(caught.value)


@pytest.mark.parametrize("changes", [
    {"id": "not-a-uuid"}, {"predicted_at": datetime(2026, 1, 1)},
    {"inference_duration_ms": -1}, {"inference_duration_ms": True},
    {"inference_duration_ms": 2**63}, {"inference_duration_ms": "12"},
])
def test_invalid_common_values(changes):
    with pytest.raises(ValidationError):
        review_record(**changes)


def test_timezone_is_normalized_to_utc():
    value = datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=7)))
    assert review_record(predicted_at=value).predicted_at == value.astimezone(timezone.utc)


@pytest.mark.parametrize("changes", [
    {"class_index": 4}, {"class_index": True}, {"label": "Footwear"},
    {"image_size_bytes": 0}, {"image_size_bytes": 5242881},
    {"image_format": "GIF"}, {"p_apparel": float("nan")},
    {"p_apparel": float("inf")}, {"p_apparel": -0.1},
    {"p_apparel": 1.1}, {"p_apparel": 0.2},
])
def test_fashion_constraints(changes):
    with pytest.raises(ValidationError):
        fashion_record(**changes)


@pytest.mark.parametrize("changes", [
    {"cluster": 2}, {"cluster": True}, {"input_character_count": 0},
    {"input_character_count": 10001}, {"cleaned_character_count": 0},
    {"cleaned_character_count": 2**31},
])
def test_review_constraints(changes):
    with pytest.raises(ValidationError):
        review_record(**changes)


def test_cleaned_length_may_exceed_original():
    assert review_record(input_character_count=3, cleaned_character_count=4)


@pytest.mark.parametrize("changes", [
    {"rf_revenue_class": 2}, {"svm_revenue_class": True},
    {"mlp_revenue_class": -1}, {"rf_probability_0": float("-inf")},
    {"rf_probability_0": float("nan")}, {"rf_probability_1": 0.2},
    {"mlp_probability_0": 1.1}, {"mlp_probability_1": 0.1},
    {"svm_probabilities": None},
])
def test_purchase_constraints(changes):
    with pytest.raises(ValidationError):
        purchase_record(**changes)


def test_receipt_is_immutable_and_disabled_has_no_transaction_outcome():
    from dataclasses import FrozenInstanceError
    receipt = PersistenceReceipt(uuid4(), PersistenceStatus.DISABLED, PersistenceReason.DISABLED)
    assert receipt.outcome is None
    with pytest.raises(FrozenInstanceError):
        receipt.reason = PersistenceReason.SAVED
