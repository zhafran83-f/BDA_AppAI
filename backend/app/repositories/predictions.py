"""Explicit field mapping only. No commit, flush, query, retry or HTTP logic."""

from ..core.model_metadata import BundleSnapshot
from ..db.models import FashionPrediction, PurchasePrediction, ReviewCluster
from ..persistence.contracts import FashionRecord, PurchaseRecord, ReviewRecord


def _common(record, snapshot, module):
    if not isinstance(snapshot, BundleSnapshot) or snapshot.module != module:
        raise ValueError("Model metadata does not match the record.")
    return {
        "id": record.id,
        "predicted_at": record.predicted_at,
        "inference_duration_ms": record.inference_duration_ms,
        "model_version": snapshot.model_version,
        "preprocessing_version": snapshot.preprocessing_version,
        "artifact_sha256": dict(snapshot.artifact_sha256),
    }


class PredictionRepository:
    def add_fashion(self, session, record: FashionRecord, snapshot: BundleSnapshot):
        if type(record) is not FashionRecord:
            raise TypeError("Expected a fashion record.")
        row = FashionPrediction(
            **_common(record, snapshot, "fashion"),
            class_index=record.class_index, label=record.label,
            p_apparel=record.p_apparel, p_accessories=record.p_accessories,
            p_footwear=record.p_footwear, p_personal_care=record.p_personal_care,
            image_size_bytes=record.image_size_bytes, image_format=record.image_format,
        )
        session.add(row)

    def add_review(self, session, record: ReviewRecord, snapshot: BundleSnapshot):
        if type(record) is not ReviewRecord:
            raise TypeError("Expected a review record.")
        row = ReviewCluster(
            **_common(record, snapshot, "reviews"),
            cluster=record.cluster,
            input_character_count=record.input_character_count,
            cleaned_character_count=record.cleaned_character_count,
        )
        session.add(row)

    def add_purchase(self, session, record: PurchaseRecord, snapshot: BundleSnapshot):
        if type(record) is not PurchaseRecord:
            raise TypeError("Expected a purchase record.")
        row = PurchasePrediction(
            **_common(record, snapshot, "purchase"),
            rf_revenue_class=record.rf_revenue_class,
            svm_revenue_class=record.svm_revenue_class,
            mlp_revenue_class=record.mlp_revenue_class,
            rf_probability_0=record.rf_probability_0,
            rf_probability_1=record.rf_probability_1,
            mlp_probability_0=record.mlp_probability_0,
            mlp_probability_1=record.mlp_probability_1,
            svm_probabilities=None,
        )
        session.add(row)
