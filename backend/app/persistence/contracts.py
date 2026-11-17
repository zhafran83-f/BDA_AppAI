"""Allowlisted immutable records and sanitized persistence receipts."""

from dataclasses import dataclass
from datetime import timezone
from enum import Enum
from math import isclose
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from ..db import TransactionOutcome

Probability = Annotated[float, Field(ge=0, le=1)]
RevenueClass = Annotated[int, Field(ge=0, le=1)]
FASHION_LABELS = ("Apparel", "Accessories", "Footwear", "Personal Care")


class PersistenceStatus(str, Enum):
    DISABLED = "disabled"
    COMMITTED = "committed"
    NOT_COMMITTED = "not_committed"
    UNKNOWN = "unknown"


class PersistenceReason(str, Enum):
    DISABLED = "disabled"
    NOT_STARTED = "not_started"
    SHUTDOWN = "shutdown"
    UNAVAILABLE = "configuration_unavailable"
    METADATA_UNVERIFIED = "metadata_unverified"
    BUSY = "busy"
    SAVED = "saved"
    TRANSACTION_ERROR = "transaction_error"
    CLEANUP_FAILED = "cleanup_failed"


@dataclass(frozen=True)
class PersistenceReceipt:
    prediction_id: UUID
    status: PersistenceStatus
    reason: PersistenceReason
    cleanup_failures: tuple[str, ...] = ()
    failed_stage: str | None = None

    @property
    def outcome(self) -> TransactionOutcome | None:
        if self.status is PersistenceStatus.DISABLED:
            return None
        return TransactionOutcome(self.status.value)


class PredictionContext(BaseModel):
    model_config = ConfigDict(
        strict=True, extra="forbid", frozen=True,
        allow_inf_nan=False, hide_input_in_errors=True,
    )

    id: UUID
    predicted_at: AwareDatetime
    inference_duration_ms: Annotated[int, Field(ge=0, le=9223372036854775807)]

    @field_validator("predicted_at")
    @classmethod
    def normalize_utc(cls, value):
        return value.astimezone(timezone.utc)


def _check_sum(values):
    if not isclose(sum(values), 1.0, rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError("Probabilities must sum to one.")


class FashionRecord(PredictionContext):
    class_index: Annotated[int, Field(ge=0, le=3)]
    label: Literal["Apparel", "Accessories", "Footwear", "Personal Care"]
    p_apparel: Probability
    p_accessories: Probability
    p_footwear: Probability
    p_personal_care: Probability
    image_size_bytes: Annotated[int, Field(ge=1, le=5242880)]
    image_format: Literal["JPEG", "PNG"]

    @model_validator(mode="after")
    def validate_prediction(self):
        if self.label != FASHION_LABELS[self.class_index]:
            raise ValueError("Class index and label do not match.")
        _check_sum((self.p_apparel, self.p_accessories, self.p_footwear, self.p_personal_care))
        return self


class ReviewRecord(PredictionContext):
    cluster: RevenueClass
    input_character_count: Annotated[int, Field(ge=1, le=10000)]
    cleaned_character_count: Annotated[int, Field(ge=1, le=2147483647)]


class PurchaseRecord(PredictionContext):
    rf_revenue_class: RevenueClass
    svm_revenue_class: RevenueClass
    mlp_revenue_class: RevenueClass
    rf_probability_0: Probability
    rf_probability_1: Probability
    mlp_probability_0: Probability
    mlp_probability_1: Probability

    @model_validator(mode="after")
    def validate_distributions(self):
        _check_sum((self.rf_probability_0, self.rf_probability_1))
        _check_sum((self.mlp_probability_0, self.mlp_probability_1))
        return self
