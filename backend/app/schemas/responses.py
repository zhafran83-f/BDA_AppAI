from math import isclose
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..services.fashion_service import FASHION_LABELS

Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
FashionLabel = Literal[*FASHION_LABELS]
ModelStatus = Literal["not_loaded", "loading", "ready", "unavailable"]


class APIResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class ValidationIssue(APIResponse):
    location: list[str | int]
    message: str
    type: str


class ErrorBody(APIResponse):
    code: str
    message: str
    details: list[ValidationIssue] | dict[str, Any] | None = None


class ErrorResponse(APIResponse):
    error: ErrorBody


class HealthResponse(APIResponse):
    status: Literal["ok"]


class ReadyResponse(APIResponse):
    status: Literal["ready"]
    models: dict[Literal["fashion", "reviews", "purchase"], ModelStatus]


class FashionResponse(APIResponse):
    class_index: Annotated[int, Field(ge=0, le=3)]
    label: FashionLabel
    probabilities: Annotated[dict[FashionLabel, Probability], Field(min_length=4, max_length=4)]

    @model_validator(mode="after")
    def check_distribution(self):
        if self.label != FASHION_LABELS[self.class_index]:
            raise ValueError("Class index and label do not match.")
        if not isclose(sum(self.probabilities.values()), 1, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError("Probabilities must sum to one.")
        return self


class ReviewResponse(APIResponse):
    cleaned_text: str
    cluster: Literal[0, 1]


class ProbabilisticPrediction(APIResponse):
    revenue_class: Literal[0, 1]
    probabilities: Annotated[
        dict[Literal["0", "1"], Probability], Field(min_length=2, max_length=2)
    ]

    @model_validator(mode="after")
    def check_distribution(self):
        if not isclose(sum(self.probabilities.values()), 1, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError("Probabilities must sum to one.")
        return self


class ClassPrediction(APIResponse):
    revenue_class: Literal[0, 1]
    probabilities: None


class PurchasePredictions(APIResponse):
    random_forest: ProbabilisticPrediction
    svm: ClassPrediction
    mlp: ProbabilisticPrediction


class PurchaseResponse(APIResponse):
    predictions: PurchasePredictions
