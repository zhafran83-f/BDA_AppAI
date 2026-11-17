"""ORM/contract structure tests without a database or model loading."""

from uuid import UUID

import pytest
from sqlalchemy.dialects.postgresql import dialect
from sqlalchemy.schema import CreateIndex, CreateTable

from backend.app.db.base import Base
from backend.app.db.models import (
    FashionPrediction, PurchasePrediction, ReviewCluster,
)
from backend.app.preprocessing.reviews import clean_text
from backend.app.schemas.responses import (
    ClassPrediction, FashionResponse, ProbabilisticPrediction,
)

MODELS = (FashionPrediction, ReviewCluster, PurchasePrediction)
COMMON = {
    "id", "predicted_at", "stored_at", "inference_duration_ms",
    "model_version", "preprocessing_version", "artifact_sha256",
}


def ddl(model):
    return str(CreateTable(model.__table__).compile(dialect=dialect()))


def test_only_three_public_tables():
    assert set(Base.metadata.tables) == {
        "public.fashion_predictions", "public.review_clusters",
        "public.purchase_predictions",
    }


@pytest.mark.parametrize("model", MODELS)
def test_uuid_timezone_nullability_and_index(model):
    table = model.__table__
    assert table.schema == "public"
    assert list(table.primary_key.columns.keys()) == ["id"]
    assert table.c.id.default.is_callable
    assert table.c.id.server_default is None
    first, second = table.c.id.default.arg(None), table.c.id.default.arg(None)
    assert isinstance(first, UUID) and first.version == 4 and first != second
    for name in ("predicted_at", "stored_at"):
        assert table.c[name].type.timezone is True
    assert table.c.predicted_at.default is None
    assert str(table.c.stored_at.server_default.arg) == "now()"
    for column in table.c:
        assert column.nullable is (column.name == "svm_probabilities")
    assert not table.foreign_keys
    assert len(table.indexes) == 1
    index = next(iter(table.indexes))
    sql = str(CreateIndex(index).compile(dialect=dialect()))
    assert sql == (
        f"CREATE INDEX ix_{table.name}_history ON public.{table.name} "
        "(predicted_at DESC, id DESC)"
    )


@pytest.mark.parametrize("model,specific", [
    (FashionPrediction, {
        "class_index", "label", "p_apparel", "p_accessories", "p_footwear",
        "p_personal_care", "image_size_bytes", "image_format",
    }),
    (ReviewCluster, {
        "cluster", "input_character_count", "cleaned_character_count",
    }),
    (PurchasePrediction, {
        "rf_revenue_class", "svm_revenue_class", "mlp_revenue_class",
        "rf_probability_0", "rf_probability_1",
        "mlp_probability_0", "mlp_probability_1", "svm_probabilities",
    }),
])
def test_exact_columns_prevent_raw_input_storage(model, specific):
    assert set(model.__table__.c.keys()) == COMMON | specific


@pytest.mark.parametrize("model,keys", [
    (FashionPrediction, ("vgg16",)),
    (ReviewCluster, ("tfidf", "kmeans")),
    (PurchasePrediction, ("random_forest", "svm", "mlp", "feature_columns")),
])
def test_common_constraints_and_exact_checksum_keys(model, keys):
    sql = ddl(model)
    assert "inference_duration_ms >= 0" in sql
    assert "btrim(model_version) <> ''" in sql
    assert "btrim(preprocessing_version) <> ''" in sql
    assert "jsonb_typeof(artifact_sha256) = 'object'" in sql
    assert "artifact_sha256 ?& ARRAY[" in sql
    assert "= '{}'::jsonb" in sql
    assert ") IS TRUE" in sql
    for key in keys:
        assert f"jsonb_typeof(artifact_sha256 -> '{key}') = 'string'" in sql
        assert f"(artifact_sha256 ->> '{key}') ~ '^[0-9a-f]{{64}}$'" in sql


def test_fashion_contract_constraints():
    labels = ("Apparel", "Accessories", "Footwear", "Personal Care")
    sql = ddl(FashionPrediction)
    probabilities = dict(zip(labels, (0.1, 0.2, 0.3, 0.4)))
    for index, label in enumerate(labels):
        FashionResponse(class_index=index, label=label, probabilities=probabilities)
        assert f"class_index = {index} AND label = '{label}'" in sql
    assert "image_size_bytes BETWEEN 1 AND 5242880" in sql
    assert "image_format IN ('JPEG', 'PNG')" in sql
    for name in ("p_apparel", "p_accessories", "p_footwear", "p_personal_care"):
        assert f"{name} BETWEEN 0 AND 1" in sql


def test_review_unicode_lengths_are_not_incorrectly_bounded():
    original = "a\u0130b"
    assert len(clean_text(original)) > len(original)
    sql = ddl(ReviewCluster)
    assert "cluster IN (0, 1)" in sql
    assert "input_character_count BETWEEN 1 AND 10000" in sql
    assert "cleaned_character_count > 0" in sql
    assert "cleaned_character_count <=" not in sql


def test_purchase_contract_and_sql_null():
    sql = ddl(PurchasePrediction)
    for name in ("rf", "svm", "mlp"):
        assert f"{name}_revenue_class IN (0, 1)" in sql
    assert "svm_probabilities IS NULL" in sql
    column = PurchasePrediction.__table__.c.svm_probabilities
    assert column.nullable and column.type.none_as_null is True
    ClassPrediction(revenue_class=0, probabilities=None)
    with pytest.raises(ValueError):
        ClassPrediction(revenue_class=0, probabilities={"0": 0.5, "1": 0.5})


@pytest.mark.parametrize("total", [0.99998, 0.999995, 1.0, 1.000005, 1.00002])
def test_probability_sum_tolerance_matches_response_schema(total):
    values = {"0": total / 2, "1": total / 2}
    database_accepts = abs(total - 1.0) <= 0.00001 * max(abs(total), 1.0)
    try:
        ProbabilisticPrediction(revenue_class=0, probabilities=values)
    except ValueError:
        assert not database_accepts
    else:
        assert database_accepts
    for model in (FashionPrediction, PurchasePrediction):
        assert "0.00001 * greatest(" in ddl(model)
