from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from backend.app.preprocessing import images
from backend.app.preprocessing.images import preprocess_image
from backend.app.preprocessing.reviews import clean_text, preprocess_review
from backend.app.preprocessing.shopper_features import (
    EXPECTED_COLUMNS, MONTHS, VISITOR_TYPES, encode_shopper,
)


def image_bytes(mode="RGB", size=(224, 224), format="PNG", **kwargs):
    stream = BytesIO()
    Image.new(mode, size, 128 if mode == "L" else (255, 128, 0)).save(
        stream, format=format, **kwargs
    )
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


def test_png_preserves_rgb_normalization_and_batch():
    batch = preprocess_image(image_bytes())
    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32
    np.testing.assert_allclose(
        batch[0, 0, 0], np.array([255, 128, 0], dtype=np.float32) / 255.0
    )


def test_jpeg_is_accepted():
    assert preprocess_image(image_bytes(format="JPEG")).shape == (1, 224, 224, 3)


@pytest.mark.parametrize("data", [b"", b"not an image", None, "image.png"])
def test_invalid_image_data(data):
    with pytest.raises(ValueError):
        preprocess_image(data)


def test_size_is_rejected_without_resizing():
    with pytest.raises(ValueError, match="exactly 224x224"):
        preprocess_image(image_bytes(size=(225, 224)))


def test_rgb_is_required():
    with pytest.raises(ValueError, match="RGB"):
        preprocess_image(image_bytes(mode="L"))


def test_unsupported_image_format():
    with pytest.raises(ValueError, match="JPEG and PNG"):
        preprocess_image(image_bytes(format="BMP"))


def test_upload_limit_is_checked_before_decode(monkeypatch):
    monkeypatch.setattr(images, "MAX_IMAGE_BYTES", 8)
    with pytest.raises(ValueError, match="5 MiB"):
        preprocess_image(b"x" * 9)


def test_nonstandard_exif_orientation_is_rejected():
    exif = Image.Exif()
    exif[274] = 6
    with pytest.raises(ValueError, match="orientation"):
        preprocess_image(image_bytes(format="JPEG", exif=exif))


@pytest.mark.parametrize(("original", "expected"), [
    ("The product quality is excellent and I love it.", "product quality excellent love"),
    ("The product is terrible and very disappointing.", "product terrible disappointing"),
    ("The package arrived yesterday and contains two items.",
     "package arrived yesterday contains items"),
    ("  QUALITY!!!\n123\tProduct  ", "quality product"),
])
def test_original_colab_cleaning(original, expected):
    assert clean_text(original) == expected


@pytest.mark.parametrize("text", [None, 123, "", "   ", "the and it", "123 !!!"])
def test_review_rejects_empty_or_invalid_content(text):
    with pytest.raises(ValueError):
        preprocess_review(text)


def test_review_length_limit():
    with pytest.raises(ValueError, match="10000"):
        preprocess_review("x" * 10_001)


def test_shopper_encoding_has_exact_order_values_and_types():
    data = shopper()
    frame = encode_shopper(data, EXPECTED_COLUMNS)
    assert frame.shape == (1, 28)
    assert tuple(frame.columns) == EXPECTED_COLUMNS
    assert frame["Administrative"].dtype == np.dtype("int64")
    assert frame["Administrative_Duration"].dtype == np.dtype("float64")
    assert frame.loc[0, "Administrative_Duration"] == 20.0
    assert frame.loc[0, "Weekend"] == 0
    assert frame.loc[0, "Month_Nov"] == 1
    assert frame.loc[0, "VisitorType_Returning_Visitor"] == 1
    assert frame.filter(like="Month_").sum(axis=1).item() == 1
    assert frame.filter(like="VisitorType_").sum(axis=1).item() == 1
    assert "Month" in data and "Month_Nov" not in data


def test_artifact_column_order_is_preserved():
    order = tuple(reversed(EXPECTED_COLUMNS))
    frame = encode_shopper(shopper(), order)
    assert tuple(frame.columns) == order
    assert frame.loc[0, "Month_Nov"] == 1


@pytest.mark.parametrize("month", MONTHS)
def test_each_verified_month_has_one_hot_encoding(month):
    data = shopper()
    data["Month"] = month
    frame = encode_shopper(data, EXPECTED_COLUMNS)
    assert frame.loc[0, f"Month_{month}"] == 1
    assert frame.filter(like="Month_").sum(axis=1).item() == 1


@pytest.mark.parametrize("visitor", VISITOR_TYPES)
def test_each_verified_visitor_has_one_hot_encoding(visitor):
    data = shopper()
    data["VisitorType"] = visitor
    frame = encode_shopper(data, EXPECTED_COLUMNS)
    assert frame.loc[0, f"VisitorType_{visitor}"] == 1
    assert frame.filter(like="VisitorType_").sum(axis=1).item() == 1


@pytest.mark.parametrize(("field", "value"), [
    ("Month", "Jan"), ("Month", "Jun"), ("Month", None),
    ("VisitorType", "Returning Visitor"), ("VisitorType", "unknown"),
    ("Administrative", -1), ("Administrative", 1.5),
    ("Administrative", True), ("Administrative", "1"),
    ("ProductRelated_Duration", float("nan")),
    ("PageValues", float("inf")), ("PageValues", -0.1),
    ("BounceRates", 1.01), ("ExitRates", -0.1), ("SpecialDay", 2),
    ("Weekend", 1), ("Weekend", "false"),
    ("OperatingSystems", 2.5),
])
def test_shopper_invalid_values(field, value):
    data = shopper()
    data[field] = value
    with pytest.raises(ValueError):
        encode_shopper(data, EXPECTED_COLUMNS)


def test_missing_extra_and_non_mapping_input():
    missing = shopper()
    del missing["Month"]
    extra = {**shopper(), "Revenue": 1}
    for data in (missing, extra, None):
        with pytest.raises(ValueError):
            encode_shopper(data, EXPECTED_COLUMNS)


@pytest.mark.parametrize("columns", [
    EXPECTED_COLUMNS[:-1],
    (*EXPECTED_COLUMNS[:-1], EXPECTED_COLUMNS[0]),
    (*EXPECTED_COLUMNS[:-1], "unexpected"),
])
def test_bad_artifact_feature_contract(columns):
    with pytest.raises(ValueError, match="28 unique"):
        encode_shopper(shopper(), columns)


def test_truncated_png_is_rejected():
    with pytest.raises(ValueError, match="damaged"):
        preprocess_image(image_bytes()[:-20])
