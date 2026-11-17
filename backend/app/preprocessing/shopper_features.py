"""Convert the verified 17-field contract into the model's ordered 28 features."""
from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Integral, Real

import pandas as pd

BASE_COLUMNS = (
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
    "OperatingSystems", "Browser", "Region", "TrafficType", "Weekend",
)
# Exact suffixes present in the saved training feature list, not guessed aliases.
MONTHS = ("Aug", "Dec", "Feb", "Jul", "June", "Mar", "May", "Nov", "Oct", "Sep")
VISITOR_TYPES = ("New_Visitor", "Other", "Returning_Visitor")
EXPECTED_COLUMNS = (
    *BASE_COLUMNS,
    *(f"Month_{month}" for month in MONTHS),
    *(f"VisitorType_{visitor}" for visitor in VISITOR_TYPES),
)
RAW_FIELDS = frozenset((*BASE_COLUMNS, "Month", "VisitorType"))
INTEGER_FIELDS = frozenset({
    "Administrative", "Informational", "ProductRelated",
    "OperatingSystems", "Browser", "Region", "TrafficType",
})
RATE_FIELDS = frozenset({"BounceRates", "ExitRates", "SpecialDay"})


def validate_feature_columns(columns: Sequence[str]) -> tuple[str, ...]:
    result = tuple(columns)
    if (
        len(result) != 28
        or not all(isinstance(column, str) for column in result)
        or len(set(result)) != 28
        or set(result) != set(EXPECTED_COLUMNS)
    ):
        raise ValueError("Model feature columns must contain the verified 28 unique names.")
    return result


def encode_shopper(
    data: Mapping[str, object], feature_columns: Sequence[str]
) -> pd.DataFrame:
    """Validate all 17 fields; preserve artifact order and apply no scaling."""
    if not isinstance(data, Mapping):
        raise ValueError("Shopper input must be a mapping of 17 named fields.")
    missing = RAW_FIELDS - data.keys()
    extra = data.keys() - RAW_FIELDS
    if missing or extra:
        raise ValueError(
            f"Shopper fields do not match: missing={sorted(missing)}, "
            f"extra={sorted(map(str, extra))}."
        )
    columns = validate_feature_columns(feature_columns)
    month = data["Month"]
    visitor = data["VisitorType"]
    if not isinstance(month, str) or month not in MONTHS:
        raise ValueError(f"Unknown Month; supported values are {', '.join(MONTHS)}.")
    if not isinstance(visitor, str) or visitor not in VISITOR_TYPES:
        raise ValueError(
            f"Unknown VisitorType; supported values are {', '.join(VISITOR_TYPES)}."
        )
    row: dict[str, int | float] = {}
    for field in BASE_COLUMNS:
        value = data[field]
        if field == "Weekend":
            if not isinstance(value, bool):
                raise ValueError("Weekend must be a boolean.")
            row[field] = int(value)
            continue
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{field} must be numeric, not a string or boolean.")
        try:
            finite = isfinite(value)
        except (OverflowError, TypeError):
            finite = False
        if not finite or value < 0:
            raise ValueError(f"{field} must be finite and non-negative.")
        if field in INTEGER_FIELDS:
            if not isinstance(value, Integral):
                raise ValueError(f"{field} must be an integer.")
            row[field] = int(value)
        else:
            if field in RATE_FIELDS and value > 1:
                raise ValueError(f"{field} must be between 0 and 1.")
            row[field] = float(value)
    row.update({f"Month_{item}": int(item == month) for item in MONTHS})
    row.update({f"VisitorType_{item}": int(item == visitor) for item in VISITOR_TYPES})
    return pd.DataFrame([row], columns=list(columns))
