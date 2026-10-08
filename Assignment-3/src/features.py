from __future__ import annotations

from typing import Any

import pandas as pd

from src.validation import validate_records


EXPECTED_FEATURES = [
    "customer_state",
    "purchase_year",
    "purchase_month",
    "purchase_dayofweek",
    "purchase_hour",
    "item_count",
    "total_price",
    "total_freight",
    "unique_products",
    "unique_sellers",
    "total_payment",
    "payment_count",
    "max_installments",
]


NUMERIC_FEATURES = [
    "purchase_year",
    "purchase_month",
    "purchase_dayofweek",
    "purchase_hour",
    "item_count",
    "total_price",
    "total_freight",
    "unique_products",
    "unique_sellers",
    "total_payment",
    "payment_count",
    "max_installments",
]


def validate_feature_frame(
    frame: pd.DataFrame,
    expected_features: list[str] | None = None,
) -> None:
    """
    Validate the feature DataFrame before inference.

    This function does not train or fit anything.
    """

    expected = expected_features or EXPECTED_FEATURES

    missing = [
        column
        for column in expected
        if column not in frame.columns
    ]

    extra = [
        column
        for column in frame.columns
        if column not in expected
    ]

    if missing:
        raise ValueError(
            f"Missing required features: {missing}"
        )

    if extra:
        raise ValueError(
            f"Unexpected features: {extra}"
        )

    if frame.empty:
        raise ValueError(
            "At least one input row is required"
        )

    for column in NUMERIC_FEATURES:
        if not pd.api.types.is_numeric_dtype(
            frame[column]
        ):
            raise ValueError(
                f"Feature '{column}' must be numeric"
            )

    if frame["customer_state"].isna().any():
        raise ValueError(
            "customer_state cannot be null"
        )

    if (frame["purchase_year"] < 2000).any():
        raise ValueError(
            "purchase_year cannot be less than 2000"
        )

    if (frame["purchase_year"] > 2100).any():
        raise ValueError(
            "purchase_year cannot be greater than 2100"
        )

    if (frame["purchase_month"] < 1).any():
        raise ValueError(
            "purchase_month cannot be less than 1"
        )

    if (frame["purchase_month"] > 12).any():
        raise ValueError(
            "purchase_month cannot be greater than 12"
        )

    if (frame["purchase_dayofweek"] < 0).any():
        raise ValueError(
            "purchase_dayofweek cannot be negative"
        )

    if (frame["purchase_dayofweek"] > 6).any():
        raise ValueError(
            "purchase_dayofweek cannot be greater than 6"
        )

    if (frame["purchase_hour"] < 0).any():
        raise ValueError(
            "purchase_hour cannot be negative"
        )

    if (frame["purchase_hour"] > 23).any():
        raise ValueError(
            "purchase_hour cannot be greater than 23"
        )


def build_feature_frame(
    records: list[dict[str, Any]],
    expected_features: list[str],
) -> pd.DataFrame:
    """
    Convert raw records into a validated DataFrame.

    The validation is delegated to validation.py.
    """

    if not records:
        raise ValueError(
            "At least one input row is required"
        )

    if expected_features != EXPECTED_FEATURES:
        raise ValueError(
            "Unexpected feature definition"
        )

    frame = validate_records(
        records=records
    )

    validate_feature_frame(
        frame=frame,
        expected_features=expected_features,
    )

    return frame


def transform_for_inference(
    frame: pd.DataFrame,
    preprocessor: Any,
):
    """
    Transform data using the already-fitted preprocessor.

    Important:
    This function uses transform() only.
    It never uses fit() or fit_transform().
    """

    validate_feature_frame(
        frame=frame,
        expected_features=list(frame.columns),
    )

    transformed = preprocessor.transform(
        frame
    )

    return transformed
