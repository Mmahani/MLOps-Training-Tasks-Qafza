from __future__ import annotations

from typing import Any

import pandas as pd


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
    """Validate input data before sending it to the model."""

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
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise ValueError(
                f"Feature '{column}' must be numeric"
            )

    if frame["customer_state"].isna().any():
        raise ValueError(
            "customer_state cannot be null"
        )

    if (frame["purchase_month"].dropna() < 0).any():
        raise ValueError(
            "purchase_month cannot be negative"
        )

    if (frame["purchase_dayofweek"].dropna() < 0).any():
        raise ValueError(
            "purchase_dayofweek cannot be negative"
        )

    if (frame["purchase_hour"].dropna() < 0).any():
        raise ValueError(
            "purchase_hour cannot be negative"
        )


def build_feature_frame(
    records: list[dict[str, Any]],
    expected_features: list[str],
) -> pd.DataFrame:
    """Convert request records into a validated DataFrame."""

    if not records:
        raise ValueError(
            "At least one input row is required"
        )

    expected = set(expected_features)

    for record in records:
        incoming = set(record.keys())

        missing = sorted(expected - incoming)
        extra = sorted(incoming - expected)

        if missing:
            raise ValueError(
                f"Missing required features: {missing}"
            )

        if extra:
            raise ValueError(
                f"Unexpected features: {extra}"
            )

    frame = pd.DataFrame(
        records,
        columns=expected_features,
    )

    validate_feature_frame(
        frame,
        expected_features,
    )

    return frame


def transform_for_inference(
    frame: pd.DataFrame,
    preprocessor: Any,
):
    """
    Transform data using the fitted preprocessor.

    Important:
    We use transform() only.
    We do not use fit() or fit_transform().
    """

    validate_feature_frame(
        frame,
        list(frame.columns),
    )

    transformed = preprocessor.transform(frame)

    return transformed
