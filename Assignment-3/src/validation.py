from __future__ import annotations

from typing import Any

import pandas as pd


REQUIRED_FEATURES = [
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


def validate_columns(
    frame: pd.DataFrame,
    required_columns: list[str] | None = None,
) -> None:
    """
    Check that required columns exist and
    reject unexpected columns.
    """

    expected = required_columns or REQUIRED_FEATURES

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


def validate_not_empty(
    frame: pd.DataFrame,
) -> None:
    """
    Reject an empty DataFrame.
    """

    if frame.empty:
        raise ValueError(
            "At least one input row is required"
        )


def validate_numeric_columns(
    frame: pd.DataFrame,
) -> None:
    """
    Validate numeric feature data types.
    """

    for column in NUMERIC_FEATURES:
        if not pd.api.types.is_numeric_dtype(
            frame[column]
        ):
            raise ValueError(
                f"Feature '{column}' must be numeric"
            )


def validate_ranges(
    frame: pd.DataFrame,
) -> None:
    """
    Validate accepted ranges for date and time features.
    """

    if (
        (frame["purchase_year"] < 2000)
        | (frame["purchase_year"] > 2100)
    ).any():
        raise ValueError(
            "purchase_year must be between 2000 and 2100"
        )

    if (
        (frame["purchase_month"] < 1)
        | (frame["purchase_month"] > 12)
    ).any():
        raise ValueError(
            "purchase_month must be between 1 and 12"
        )

    if (
        (frame["purchase_dayofweek"] < 0)
        | (frame["purchase_dayofweek"] > 6)
    ).any():
        raise ValueError(
            "purchase_dayofweek must be between 0 and 6"
        )

    if (
        (frame["purchase_hour"] < 0)
        | (frame["purchase_hour"] > 23)
    ).any():
        raise ValueError(
            "purchase_hour must be between 0 and 23"
        )


def validate_missing_values(
    frame: pd.DataFrame,
) -> None:
    """
    Validate required non-null values.
    """

    if frame["customer_state"].isna().any():
        raise ValueError(
            "customer_state cannot be null"
        )


def validate_records(
    records: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Validate raw records and return a DataFrame.
    """

    if not records:
        raise ValueError(
            "At least one input row is required"
        )

    expected = set(REQUIRED_FEATURES)

    for record in records:
        incoming = set(record.keys())

        missing = sorted(
            expected - incoming
        )

        extra = sorted(
            incoming - expected
        )

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
        columns=REQUIRED_FEATURES,
    )

    validate_columns(
        frame=frame,
        required_columns=REQUIRED_FEATURES,
    )

    validate_not_empty(frame)
    validate_numeric_columns(frame)
    validate_ranges(frame)
    validate_missing_values(frame)

    return frame
