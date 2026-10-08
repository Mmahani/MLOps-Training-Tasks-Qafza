import pytest

from src.validation import (
    REQUIRED_FEATURES,
    validate_columns,
    validate_ranges,
    validate_records,
)


SAMPLE_ORDER = {
    "customer_state": "SP",
    "purchase_year": 2018,
    "purchase_month": 1,
    "purchase_dayofweek": 2,
    "purchase_hour": 14,
    "item_count": 1,
    "total_price": 99.9,
    "total_freight": 15.5,
    "unique_products": 1,
    "unique_sellers": 1,
    "total_payment": 115.4,
    "payment_count": 1,
    "max_installments": 1,
}


def test_validate_records_success():
    frame = validate_records(
        records=[SAMPLE_ORDER]
    )

    assert frame.shape == (1, 13)
    assert list(frame.columns) == REQUIRED_FEATURES


def test_empty_records_are_rejected():
    with pytest.raises(
        ValueError,
        match="At least one input row",
    ):
        validate_records(records=[])


def test_missing_column_is_rejected():
    invalid_order = dict(SAMPLE_ORDER)
    del invalid_order["total_payment"]

    with pytest.raises(
        ValueError,
        match="Missing required",
    ):
        validate_columns(
            frame=__import__("pandas").DataFrame(
                [invalid_order]
            ),
                required_columns=REQUIRED_FEATURES,
        )


def test_extra_column_is_rejected():
    invalid_order = dict(SAMPLE_ORDER)
    invalid_order["is_late"] = 1

    import pandas as pd

    frame = pd.DataFrame([invalid_order])

    with pytest.raises(
        ValueError,
        match="Unexpected",
    ):
        validate_columns(
            frame=frame,
            required_columns=REQUIRED_FEATURES,
        )


def test_invalid_month_is_rejected():
    import pandas as pd

    invalid_order = dict(SAMPLE_ORDER)
    invalid_order["purchase_month"] = 15

    frame = pd.DataFrame([invalid_order])

    with pytest.raises(
        ValueError,
        match="purchase_month",
    ):
        validate_ranges(frame)


def test_invalid_hour_is_rejected():
    import pandas as pd

    invalid_order = dict(SAMPLE_ORDER)
    invalid_order["purchase_hour"] = 25

    frame = pd.DataFrame([invalid_order])

    with pytest.raises(
        ValueError,
        match="purchase_hour",
    ):
        validate_ranges(frame)
