import pytest

from src.features import (
    EXPECTED_FEATURES,
    build_feature_frame,
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


def test_build_feature_frame_success():
    frame = build_feature_frame(
        records=[SAMPLE_ORDER],
        expected_features=EXPECTED_FEATURES,
    )

    assert frame.shape == (1, 13)
    assert list(frame.columns) == EXPECTED_FEATURES
    assert frame.loc[0, "customer_state"] == "SP"


def test_missing_feature_is_rejected():
    incomplete_order = dict(SAMPLE_ORDER)

    del incomplete_order["total_payment"]

    with pytest.raises(
        ValueError,
        match="Missing required features",
    ):
        build_feature_frame(
            records=[incomplete_order],
            expected_features=EXPECTED_FEATURES,
        )


def test_extra_feature_is_rejected():
    order_with_extra_feature = dict(SAMPLE_ORDER)

    order_with_extra_feature["is_late"] = 1

    with pytest.raises(
        ValueError,
        match="Unexpected features",
    ):
        build_feature_frame(
            records=[order_with_extra_feature],
            expected_features=EXPECTED_FEATURES,
        )


def test_empty_records_are_rejected():
    with pytest.raises(
        ValueError,
        match="At least one input row",
    ):
        build_feature_frame(
            records=[],
            expected_features=EXPECTED_FEATURES,
        )


def test_multiple_records_are_supported():
    frame = build_feature_frame(
        records=[
            SAMPLE_ORDER,
            SAMPLE_ORDER,
        ],
        expected_features=EXPECTED_FEATURES,
    )

    assert frame.shape == (2, 13)
