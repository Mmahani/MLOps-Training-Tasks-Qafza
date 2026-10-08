from pathlib import Path

import pytest

from src.predictor import Predictor


MODEL_DIR = Path("models")


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


def test_model_artifacts_exist():
    assert (
        MODEL_DIR / "final_model.joblib"
    ).exists()

    assert (
        MODEL_DIR / "preprocessor.joblib"
    ).exists()

    assert (
        MODEL_DIR / "feature_list.joblib"
    ).exists()


def test_predictor_loads_model():
    predictor = Predictor(
        model_dir=MODEL_DIR
    )

    assert predictor.model is not None
    assert predictor.preprocessor is not None
    assert len(predictor.feature_list) == 13


def test_predictor_returns_prediction():
    predictor = Predictor(
        model_dir=MODEL_DIR
    )

    results = predictor.predict(
        records=[SAMPLE_ORDER]
    )

    assert len(results) == 1

    result = results[0]

    assert "prediction" in result
    assert "probability" in result
    assert "model_version" in result

    assert result["prediction"] in [0, 1]
    assert 0 <= result["probability"] <= 1
    assert result["model_version"] == (
        "task2-final-model"
    )


def test_predictor_supports_batch():
    predictor = Predictor(
        model_dir=MODEL_DIR
    )

    results = predictor.predict(
        records=[
            SAMPLE_ORDER,
            SAMPLE_ORDER,
        ]
    )

    assert len(results) == 2

    for result in results:
        assert result["prediction"] in [0, 1]
        assert 0 <= result["probability"] <= 1


def test_predictor_rejects_target_leakage():
    predictor = Predictor(
        model_dir=MODEL_DIR
    )

    invalid_order = dict(SAMPLE_ORDER)
    invalid_order["is_late"] = 1

    with pytest.raises(
        ValueError,
        match="Unexpected features",
    ):
        predictor.predict(
            records=[invalid_order]
        )
