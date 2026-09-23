from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


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


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info_endpoint():
    response = client.get("/model-info")

    assert response.status_code == 200

    body = response.json()

    assert body["model_version"] == "task2-final-model"
    assert body["inference_only"] is True
    assert len(body["features"]) == 13


def test_predict_endpoint():
    response = client.post(
        "/predict",
        json=SAMPLE_ORDER,
    )

    assert response.status_code == 200

    body = response.json()

    assert "prediction" in body
    assert "probability" in body
    assert "model_version" in body

    assert body["prediction"] in [0, 1]
    assert 0 <= body["probability"] <= 1


def test_batch_predict_endpoint():
    response = client.post(
        "/predict/batch",
        json={
            "records": [
                SAMPLE_ORDER,
                SAMPLE_ORDER,
            ]
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert "predictions" in body
    assert len(body["predictions"]) == 2


def test_invalid_month_is_rejected():
    invalid_order = dict(SAMPLE_ORDER)

    invalid_order["purchase_month"] = 15

    response = client.post(
        "/predict",
        json=invalid_order,
    )

    assert response.status_code == 422


def test_target_leakage_is_rejected():
    leakage_order = dict(SAMPLE_ORDER)

    leakage_order["is_late"] = 1

    response = client.post(
        "/predict",
        json=leakage_order,
    )

    assert response.status_code == 422
