from pathlib import Path

from src.predictor import Predictor


MODEL_DIR = Path("models")


sample = {
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


predictor = Predictor(
    model_dir=MODEL_DIR
)

results = predictor.predict(
    records=[sample]
)

print("Prediction result:")
print(results)
