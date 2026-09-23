from pydantic import ValidationError

from app.schemas import OrderFeatures


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


valid_order = OrderFeatures(
    **sample
)

print("Valid order:")
print(valid_order.model_dump())


invalid_order = dict(sample)

invalid_order["purchase_month"] = 15

try:
    OrderFeatures(**invalid_order)
except ValidationError as error:
    print("Invalid order rejected successfully:")
    print(error)


leakage_order = dict(sample)

leakage_order["is_late"] = 1

try:
    OrderFeatures(**leakage_order)
except ValidationError as error:
    print("Leakage field rejected successfully:")
    print(error)
