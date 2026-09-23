from pydantic import BaseModel, ConfigDict, Field


class OrderFeatures(BaseModel):
    """
    Features required by the trained model.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    customer_state: str = Field(
        min_length=2,
        max_length=2,
        examples=["SP"],
    )

    purchase_year: int = Field(
        ge=2000,
        le=2100,
        examples=[2018],
    )

    purchase_month: int = Field(
        ge=1,
        le=12,
        examples=[1],
    )

    purchase_dayofweek: int = Field(
        ge=0,
        le=6,
        examples=[2],
    )

    purchase_hour: int = Field(
        ge=0,
        le=23,
        examples=[14],
    )

    item_count: float = Field(
        ge=0,
        examples=[1],
    )

    total_price: float = Field(
        ge=0,
        examples=[99.9],
    )

    total_freight: float = Field(
        ge=0,
        examples=[15.5],
    )

    unique_products: float = Field(
        ge=0,
        examples=[1],
    )

    unique_sellers: float = Field(
        ge=0,
        examples=[1],
    )

    total_payment: float = Field(
        ge=0,
        examples=[115.4],
    )

    payment_count: float = Field(
        ge=0,
        examples=[1],
    )

    max_installments: float = Field(
        ge=0,
        examples=[1],
    )


class PredictionResponse(BaseModel):
    """
    Response returned after prediction.
    """

    prediction: int
    probability: float | None
    model_version: str


class BatchPredictionRequest(BaseModel):
    """
    Request for multiple orders.
    """

    records: list[OrderFeatures] = Field(
        min_length=1,
        max_length=1000,
    )


class BatchPredictionResponse(BaseModel):
    """
    Response for multiple orders.
    """

    predictions: list[PredictionResponse]
