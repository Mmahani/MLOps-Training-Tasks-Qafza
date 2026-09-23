import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    OrderFeatures,
    PredictionResponse,
)
from src.logging_config import setup_logging
from src.predictor import Predictor


setup_logging()

logger = logging.getLogger(
    "qafza-inference"
)


MODEL_DIR = (
    Path(__file__).resolve().parents[1]
    / "models"
)


predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the model once when the API starts.
    """

    global predictor

    predictor = Predictor(
        model_dir=MODEL_DIR
    )

    logger.info(
        "Model loaded successfully from %s",
        MODEL_DIR,
    )

    yield

    logger.info(
        "API shutting down"
    )


app = FastAPI(
    title="Qafza Order Delay Prediction API",
    description=(
        "Inference-only API for predicting delayed orders."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http" )
async def log_requests(
    request: Request,
    call_next,
):
    """
    Log every HTTP request and its duration.
    """

    start_time = time.perf_counter()

    try:
        response = await call_next(request)

        duration = (
            time.perf_counter() - start_time
        ) * 1000

        logger.info(
            "%s %s status=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            duration,
        )

        return response

    except Exception:
        duration = (
            time.perf_counter() - start_time
        ) * 1000

        logger.exception(
            "%s %s failed duration_ms=%.2f",
            request.method,
            request.url.path,
            duration,
        )

        raise


@app.get("/health")
def health():
    """
    Check whether the API is running
    and the model is loaded.
    """

    return {
        "status": "ok",
        "model_loaded": predictor is not None,
    }


@app.get("/model-info")
def model_info():
    """
    Return information about the loaded model.
    """

    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    return {
        "model_version": predictor.model_version,
        "features": predictor.feature_list,
        "inference_only": True,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(order: OrderFeatures):
    """
    Predict one order.
    """

    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    result = predictor.predict(
        records=[order.model_dump()]
    )

    return result[0]


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
)
def predict_batch(
    request: BatchPredictionRequest,
):
    """
    Predict multiple orders.
    """

    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    records = [
        order.model_dump()
        for order in request.records
    ]

    results = predictor.predict(
        records=records
    )

    return {
        "predictions": results,
    }
