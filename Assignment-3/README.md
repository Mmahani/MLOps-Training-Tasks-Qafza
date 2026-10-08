# Assignment 3 — From Notebooks to Production

This project converts the trained machine learning model from Assignment 2 into an inference-only production service.

The service predicts whether an order belongs to the delayed-order class. It loads the fitted model and preprocessing artifacts from the `models` directory and exposes prediction endpoints through FastAPI.

## Important Design Rule

This project does not train or refit the model during inference.

The service uses:
