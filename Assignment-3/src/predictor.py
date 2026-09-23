from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib

from src.features import (
    build_feature_frame,
    transform_for_inference,
)


class Predictor:
    """
    Inference-only predictor.

    This class loads already-fitted artifacts.
    It does not train or fit anything.
    """

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)

        self.model = joblib.load(
            self.model_dir / "final_model.joblib"
        )

        self.preprocessor = joblib.load(
            self.model_dir / "preprocessor.joblib"
        )

        self.feature_list = list(
            joblib.load(
                self.model_dir / "feature_list.joblib"
            )
        )

        if not self.feature_list:
            raise ValueError(
                "feature_list.joblib is empty"
            )

        if not hasattr(self.model, "predict"):
            raise TypeError(
                "Loaded object does not have predict()"
            )

        if not hasattr(self.preprocessor, "transform"):
            raise TypeError(
                "Loaded preprocessor does not have transform()"
            )

    @property
    def model_version(self) -> str:
        """
        Return the model version.

        It can be changed using the MODEL_VERSION
        environment variable.
        """

        return os.getenv(
            "MODEL_VERSION",
            "task2-final-model",
        )

    def predict(
        self,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Predict one or more records.
        """

        frame = build_feature_frame(
            records=records,
            expected_features=self.feature_list,
        )

        transformed = transform_for_inference(
            frame=frame,
            preprocessor=self.preprocessor,
        )

        predictions = self.model.predict(
            transformed
        )

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(
                transformed
            )[:, 1]
        else:
            probabilities = [None] * len(predictions)

        results = []

        for prediction, probability in zip(
            predictions,
            probabilities,
        ):
            result = {
                "prediction": int(prediction),
                "probability": (
                    None
                    if probability is None
                    else float(probability)
                ),
                "model_version": self.model_version,
            }

            results.append(result)

        return results
