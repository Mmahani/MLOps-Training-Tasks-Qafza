from __future__ import annotations

import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_DIR = PROJECT_DIR / "models"


def get_model_dir() -> Path:
    """
    Return the directory that contains the model artifacts.

    The MODEL_DIR environment variable can override
    the default models directory.
    """

    configured_dir = os.getenv("MODEL_DIR")

    if configured_dir:
        return Path(configured_dir)

    return DEFAULT_MODEL_DIR


def get_model_version() -> str:
    """
    Return the model version from an environment variable.

    If MODEL_VERSION is not defined, use the default version.
    """

    return os.getenv(
        "MODEL_VERSION",
        "task2-final-model",
    )


def get_log_level() -> str:
    """
    Return the logging level.

    If LOG_LEVEL is not defined, use INFO.
    """

    return os.getenv(
        "LOG_LEVEL",
        "INFO",
    ).upper()
