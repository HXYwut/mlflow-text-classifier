"""Model loading and inference wrapper.

The trained pipeline (TF-IDF vectorizer + LogisticRegression) is saved as a
joblib bundle under ``ml/registry/``. The path can be overridden with the
``MODEL_PATH`` env var, which is what the Docker image uses.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "ml" / "registry" / "model.joblib"

LABELS = {0: "hockey", 1: "space"}


@lru_cache(maxsize=1)
def load_model() -> Pipeline:
    """Load the trained sklearn pipeline once and cache it."""
    model_path = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Run `python -m ml.train` first, or set MODEL_PATH env var."
        )
    model = joblib.load(model_path)
    return model


def predict(text: str) -> tuple[str, float]:
    """Return (label, probability) for a single input text."""
    model = load_model()
    label_id = int(model.predict([text])[0])
    proba = float(model.predict_proba([text])[0].max())
    return LABELS[label_id], proba
