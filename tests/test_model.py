"""Tests for the model-loading wrapper. We don't ship a trained artifact in
CI, so we exercise the wrapper logic with a fake model."""


import joblib
import numpy as np
import pytest

from app import model as model_mod


class _AlwaysSpace:
    """Stand-in for a fitted sklearn estimator."""

    def predict(self, X):
        return np.array([1] * len(X))

    def predict_proba(self, X):
        return np.array([[0.1, 0.9]] * len(X))


def test_labels_mapping():
    assert model_mod.LABELS == {0: "hockey", 1: "space"}


def test_load_model_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "does_not_exist.joblib"))
    model_mod.load_model.cache_clear()
    with pytest.raises(FileNotFoundError):
        model_mod.load_model()


def test_predict_uses_loaded_model(tmp_path, monkeypatch):
    path = tmp_path / "m.joblib"
    joblib.dump(_AlwaysSpace(), path)

    monkeypatch.setenv("MODEL_PATH", str(path))
    model_mod.load_model.cache_clear()

    label, proba = model_mod.predict("any text here")
    assert label == "space"
    assert proba == pytest.approx(0.9)
