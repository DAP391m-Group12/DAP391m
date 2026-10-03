"""Contract tests for the sklearn-compatible Dataset B RealMLP wrapper."""

import importlib.util

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src.dataset_b.models import (
    RealMLPClassifier,
    RealMLPConfigurationError,
    load_realmlp_v1_config,
)
from src.dataset_b.preprocessing import make_realmlp_native_preprocessing_pipeline
from src.dataset_b.preprocessing.categorical_encoder import CATEGORICAL_COLUMNS
from src.dataset_b.preprocessing.raw_parser import DatasetBRawParser, FEATURE_COLUMNS


class _FakeRealMLPBackend:
    instances = []

    def __init__(self, **parameters):
        self.parameters = parameters
        self.__class__.instances.append(self)

    def fit(self, X, y, **kwargs):
        self.fit_X = X
        self.fit_y = y
        self.fit_kwargs = kwargs
        self.classes_ = np.asarray([0, 1])
        return self

    def predict_proba(self, X):
        positive = np.linspace(0.2, 0.8, len(X), dtype=np.float64)
        return np.column_stack((1.0 - positive, positive))


def _row(**changes):
    row = {
        "loan_type": "1",
        "loan_purpose": "1",
        "lien_status": "1",
        "reverse_mortgage": "2",
        "open-end_line_of_credit": "2",
        "business_or_commercial_purpose": "2",
        "income": "100",
        "debt_to_income_ratio": "42",
        "loan_amount": "200",
        "loan_to_value_ratio": "75",
        "loan_term": "360",
        "occupancy_type": "1",
        "preapproval": "1",
        "interest_only_payment": "2",
        "balloon_payment": "2",
    }
    row.update(changes)
    return row


def _native_frame():
    raw = pd.DataFrame(
        [
            _row(income="0"),
            _row(income="100"),
            _row(income="NA"),
            _row(income="200"),
        ],
        columns=FEATURE_COLUMNS,
        dtype=object,
    )
    normalized = DatasetBRawParser().parse(raw).cleaned
    return make_realmlp_native_preprocessing_pipeline().fit_transform(normalized)


def _fake_backend(monkeypatch):
    _FakeRealMLPBackend.instances.clear()
    monkeypatch.setattr(
        RealMLPClassifier,
        "_backend_class",
        staticmethod(lambda: _FakeRealMLPBackend),
    )


def test_realmlp_config_is_frozen_and_cloneable():
    payload = load_realmlp_v1_config()
    estimator = RealMLPClassifier.from_yaml()
    cloned = clone(estimator)

    assert payload["backend"]["package_version"] == "1.7.3"
    assert payload["backend_controls"]["val_fraction"] == 0.0
    assert estimator.get_params(deep=False) == cloned.get_params(deep=False)
    assert estimator.device == "cuda"
    assert estimator.hidden_sizes == [256, 256, 256]
    assert estimator.act == "selu"
    assert estimator.use_early_stopping is False
    assert estimator.use_best_epoch is True


def test_realmlp_wrapper_passes_explicit_validation_and_native_categories(monkeypatch):
    _fake_backend(monkeypatch)
    X = _native_frame()
    y = np.asarray([0, 1, 0, 1], dtype=np.int8)
    estimator = RealMLPClassifier(device="cpu", verbosity=0)

    estimator.fit(X.iloc[:2], y[:2], X_val=X.iloc[2:], y_val=y[2:])
    first_backend = estimator.backend_
    probabilities = estimator.predict_proba(X)
    estimator.fit(X.iloc[:2], y[:2], X_val=X.iloc[2:], y_val=y[2:])

    assert estimator.backend_ is not first_backend
    assert first_backend.parameters["val_fraction"] == 0.0
    assert first_backend.fit_kwargs["cat_col_names"] == list(CATEGORICAL_COLUMNS)
    pd.testing.assert_frame_equal(first_backend.fit_kwargs["X_val"], X.iloc[2:])
    np.testing.assert_array_equal(first_backend.fit_kwargs["y_val"], y[2:])
    assert estimator.runtime_metadata_["validation_source"] == "project_explicit"
    assert estimator.classes_.tolist() == [0, 1]
    assert probabilities.shape == (4, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    np.testing.assert_array_equal(
        estimator.predict(X), (estimator.predict_proba(X)[:, 1] >= 0.5).astype(np.int8)
    )


def test_realmlp_wrapper_rejects_invalid_boundaries_and_hidden_splits(monkeypatch):
    _fake_backend(monkeypatch)
    X = _native_frame()
    y = np.asarray([0, 1, 0, 1], dtype=np.int8)

    with pytest.raises(ValueError, match="supplied together"):
        RealMLPClassifier(device="cpu").fit(X, y, X_val=X)
    with pytest.raises(ValueError, match="both binary classes"):
        RealMLPClassifier(device="cpu").fit(X, np.zeros(len(X), dtype=np.int8))
    wrong_dtype = X.copy(deep=True)
    wrong_dtype["income_value"] = wrong_dtype["income_value"].astype(np.float64)
    with pytest.raises(ValueError, match="float32"):
        RealMLPClassifier(device="cpu").fit(wrong_dtype, y)
    with pytest.raises(ValueError, match="internal CV"):
        RealMLPClassifier(device="cpu", n_cv=2).fit(X, y)
    with pytest.raises(ValueError, match="silent CPU fallback"):
        RealMLPClassifier(device="cpu", allow_silent_cpu_fallback=True).fit(X, y)


def test_realmlp_wrapper_detects_invalid_backend_probabilities(monkeypatch):
    _fake_backend(monkeypatch)
    X = _native_frame()
    y = np.asarray([0, 1, 0, 1], dtype=np.int8)
    estimator = RealMLPClassifier(device="cpu").fit(X, y)
    estimator.backend_.predict_proba = lambda frame: np.ones((len(frame), 1))
    with pytest.raises(ValueError, match="two columns"):
        estimator.predict_proba(X)


@pytest.mark.skipif(
    importlib.util.find_spec("pytabkit") is None,
    reason="pytabkit is installed only in the model_train environment",
)
def test_realmlp_backend_173_public_api_is_available():
    backend_class = RealMLPClassifier._backend_class()
    backend = backend_class(**RealMLPClassifier(device="cpu", verbosity=0).backend_parameters())

    assert backend.__class__.__name__ == "RealMLP_TD_Classifier"
    assert callable(backend.fit)
    assert callable(backend.predict_proba)
    assert clone(backend).__class__ is backend.__class__


def test_realmlp_config_loader_rejects_drift(tmp_path):
    config_path = tmp_path / "realmlp.yaml"
    config_path.write_text(
        "schema_version: 1\ndataset: dataset_b\nmodel: realmlp\n",
        encoding="utf-8",
    )
    with pytest.raises(RealMLPConfigurationError, match="header"):
        load_realmlp_v1_config(config_path)
