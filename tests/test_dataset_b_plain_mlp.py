"""Contract tests for the sklearn-compatible plain MLP wrapper."""

import io

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.base import clone

from src.dataset_b.models import PlainMLPClassifier


def _data():
    dense = np.asarray(
        [
            [0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.2, 1.0, 0.0],
            [0.8, 0.0, 1.0], [0.1, 1.0, 0.0], [0.9, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return csr_matrix(dense), np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int8)


def _estimator():
    return PlainMLPClassifier(
        hidden_layers=(4,), dropout=0.1, batch_size=2, epochs=3, random_state=7
    )


def test_plain_mlp_is_cloneable_probabilistic_reinitialized_and_deterministic():
    X, y = _data()
    first = clone(_estimator()).fit(X, y)
    first_model = first.model_
    first_optimizer = first.optimizer_
    first_probability = first.predict_proba(X)
    first.fit(X, y)

    assert first.model_ is not first_model
    assert first.optimizer_ is not first_optimizer
    np.testing.assert_allclose(first.predict_proba(X), first_probability, rtol=0, atol=0)
    second = clone(_estimator()).fit(X, y)
    np.testing.assert_allclose(second.predict_proba(X), first_probability, rtol=0, atol=0)
    assert first.classes_.tolist() == [0, 1]
    assert first_probability.shape == (6, 2)
    np.testing.assert_allclose(first_probability.sum(axis=1), 1.0)
    assert np.array_equal(first.predict(X), (first_probability[:, 1] >= 0.5).astype(np.int8))
    checkpoint = first.checkpoint_payload()
    assert checkpoint["input_width"] == 3
    assert checkpoint["classes"] == [0, 1]
    assert len(checkpoint["training_history"]) == 3
    checkpoint_bytes = io.BytesIO()
    torch.save(checkpoint, checkpoint_bytes)
    checkpoint_bytes.seek(0)
    restored = PlainMLPClassifier.from_checkpoint_payload(torch.load(checkpoint_bytes))
    np.testing.assert_allclose(restored.predict_proba(X), first_probability, rtol=0, atol=0)


def test_plain_mlp_densifies_no_more_than_one_batch(monkeypatch):
    X, y = _data()
    observed_rows = []
    original = csr_matrix.toarray

    def tracked(self, *args, **kwargs):
        observed_rows.append(self.shape[0])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(csr_matrix, "toarray", tracked)
    estimator = _estimator().fit(X, y)
    estimator.predict_proba(X)
    assert observed_rows and max(observed_rows) <= estimator.batch_size


def test_plain_mlp_rejects_wrong_input_dtype_format_labels_and_device():
    X, y = _data()
    with pytest.raises(ValueError, match="float32"):
        _estimator().fit(X.astype(np.float64), y)
    with pytest.raises(ValueError, match="CSR"):
        _estimator().fit(csc_matrix(X), y)
    with pytest.raises(ValueError, match="both binary classes"):
        _estimator().fit(X, np.zeros(len(y), dtype=np.int8))
    with pytest.raises(ValueError, match="CPU"):
        _estimator().set_params(device="cuda").fit(X, y)
