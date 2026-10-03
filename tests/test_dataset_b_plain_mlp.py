"""Contract tests for the locked CUDA/float64 PlainMLP wrapper."""

import io
import os

import numpy as np
import pytest
import torch
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.base import clone

from src.dataset_b.models.plain_mlp import (
    EXPECTED_INPUT_WIDTH,
    LOCKED_DEVICE,
    LOCKED_DTYPE,
    LOCKED_HIDDEN_LAYERS,
    PlainMLPClassifier,
)


def _data():
    dense = np.zeros((6, EXPECTED_INPUT_WIDTH), dtype=np.float64)
    dense[:, 0] = [0.0, 1.0, 0.2, 0.8, 0.1, 0.9]
    dense[:, 1] = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    dense[:, 4] = 1.0
    labels = np.asarray([0, 1, 0, 0, 0, 1], dtype=np.int8)
    return csr_matrix(dense), labels


def _estimator():
    return PlainMLPClassifier(batch_size=2, epochs=1, random_state=7)


@pytest.fixture
def simulated_cuda_on_cpu(monkeypatch):
    """Exercise numerical code without weakening the production CUDA gate."""
    monkeypatch.setattr(
        PlainMLPClassifier,
        "_validated_device",
        lambda self: torch.device("cpu"),
    )


def test_locked_architecture_parameter_and_training_tensor_dtypes_pos_weight_and_determinism(
    simulated_cuda_on_cpu,
):
    X, y = _data()
    first = clone(_estimator()).fit(X, y)
    first_probability = first.predict_proba(X)
    second = clone(_estimator()).fit(X, y)

    assert first.hidden_layers == LOCKED_HIDDEN_LAYERS == (256, 128, 64)
    linear_layers = [
        layer for layer in first.model_.network if isinstance(layer, torch.nn.Linear)
    ]
    assert [(layer.in_features, layer.out_features) for layer in linear_layers] == [
        (62, 256),
        (256, 128),
        (128, 64),
        (64, 1),
    ]
    assert sum(isinstance(layer, torch.nn.ReLU) for layer in first.model_.network) == 3
    assert all(parameter.dtype == LOCKED_DTYPE for parameter in first.model_.parameters())
    assert first.last_dtype_audit_ == {
        "dense_batch": "float64",
        "target": "float64",
        "logits": "float64",
        "loss": "float64",
    }
    assert first.pos_weight_ == 2.0
    assert first.criterion_.pos_weight.dtype == LOCKED_DTYPE
    assert first.criterion_.pos_weight.item() == 2.0
    assert first_probability.dtype == np.float64
    assert first_probability.shape == (6, 2)
    np.testing.assert_array_equal(second.predict_proba(X), first_probability)
    np.testing.assert_allclose(
        first_probability.sum(axis=1), 1.0, rtol=0, atol=1.0e-7
    )


def test_plain_mlp_densifies_no_more_than_one_float64_batch(
    monkeypatch, simulated_cuda_on_cpu
):
    X, y = _data()
    observed_shapes = []
    original = csr_matrix.toarray

    def tracked(self, *args, **kwargs):
        observed_shapes.append(self.shape)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(csr_matrix, "toarray", tracked)
    estimator = _estimator().fit(X, y)
    estimator.predict_proba(X)

    assert observed_shapes
    assert max(rows for rows, _ in observed_shapes) <= estimator.batch_size
    assert {columns for _, columns in observed_shapes} == {EXPECTED_INPUT_WIDTH}


def test_plain_mlp_rejects_wrong_input_dtype_format_width_and_labels(
    simulated_cuda_on_cpu,
):
    X, y = _data()
    with pytest.raises(ValueError, match="float64"):
        _estimator().fit(X.astype(np.float32), y)
    with pytest.raises(ValueError, match="CSR"):
        _estimator().fit(csc_matrix(X), y)
    with pytest.raises(ValueError, match="62 encoded columns"):
        _estimator().fit(X[:, :61], y)
    with pytest.raises(ValueError, match="both binary classes"):
        _estimator().fit(X, np.zeros(len(y), dtype=np.int8))


def test_cuda_device_gate_rejects_wrong_device_build_and_unavailable_cuda(monkeypatch):
    X, y = _data()
    with pytest.raises(ValueError, match="cuda:0"):
        _estimator().set_params(device="cpu").fit(X, y)
    with pytest.raises(ValueError, match="cuda:0"):
        _estimator().set_params(device="cuda:1").fit(X, y)

    estimator = _estimator()
    monkeypatch.setattr(torch.version, "cuda", None)
    with pytest.raises(RuntimeError, match="CUDA-enabled PyTorch build"):
        estimator._validated_device()

    monkeypatch.setattr(torch.version, "cuda", "12.test")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CPU fallback is forbidden"):
        estimator._validated_device()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    with pytest.raises(RuntimeError, match="device index 0"):
        estimator._validated_device()

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(estimator, "_cuda_float64_probe", lambda device: None)
    assert estimator._validated_device() == torch.device(LOCKED_DEVICE)


def test_checkpoint_payload_preserves_float64_schema_device_and_policy(
    simulated_cuda_on_cpu,
):
    X, y = _data()
    fitted = _estimator().fit(X, y)
    expected = fitted.predict_proba(X)
    payload = fitted.checkpoint_payload()

    assert payload["schema"] == {
        "raw_feature_count": 15,
        "semantic_feature_count": 19,
        "encoded_feature_count": 62,
        "input_width": 62,
        "classes": [0, 1],
    }
    assert payload["runtime_policy"] == {
        "requested_device": "cuda:0",
        "device_index": 0,
        "dtype": "float64",
        "allow_cpu_fallback": False,
    }
    assert payload["model_policy"]["hidden_layers"] == [256, 128, 64]
    assert payload["model_policy"]["activation"] == "ReLU"
    assert payload["model_policy"]["pos_weight"] == 2.0
    assert all(
        tensor.device.type == "cpu" and tensor.dtype == LOCKED_DTYPE
        for tensor in payload["model_state_dict"].values()
    )

    checkpoint_bytes = io.BytesIO()
    torch.save(payload, checkpoint_bytes)
    checkpoint_bytes.seek(0)
    restored_payload = torch.load(checkpoint_bytes, weights_only=False)
    restored = PlainMLPClassifier.from_checkpoint_payload(restored_payload)

    assert all(parameter.dtype == LOCKED_DTYPE for parameter in restored.model_.parameters())
    assert restored.runtime_device_ == torch.device("cpu")
    np.testing.assert_array_equal(restored.predict_proba(X), expected)

    corrupted = fitted.checkpoint_payload()
    key = next(iter(corrupted["model_state_dict"]))
    corrupted["model_state_dict"][key] = corrupted["model_state_dict"][key].float()
    with pytest.raises(ValueError, match="state must be float64"):
        PlainMLPClassifier.from_checkpoint_payload(corrupted)


def test_probability_validation_is_exact_and_non_mutating(simulated_cuda_on_cpu):
    X, y = _data()
    original_matrix = X.copy()
    estimator = _estimator().fit(X, y)
    probabilities = estimator.predict_proba(X)
    before = probabilities.copy()

    validated = estimator._validate_probabilities(probabilities)

    assert validated is probabilities
    np.testing.assert_array_equal(validated, before)
    np.testing.assert_array_equal(X.data, original_matrix.data)
    np.testing.assert_array_equal(X.indices, original_matrix.indices)
    np.testing.assert_array_equal(X.indptr, original_matrix.indptr)

    invalid = probabilities.copy()
    invalid[0, 0] += 0.01
    invalid_before = invalid.copy()
    with pytest.raises(ValueError, match="sum to one"):
        estimator._validate_probabilities(invalid)
    np.testing.assert_array_equal(invalid, invalid_before)


def test_seed_setup_enables_locked_deterministic_controls(monkeypatch):
    calls = {"manual_seed": [], "manual_seed_all": [], "deterministic": []}
    monkeypatch.setattr(torch, "manual_seed", lambda seed: calls["manual_seed"].append(seed))
    monkeypatch.setattr(
        torch.cuda,
        "manual_seed_all",
        lambda seed: calls["manual_seed_all"].append(seed),
    )
    monkeypatch.setattr(
        torch,
        "use_deterministic_algorithms",
        lambda enabled: calls["deterministic"].append(enabled),
    )

    _estimator()._seed_runtime()

    assert calls == {
        "manual_seed": [7],
        "manual_seed_all": [7],
        "deterministic": [True],
    }
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
