"""Canonical CPU-only rehydration helpers for frozen RQ2 neural inference.

These helpers mutate only newly loaded in-memory objects.  They never fit a
model, save a checkpoint, or write into a frozen artifact directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np


class CPUInferenceError(RuntimeError):
    """Raised when a frozen neural bundle cannot be rehydrated on CPU."""


def load_plain_mlp_cpu(path: str | Path, *, expected_run_id: str) -> tuple[Any, Any]:
    """Rehydrate the frozen PlainMLP checkpoint on CPU without changing it."""

    try:
        import torch

        from src.dataset_b.models.plain_mlp import (
            EXPECTED_INPUT_WIDTH,
            LOCKED_DTYPE,
            LOCKED_HIDDEN_LAYERS,
            PlainMLPClassifier,
            _PlainMLPModule,
        )

        bundle = joblib.load(Path(path))
    except Exception as exc:
        raise CPUInferenceError("Cannot read the frozen PlainMLP bundle.") from exc

    if not isinstance(bundle, dict) or set(bundle) != {
        "preprocessing",
        "model_checkpoint",
        "scope",
    }:
        raise CPUInferenceError("Frozen PlainMLP bundle structure is invalid.")
    scope = bundle["scope"]
    if (
        not isinstance(scope, dict)
        or scope.get("run_id") != expected_run_id
        or scope.get("stage") != "full_train_final_estimator"
        or scope.get("validation_accessed") is not False
        or scope.get("test_accessed") is not False
    ):
        raise CPUInferenceError("Frozen PlainMLP bundle scope is invalid.")

    payload = bundle["model_checkpoint"]
    required = {
        "payload_version",
        "model_state_dict",
        "schema",
        "runtime_policy",
        "model_policy",
        "configuration",
        "training_history",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise CPUInferenceError("Frozen PlainMLP checkpoint structure is invalid.")
    if payload["payload_version"] != 2:
        raise CPUInferenceError("Frozen PlainMLP checkpoint version is invalid.")
    if payload["runtime_policy"] != {
        "requested_device": "cuda:0",
        "device_index": 0,
        "dtype": "float64",
        "allow_cpu_fallback": False,
    }:
        raise CPUInferenceError("Frozen PlainMLP training runtime policy is invalid.")

    schema = payload["schema"]
    if schema != {
        "raw_feature_count": 15,
        "semantic_feature_count": 19,
        "encoded_feature_count": EXPECTED_INPUT_WIDTH,
        "input_width": EXPECTED_INPUT_WIDTH,
        "classes": [0, 1],
    }:
        raise CPUInferenceError("Frozen PlainMLP checkpoint schema is invalid.")
    model_policy = payload["model_policy"]
    if (
        not isinstance(model_policy, dict)
        or model_policy.get("hidden_layers") != list(LOCKED_HIDDEN_LAYERS)
        or model_policy.get("activation") != "ReLU"
        or model_policy.get("loss") != "BCEWithLogitsLoss"
        or model_policy.get("positive_class") != "approved"
        or model_policy.get("pos_weight_strategy")
        != "fit_population_n_denied_over_n_approved"
    ):
        raise CPUInferenceError("Frozen PlainMLP model policy is invalid.")
    pos_weight = model_policy.get("pos_weight")
    if (
        isinstance(pos_weight, bool)
        or not isinstance(pos_weight, (int, float))
        or not np.isfinite(pos_weight)
        or pos_weight <= 0
    ):
        raise CPUInferenceError("Frozen PlainMLP pos_weight is invalid.")

    state_dict = payload["model_state_dict"]
    if not isinstance(state_dict, dict) or not state_dict or any(
        not isinstance(tensor, torch.Tensor) or tensor.dtype != LOCKED_DTYPE
        for tensor in state_dict.values()
    ):
        raise CPUInferenceError("Frozen PlainMLP checkpoint state is invalid.")
    configuration = payload["configuration"]
    history = payload["training_history"]
    if (
        not isinstance(configuration, dict)
        or not isinstance(history, list)
        or not all(
            isinstance(value, (int, float)) and np.isfinite(value) for value in history
        )
    ):
        raise CPUInferenceError("Frozen PlainMLP checkpoint metadata is invalid.")

    try:
        estimator = PlainMLPClassifier(**configuration)
        estimator._validate_hyperparameters()
        estimator.runtime_device_ = torch.device("cpu")
        estimator.n_features_in_ = EXPECTED_INPUT_WIDTH
        estimator.classes_ = np.asarray([0, 1], dtype=np.int64)
        estimator.pos_weight_ = float(pos_weight)
        estimator.model_ = _PlainMLPModule(
            EXPECTED_INPUT_WIDTH,
            tuple(estimator.hidden_layers),
            float(estimator.dropout),
        ).to(device=estimator.runtime_device_, dtype=LOCKED_DTYPE)
        estimator.model_.load_state_dict(state_dict, strict=True)
        estimator.model_.eval()
        estimator._assert_model_runtime_state()
        estimator.training_history_ = [float(value) for value in history]
        estimator.last_dtype_audit_ = None
    except Exception as exc:
        raise CPUInferenceError("Cannot rehydrate the frozen PlainMLP state on CPU.") from exc
    return bundle["preprocessing"], estimator


def load_realmlp_cpu(path: str | Path) -> tuple[Any, Any]:
    """Load the frozen RealMLP bundle and retarget its in-memory runtime to CPU."""

    try:
        bundle = joblib.load(Path(path))
    except Exception as exc:
        raise CPUInferenceError("Cannot read the frozen RealMLP bundle.") from exc
    if not isinstance(bundle, dict) or set(bundle) != {"estimator", "preprocessing", "scope"}:
        raise CPUInferenceError("Frozen RealMLP bundle structure is invalid.")
    if bundle["scope"] != "all_immutable_train_rows":
        raise CPUInferenceError("Frozen RealMLP bundle scope is invalid.")

    estimator = bundle["estimator"]
    backend = getattr(estimator, "backend_", None)
    if backend is None:
        raise CPUInferenceError("Frozen RealMLP estimator has no fitted backend.")
    estimator.device = "cpu"
    backend.device = "cpu"
    interfaces = {
        id(interface): interface
        for interface in (
            getattr(backend, "alg_interface_", None),
            getattr(backend, "cv_alg_interface_", None),
        )
        if interface is not None
    }
    if not interfaces:
        raise CPUInferenceError("Frozen RealMLP backend has no fitted inference interface.")
    for interface in interfaces.values():
        interface.device = "cpu"
        config = getattr(interface, "config", None)
        if not isinstance(config, dict):
            raise CPUInferenceError("Frozen RealMLP inference configuration is invalid.")
        config["device"] = "cpu"
    return bundle["preprocessing"], estimator
