"""Deterministic CUDA/float64 PlainMLP wrapper for Dataset B CSR inputs."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch
from scipy.sparse import csr_matrix, isspmatrix_csr
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from torch import nn


LOCKED_HIDDEN_LAYERS = (256, 128, 64)
LOCKED_DEVICE = "cuda:0"
LOCKED_DTYPE = torch.float64
EXPECTED_RAW_FEATURE_COUNT = 15
EXPECTED_SEMANTIC_FEATURE_COUNT = 19
EXPECTED_INPUT_WIDTH = 62
PROBABILITY_ROW_SUM_RTOL = 0.0
PROBABILITY_ROW_SUM_ATOL = 1.0e-7


class _PlainMLPModule(nn.Module):
    def __init__(self, input_width: int, hidden_layers: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = input_width
        for hidden_width in hidden_layers:
            layers.extend((nn.Linear(width, hidden_width), nn.ReLU()))
            if dropout:
                layers.append(nn.Dropout(dropout))
            width = hidden_width
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.network(X).squeeze(1)


class PlainMLPClassifier(ClassifierMixin, BaseEstimator):
    """Locked binary MLP that densifies only the current CSR batch."""

    def __init__(
        self,
        *,
        hidden_layers: tuple[int, ...] = LOCKED_HIDDEN_LAYERS,
        activation: str = "ReLU",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.1,
        batch_size: int = 256,
        epochs: int = 100,
        random_state: int = 42,
        device: str = LOCKED_DEVICE,
        deterministic_algorithms: bool = True,
    ) -> None:
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        self.device = device
        self.deterministic_algorithms = deterministic_algorithms

    def fit(self, X: csr_matrix, y: object) -> PlainMLPClassifier:
        self._validate_hyperparameters()
        matrix = self._validate_X(X)
        labels = self._validated_labels(y, matrix.shape[0])
        runtime_device = self._validated_device()

        self._seed_runtime()
        self.runtime_device_ = runtime_device
        self.n_features_in_ = matrix.shape[1]
        self.classes_ = np.asarray([0, 1], dtype=np.int64)
        self.pos_weight_ = self._fit_population_pos_weight(labels)
        self.model_ = _PlainMLPModule(
            self.n_features_in_, tuple(self.hidden_layers), float(self.dropout)
        ).to(device=runtime_device, dtype=LOCKED_DTYPE)
        self._assert_model_runtime_state()
        self.optimizer_ = torch.optim.AdamW(
            self.model_.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )
        pos_weight_tensor = torch.tensor(
            [self.pos_weight_], device=runtime_device, dtype=LOCKED_DTYPE
        )
        self.criterion_ = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        rng = np.random.default_rng(self.random_state)
        self.training_history_: list[float] = []
        self.last_dtype_audit_: dict[str, str] | None = None

        for _ in range(self.epochs):
            self.model_.train()
            order = rng.permutation(matrix.shape[0])
            total_loss = 0.0
            total_rows = 0
            for start in range(0, matrix.shape[0], self.batch_size):
                positions = order[start : start + self.batch_size]
                batch_X = self._dense_tensor(matrix[positions])
                batch_y = self._target_tensor(labels[positions])
                self.optimizer_.zero_grad(set_to_none=True)
                logits = self.model_(batch_X)
                loss = self.criterion_(logits, batch_y)
                self._validate_training_dtypes(batch_X, batch_y, logits, loss)
                loss.backward()
                self.optimizer_.step()
                rows = len(positions)
                total_loss += float(loss.detach().cpu()) * rows
                total_rows += rows
            self.training_history_.append(total_loss / total_rows)
        return self

    def predict_proba(self, X: csr_matrix) -> np.ndarray:
        check_is_fitted(
            self,
            ("model_", "classes_", "n_features_in_", "runtime_device_"),
        )
        matrix = self._validate_X(X)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("Dataset B PlainMLP input width differs from the fitted model.")
        self._assert_model_runtime_state()
        probability_batches: list[np.ndarray] = []
        self.model_.eval()
        with torch.no_grad():
            for start in range(0, matrix.shape[0], self.batch_size):
                batch = matrix[start : start + self.batch_size]
                positive = torch.sigmoid(self.model_(self._dense_tensor(batch)))
                if positive.dtype != LOCKED_DTYPE or positive.device != self.runtime_device_:
                    raise RuntimeError(
                        "Dataset B PlainMLP probability tensor dtype/device changed."
                    )
                both = torch.stack((torch.ones_like(positive) - positive, positive), dim=1)
                probability_batches.append(both.cpu().numpy())
        probabilities = np.concatenate(probability_batches, axis=0)
        return self._validate_probabilities(probabilities)

    def predict(self, X: csr_matrix) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int8)

    def checkpoint_payload(self) -> dict[str, Any]:
        """Return the locked PMLP-F02 checkpoint payload."""
        check_is_fitted(
            self,
            ("model_", "classes_", "n_features_in_", "runtime_device_", "pos_weight_"),
        )
        self._assert_model_runtime_state()
        state_dict = {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.model_.state_dict().items()
        }
        if any(tensor.dtype != LOCKED_DTYPE for tensor in state_dict.values()):
            raise RuntimeError("Dataset B PlainMLP checkpoint state is not float64.")
        return {
            "payload_version": 2,
            "model_state_dict": state_dict,
            "schema": {
                "raw_feature_count": EXPECTED_RAW_FEATURE_COUNT,
                "semantic_feature_count": EXPECTED_SEMANTIC_FEATURE_COUNT,
                "encoded_feature_count": EXPECTED_INPUT_WIDTH,
                "input_width": self.n_features_in_,
                "classes": self.classes_.tolist(),
            },
            "runtime_policy": {
                "requested_device": LOCKED_DEVICE,
                "device_index": 0,
                "dtype": "float64",
                "allow_cpu_fallback": False,
            },
            "model_policy": {
                "hidden_layers": list(LOCKED_HIDDEN_LAYERS),
                "activation": "ReLU",
                "loss": "BCEWithLogitsLoss",
                "positive_class": "approved",
                "pos_weight_strategy": "fit_population_n_denied_over_n_approved",
                "pos_weight": self.pos_weight_,
            },
            "configuration": self.get_params(deep=False),
            "training_history": list(self.training_history_),
        }

    @classmethod
    def from_checkpoint_payload(cls, payload: dict[str, Any]) -> PlainMLPClassifier:
        """Rehydrate the locked float64 model without a CPU fallback."""
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
            raise ValueError("Invalid Dataset B PlainMLP checkpoint payload.")
        if payload["payload_version"] != 2:
            raise ValueError("Invalid Dataset B PlainMLP checkpoint version.")
        configuration = payload["configuration"]
        if not isinstance(configuration, dict):
            raise ValueError("Invalid Dataset B PlainMLP checkpoint configuration.")
        estimator = cls(**configuration)
        estimator._validate_hyperparameters()

        schema = payload["schema"]
        expected_schema = {
            "raw_feature_count": EXPECTED_RAW_FEATURE_COUNT,
            "semantic_feature_count": EXPECTED_SEMANTIC_FEATURE_COUNT,
            "encoded_feature_count": EXPECTED_INPUT_WIDTH,
            "input_width": EXPECTED_INPUT_WIDTH,
            "classes": [0, 1],
        }
        if schema != expected_schema:
            raise ValueError("Invalid Dataset B PlainMLP checkpoint schema.")
        if payload["runtime_policy"] != {
            "requested_device": LOCKED_DEVICE,
            "device_index": 0,
            "dtype": "float64",
            "allow_cpu_fallback": False,
        }:
            raise ValueError("Invalid Dataset B PlainMLP checkpoint runtime policy.")
        model_policy = payload["model_policy"]
        if not isinstance(model_policy, dict) or model_policy.get("hidden_layers") != list(
            LOCKED_HIDDEN_LAYERS
        ):
            raise ValueError("Invalid Dataset B PlainMLP checkpoint model policy.")
        if (
            model_policy.get("activation") != "ReLU"
            or model_policy.get("loss") != "BCEWithLogitsLoss"
            or model_policy.get("positive_class") != "approved"
            or model_policy.get("pos_weight_strategy")
            != "fit_population_n_denied_over_n_approved"
        ):
            raise ValueError("Invalid Dataset B PlainMLP checkpoint model policy.")
        pos_weight = model_policy.get("pos_weight")
        if (
            isinstance(pos_weight, bool)
            or not isinstance(pos_weight, (int, float))
            or not np.isfinite(pos_weight)
            or pos_weight <= 0
        ):
            raise ValueError("Invalid Dataset B PlainMLP checkpoint pos_weight.")
        state_dict = payload["model_state_dict"]
        if not isinstance(state_dict, dict) or not state_dict:
            raise ValueError("Invalid Dataset B PlainMLP checkpoint state.")
        if any(
            not isinstance(tensor, torch.Tensor) or tensor.dtype != LOCKED_DTYPE
            for tensor in state_dict.values()
        ):
            raise ValueError("Dataset B PlainMLP checkpoint state must be float64.")

        runtime_device = estimator._validated_device()
        estimator.runtime_device_ = runtime_device
        estimator.n_features_in_ = EXPECTED_INPUT_WIDTH
        estimator.classes_ = np.asarray([0, 1], dtype=np.int64)
        estimator.pos_weight_ = float(pos_weight)
        estimator.model_ = _PlainMLPModule(
            EXPECTED_INPUT_WIDTH,
            tuple(estimator.hidden_layers),
            float(estimator.dropout),
        ).to(device=runtime_device, dtype=LOCKED_DTYPE)
        estimator.model_.load_state_dict(state_dict)
        estimator.model_.eval()
        estimator._assert_model_runtime_state()
        history = payload["training_history"]
        if not isinstance(history, list) or not all(
            isinstance(value, (int, float)) and np.isfinite(value) for value in history
        ):
            raise ValueError("Invalid Dataset B PlainMLP checkpoint history.")
        estimator.training_history_ = [float(value) for value in history]
        estimator.last_dtype_audit_ = None
        return estimator

    def _seed_runtime(self) -> None:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        torch.cuda.manual_seed_all(self.random_state)
        torch.use_deterministic_algorithms(self.deterministic_algorithms)
        torch.backends.cudnn.deterministic = self.deterministic_algorithms
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def _validated_device(self) -> torch.device:
        if self.device != LOCKED_DEVICE:
            raise ValueError("plain_mlp_v1 requires locked device cuda:0; CPU fallback is forbidden.")
        if torch.version.cuda is None:
            raise RuntimeError("plain_mlp_v1 requires a CUDA-enabled PyTorch build.")
        if not torch.cuda.is_available():
            raise RuntimeError("plain_mlp_v1 requires CUDA; CPU fallback is forbidden.")
        if torch.cuda.device_count() <= 0:
            raise RuntimeError("plain_mlp_v1 requires available CUDA device index 0.")
        device = torch.device(LOCKED_DEVICE)
        self._cuda_float64_probe(device)
        return device

    @staticmethod
    def _cuda_float64_probe(device: torch.device) -> None:
        try:
            probe = torch.asarray([1.25, 2.75], device=device, dtype=LOCKED_DTYPE)
            result = (probe * probe).sum()
        except (RuntimeError, AssertionError) as exc:
            raise RuntimeError("plain_mlp_v1 CUDA float64 probe failed.") from exc
        if (
            probe.device != device
            or result.device != device
            or probe.dtype != LOCKED_DTYPE
            or result.dtype != LOCKED_DTYPE
            or not bool(torch.isfinite(result).item())
        ):
            raise RuntimeError("plain_mlp_v1 CUDA float64 probe failed.")

    def _dense_tensor(self, X: csr_matrix) -> torch.Tensor:
        dense = X.toarray().astype(np.float64, copy=False)
        tensor = torch.from_numpy(dense).to(
            device=self.runtime_device_, dtype=LOCKED_DTYPE
        )
        if tensor.dtype != LOCKED_DTYPE or tensor.device != self.runtime_device_:
            raise RuntimeError("Dataset B PlainMLP dense batch dtype/device changed.")
        return tensor

    def _target_tensor(self, labels: np.ndarray) -> torch.Tensor:
        target = torch.from_numpy(labels).to(
            device=self.runtime_device_, dtype=LOCKED_DTYPE
        )
        if target.dtype != LOCKED_DTYPE or target.device != self.runtime_device_:
            raise RuntimeError("Dataset B PlainMLP target dtype/device changed.")
        return target

    def _validate_training_dtypes(
        self,
        batch_X: torch.Tensor,
        batch_y: torch.Tensor,
        logits: torch.Tensor,
        loss: torch.Tensor,
    ) -> None:
        tensors = {
            "dense_batch": batch_X,
            "target": batch_y,
            "logits": logits,
            "loss": loss,
        }
        if any(
            tensor.dtype != LOCKED_DTYPE or tensor.device != self.runtime_device_
            for tensor in tensors.values()
        ):
            raise RuntimeError("Dataset B PlainMLP training tensor dtype/device changed.")
        if not all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors.values()):
            raise RuntimeError("Dataset B PlainMLP training tensor is non-finite.")
        self.last_dtype_audit_ = {
            name: str(tensor.dtype).removeprefix("torch.") for name, tensor in tensors.items()
        }

    def _assert_model_runtime_state(self) -> None:
        parameters = tuple(self.model_.parameters())
        if not parameters or any(
            parameter.dtype != LOCKED_DTYPE
            or parameter.device != self.runtime_device_
            for parameter in parameters
        ):
            raise RuntimeError("Dataset B PlainMLP model parameter dtype/device changed.")

    @staticmethod
    def _fit_population_pos_weight(labels: np.ndarray) -> float:
        n_approved = int(np.count_nonzero(labels == 1))
        n_denied = int(np.count_nonzero(labels == 0))
        if n_approved <= 0 or n_denied <= 0:
            raise ValueError("Dataset B PlainMLP requires both binary classes 0 and 1.")
        return float(n_denied / n_approved)

    @staticmethod
    def _validated_labels(y: object, expected_rows: int) -> np.ndarray:
        labels = np.asarray(y)
        if labels.ndim != 1 or len(labels) != expected_rows:
            raise ValueError("Dataset B PlainMLP y must be one-dimensional and align with X.")
        if not np.isin(labels, (0, 1)).all() or set(np.unique(labels).tolist()) != {0, 1}:
            raise ValueError("Dataset B PlainMLP requires both binary classes 0 and 1.")
        return labels.astype(np.float64, copy=False)

    @staticmethod
    def _validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
        if not isinstance(probabilities, np.ndarray) or probabilities.dtype != np.float64:
            raise ValueError("Dataset B PlainMLP probabilities must be float64.")
        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise ValueError("Dataset B PlainMLP probabilities must have shape (n_rows, 2).")
        if not np.isfinite(probabilities).all():
            raise ValueError("Dataset B PlainMLP probabilities contain NaN or infinity.")
        if (probabilities < 0).any() or (probabilities > 1).any():
            raise ValueError("Dataset B PlainMLP probabilities fall outside [0, 1].")
        if not np.allclose(
            probabilities.sum(axis=1),
            1.0,
            rtol=PROBABILITY_ROW_SUM_RTOL,
            atol=PROBABILITY_ROW_SUM_ATOL,
        ):
            raise ValueError("Dataset B PlainMLP probability rows do not sum to one.")
        return probabilities

    @staticmethod
    def _validate_X(X: csr_matrix) -> csr_matrix:
        if not isspmatrix_csr(X) or X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("Dataset B PlainMLP expects a non-empty CSR matrix.")
        if X.shape[1] != EXPECTED_INPUT_WIDTH:
            raise ValueError("Dataset B PlainMLP expects exactly 62 encoded columns.")
        if X.dtype != np.float64:
            raise ValueError("Dataset B PlainMLP expects CSR float64 input.")
        if not np.isfinite(X.data).all():
            raise ValueError("Dataset B PlainMLP input contains NaN or infinity.")
        return X

    def _validate_hyperparameters(self) -> None:
        if self.device != LOCKED_DEVICE:
            raise ValueError("plain_mlp_v1 requires locked device cuda:0; CPU fallback is forbidden.")
        if tuple(self.hidden_layers) != LOCKED_HIDDEN_LAYERS:
            raise ValueError("plain_mlp_v1 requires hidden_layers=(256, 128, 64).")
        if self.activation != "ReLU":
            raise ValueError("plain_mlp_v1 supports the locked ReLU activation only.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer.")
        if (
            isinstance(self.epochs, bool)
            or not isinstance(self.epochs, int)
            or self.epochs <= 0
        ):
            raise ValueError("epochs must be a positive integer.")
