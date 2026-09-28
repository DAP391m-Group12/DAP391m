"""A deterministic CPU plain-MLP wrapper for Dataset B CSR float32 inputs."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from scipy.sparse import csr_matrix, isspmatrix_csr
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from torch import nn


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
    """Sklearn-compatible binary MLP that densifies CSR input by batch only."""

    def __init__(
        self,
        *,
        hidden_layers: tuple[int, ...] = (128, 64),
        activation: str = "ReLU",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.1,
        batch_size: int = 256,
        epochs: int = 100,
        random_state: int = 42,
        device: str = "cpu",
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
        labels = np.asarray(y)
        if labels.ndim != 1 or len(labels) != matrix.shape[0]:
            raise ValueError("Dataset B plain MLP y must be one-dimensional and align with X.")
        if set(np.unique(labels).tolist()) != {0, 1}:
            raise ValueError("Dataset B plain MLP requires both binary classes 0 and 1.")
        labels = labels.astype(np.float32, copy=False)

        self._seed_runtime()
        self.n_features_in_ = matrix.shape[1]
        self.classes_ = np.asarray([0, 1], dtype=np.int64)
        self.model_ = _PlainMLPModule(
            self.n_features_in_, tuple(self.hidden_layers), float(self.dropout)
        ).to(self.device)
        self.optimizer_ = torch.optim.AdamW(
            self.model_.parameters(), lr=float(self.learning_rate), weight_decay=float(self.weight_decay)
        )
        criterion = nn.BCEWithLogitsLoss()
        rng = np.random.default_rng(self.random_state)
        self.training_history_ = []

        for _ in range(self.epochs):
            self.model_.train()
            order = rng.permutation(matrix.shape[0])
            total_loss = 0.0
            total_rows = 0
            for start in range(0, matrix.shape[0], self.batch_size):
                positions = order[start:start + self.batch_size]
                batch_X = self._dense_tensor(matrix[positions])
                batch_y = torch.from_numpy(labels[positions]).to(self.device)
                self.optimizer_.zero_grad(set_to_none=True)
                logits = self.model_(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                self.optimizer_.step()
                rows = len(positions)
                total_loss += float(loss.detach().cpu()) * rows
                total_rows += rows
            self.training_history_.append(total_loss / total_rows)
        return self

    def predict_proba(self, X: csr_matrix) -> np.ndarray:
        check_is_fitted(self, ("model_", "classes_", "n_features_in_"))
        matrix = self._validate_X(X)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("Dataset B plain MLP input width differs from the fitted model.")
        positive_batches: list[np.ndarray] = []
        self.model_.eval()
        with torch.no_grad():
            for start in range(0, matrix.shape[0], self.batch_size):
                batch = matrix[start:start + self.batch_size]
                positive = torch.sigmoid(self.model_(self._dense_tensor(batch)))
                positive_batches.append(positive.cpu().numpy())
        positive_probability = np.concatenate(positive_batches).astype(np.float64, copy=False)
        return np.column_stack((1.0 - positive_probability, positive_probability))

    def predict(self, X: csr_matrix) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int8)

    def checkpoint_payload(self) -> dict[str, Any]:
        """Return a state-dict payload suitable for the PMLP-03 fitted bundle."""
        check_is_fitted(self, ("model_", "classes_", "n_features_in_"))
        return {
            "model_state_dict": self.model_.state_dict(),
            "input_width": self.n_features_in_,
            "classes": self.classes_.tolist(),
            "configuration": self.get_params(deep=False),
            "training_history": list(self.training_history_),
        }

    @classmethod
    def from_checkpoint_payload(cls, payload: dict[str, Any]) -> PlainMLPClassifier:
        """Rehydrate a CPU inference wrapper from a trusted PMLP checkpoint payload."""
        required = {
            "model_state_dict", "input_width", "classes", "configuration", "training_history",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("Invalid Dataset B plain MLP checkpoint payload.")
        configuration = payload["configuration"]
        if not isinstance(configuration, dict):
            raise ValueError("Invalid Dataset B plain MLP checkpoint configuration.")
        estimator = cls(**configuration)
        estimator._validate_hyperparameters()
        input_width = payload["input_width"]
        if isinstance(input_width, bool) or not isinstance(input_width, int) or input_width <= 0:
            raise ValueError("Invalid Dataset B plain MLP checkpoint input width.")
        if payload["classes"] != [0, 1]:
            raise ValueError("Invalid Dataset B plain MLP checkpoint classes.")
        estimator.n_features_in_ = input_width
        estimator.classes_ = np.asarray([0, 1], dtype=np.int64)
        estimator.model_ = _PlainMLPModule(
            input_width, tuple(estimator.hidden_layers), float(estimator.dropout)
        ).to(estimator.device)
        estimator.model_.load_state_dict(payload["model_state_dict"])
        estimator.model_.eval()
        estimator.training_history_ = list(payload["training_history"])
        return estimator

    def _seed_runtime(self) -> None:
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        torch.use_deterministic_algorithms(self.deterministic_algorithms)

    def _dense_tensor(self, X: csr_matrix) -> torch.Tensor:
        dense = X.toarray().astype(np.float32, copy=False)
        return torch.from_numpy(dense).to(self.device)

    @staticmethod
    def _validate_X(X: csr_matrix) -> csr_matrix:
        if not isspmatrix_csr(X) or X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("Dataset B plain MLP expects a non-empty CSR matrix.")
        if X.dtype != np.float32:
            raise ValueError("Dataset B plain MLP expects CSR float32 input.")
        if not np.isfinite(X.data).all():
            raise ValueError("Dataset B plain MLP input contains NaN or infinity.")
        return X

    def _validate_hyperparameters(self) -> None:
        if self.device != "cpu":
            raise ValueError("plain_mlp_v1 supports the locked CPU device only.")
        if self.activation != "ReLU":
            raise ValueError("plain_mlp_v1 supports the locked ReLU activation only.")
        if not self.hidden_layers or any(
            isinstance(width, bool) or not isinstance(width, int) or width <= 0
            for width in self.hidden_layers
        ):
            raise ValueError("hidden_layers must contain positive integer widths.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int) or self.epochs <= 0:
            raise ValueError("epochs must be a positive integer.")
