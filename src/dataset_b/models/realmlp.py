"""Sklearn-compatible Dataset B wrapper for PyTabKit RealMLP-TD."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from src.dataset_b.preprocessing.categorical_encoder import (
    CATEGORICAL_COLUMNS,
    CATEGORICAL_SCHEMA,
    NUMERIC_VALUE_COLUMNS,
    TWO_CHANNEL_COLUMNS,
)
from src.utils.path import CONFIG_DIR


PYTABKIT_VERSION = "1.7.3"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "dataset_b" / "realmlp_v1.yaml"
_BACKEND_CLASS = (
    "pytabkit.models.sklearn.sklearn_interfaces.RealMLP_TD_Classifier"
)
_FROZEN_ESTIMATOR_PARAMETERS = {
    "device": "cuda",
    "random_state": 42,
    "n_cv": 1,
    "n_refit": 0,
    "n_repeats": 1,
    "n_ens": 1,
    "n_epochs": 256,
    "batch_size": 256,
    "predict_batch_size": 1024,
    "hidden_sizes": [256, 256, 256],
    "act": "selu",
    "use_parametric_act": True,
    "opt": "adam",
    "lr": 0.04,
    "lr_sched": "coslog4",
    "wd": 0.02,
    "wd_sched": "flat_cos",
    "p_drop": 0.15,
    "p_drop_sched": "flat_cos",
    "val_metric_name": "cross_entropy",
    "use_ls": False,
    "use_early_stopping": False,
    "verbosity": 2,
}
_FROZEN_BACKEND_CONTROLS = {
    "val_fraction": 0.0,
    "categorical_columns": "explicit_by_name",
}
_FROZEN_PROJECT_POLICY = {
    "allow_silent_cpu_fallback": False,
    "class_weighting": False,
    "sample_weighting": False,
}
_FROZEN_INPUT_CONTRACT = {
    "preprocessing_version": "dataset_b_neural_preprocessing_v1",
    "representation": "realmlp_native_19_columns",
    "raw_feature_count": 15,
    "semantic_feature_count": 19,
    "numerical_dtype": "float32",
    "categorical_dtype": "category",
    "external_scaler": False,
    "external_ohe": False,
}


class RealMLPConfigurationError(ValueError):
    """Raised when the locked RealMLP configuration or input contract is invalid."""


def load_realmlp_v1_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the final 15-feature RealMLP configuration."""
    path = Path(config_path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise RealMLPConfigurationError(f"Cannot read RealMLP configuration: {path}") from exc

    if not isinstance(payload, dict):
        raise RealMLPConfigurationError("RealMLP configuration must be a YAML mapping.")
    expected_header = {
        "schema_version": 2,
        "dataset": "dataset_b",
        "model": "realmlp",
        "model_version": "realmlp_v1",
        "status": "final_locked_configuration",
        "is_final_configuration": True,
        "hyperparameter_search": False,
    }
    if any(payload.get(key) != value for key, value in expected_header.items()):
        raise RealMLPConfigurationError("RealMLP configuration header is invalid.")
    expected_backend = {
        "package": "pytabkit",
        "package_version": PYTABKIT_VERSION,
        "estimator_class": _BACKEND_CLASS,
    }
    if payload.get("backend") != expected_backend:
        raise RealMLPConfigurationError("RealMLP backend contract is invalid.")
    if payload.get("estimator_parameters") != _FROZEN_ESTIMATOR_PARAMETERS:
        raise RealMLPConfigurationError(
            "RealMLP estimator parameters differ from the final locked contract."
        )
    if payload.get("backend_controls") != _FROZEN_BACKEND_CONTROLS:
        raise RealMLPConfigurationError("RealMLP backend split controls are invalid.")
    if payload.get("project_policy") != _FROZEN_PROJECT_POLICY:
        raise RealMLPConfigurationError("RealMLP project policy is invalid.")
    if payload.get("input_contract") != _FROZEN_INPUT_CONTRACT:
        raise RealMLPConfigurationError("RealMLP native input contract is invalid.")
    contracts = payload.get("contracts")
    if not isinstance(contracts, dict) or contracts.get("raw_feature_count") != 15 or contracts.get("semantic_feature_count") != 19:
        raise RealMLPConfigurationError("RealMLP final 15/19 feature contract is invalid.")
    run_policy = payload.get("run_policy")
    if not isinstance(run_policy, dict) or run_policy.get("allow_overwrite") is not False:
        raise RealMLPConfigurationError("RealMLP final run policy is invalid.")
    expected_keys = {
        *expected_header,
        "contracts",
        "backend",
        "estimator_parameters",
        "backend_controls",
        "project_policy",
        "input_contract",
        "pilot",
        "threshold",
        "bootstrap",
        "run_policy",
    }
    if set(payload) != expected_keys:
        raise RealMLPConfigurationError("RealMLP configuration contains unexpected fields.")
    return payload


class RealMLPClassifier(ClassifierMixin, BaseEstimator):
    """Dataset B binary classifier delegating native input to RealMLP-TD.

    The wrapper never creates a validation split. An explicit pilot inner-stop
    set may be supplied through ``X_val`` and ``y_val``; otherwise PyTabKit is
    configured with ``val_fraction=0.0``.
    """

    def __init__(
        self,
        *,
        device: str = "cuda",
        random_state: int = 42,
        n_cv: int = 1,
        n_refit: int = 0,
        n_repeats: int = 1,
        n_ens: int = 1,
        n_epochs: int = 256,
        batch_size: int = 256,
        predict_batch_size: int = 1024,
        hidden_sizes: Sequence[int] = (256, 256, 256),
        act: str = "selu",
        use_parametric_act: bool = True,
        opt: str = "adam",
        lr: float = 0.04,
        lr_sched: str = "coslog4",
        wd: float = 0.02,
        wd_sched: str = "flat_cos",
        p_drop: float = 0.15,
        p_drop_sched: str = "flat_cos",
        val_metric_name: str = "cross_entropy",
        use_ls: bool = False,
        use_early_stopping: bool = False,
        use_best_epoch: bool = True,
        verbosity: int = 2,
        allow_silent_cpu_fallback: bool = False,
        class_weighting: bool = False,
        sample_weighting: bool = False,
    ) -> None:
        self.device = device
        self.random_state = random_state
        self.n_cv = n_cv
        self.n_refit = n_refit
        self.n_repeats = n_repeats
        self.n_ens = n_ens
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.predict_batch_size = predict_batch_size
        self.hidden_sizes = hidden_sizes
        self.act = act
        self.use_parametric_act = use_parametric_act
        self.opt = opt
        self.lr = lr
        self.lr_sched = lr_sched
        self.wd = wd
        self.wd_sched = wd_sched
        self.p_drop = p_drop
        self.p_drop_sched = p_drop_sched
        self.val_metric_name = val_metric_name
        self.use_ls = use_ls
        self.use_early_stopping = use_early_stopping
        self.use_best_epoch = use_best_epoch
        self.verbosity = verbosity
        self.allow_silent_cpu_fallback = allow_silent_cpu_fallback
        self.class_weighting = class_weighting
        self.sample_weighting = sample_weighting

    @classmethod
    def from_yaml(cls, config_path: str | Path = DEFAULT_CONFIG_PATH) -> "RealMLPClassifier":
        """Construct the final RealMLP estimator template from YAML."""
        payload = load_realmlp_v1_config(config_path)
        return cls(**payload["estimator_parameters"], **payload["project_policy"])

    def fit(
        self,
        X: pd.DataFrame,
        y: object,
        *,
        X_val: pd.DataFrame | None = None,
        y_val: object | None = None,
    ) -> "RealMLPClassifier":
        self._validate_hyperparameters()
        train_frame = self._validate_X(X)
        train_labels = self._validate_y(y, len(train_frame), require_both_classes=True)
        if (X_val is None) != (y_val is None):
            raise ValueError("Dataset B RealMLP X_val and y_val must be supplied together.")

        validation_frame: pd.DataFrame | None = None
        validation_labels: np.ndarray | None = None
        if X_val is not None:
            validation_frame = self._validate_X(X_val)
            validation_labels = self._validate_y(
                y_val, len(validation_frame), require_both_classes=False
            )

        self._validate_requested_device()
        self.backend_ = self._make_backend()
        fit_kwargs: dict[str, Any] = {
            "cat_col_names": list(CATEGORICAL_COLUMNS),
        }
        if validation_frame is not None:
            fit_kwargs.update(X_val=validation_frame, y_val=validation_labels)
        self.backend_.fit(train_frame, train_labels, **fit_kwargs)

        backend_classes = np.asarray(getattr(self.backend_, "classes_", ()))
        if not np.array_equal(backend_classes, np.asarray([0, 1])):
            raise ValueError("PyTabKit RealMLP must publish classes_ in [0, 1] order.")
        self.classes_ = np.asarray([0, 1], dtype=np.int64)
        self.n_features_in_ = len(TWO_CHANNEL_COLUMNS)
        self.feature_names_in_ = np.asarray(TWO_CHANNEL_COLUMNS, dtype=object)
        self.runtime_metadata_ = {
            "backend_package": "pytabkit",
            "backend_version": version("pytabkit"),
            "backend_class": _BACKEND_CLASS,
            "requested_device": self.device,
            "allow_silent_cpu_fallback": self.allow_silent_cpu_fallback,
            "validation_source": (
                "project_explicit" if validation_frame is not None else "none"
            ),
            "use_best_epoch": self.use_best_epoch,
            "backend_val_fraction": 0.0,
            "categorical_columns": list(CATEGORICAL_COLUMNS),
            "numeric_dtype": "float32",
            "categorical_dtype": "category",
            "train_rows": len(train_frame),
            "validation_rows": 0 if validation_frame is None else len(validation_frame),
        }
        return self

    def _make_backend(self):
        """Construct the pinned backend; runtime adapters may override this factory."""
        backend_class = self._backend_class()
        return backend_class(**self.backend_parameters())

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, ("backend_", "classes_", "n_features_in_"))
        frame = self._validate_X(X)
        probabilities = np.asarray(self.backend_.predict_proba(frame), dtype=np.float64)
        if probabilities.shape != (len(frame), 2):
            raise ValueError("PyTabKit RealMLP predict_proba must return two columns.")
        if not np.isfinite(probabilities).all():
            raise ValueError("PyTabKit RealMLP probabilities contain NaN or infinity.")
        if (probabilities < 0).any() or (probabilities > 1).any():
            raise ValueError("PyTabKit RealMLP probabilities must be in [0, 1].")
        if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-6):
            raise ValueError("PyTabKit RealMLP probability rows must sum to one.")
        return probabilities

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int8)

    def backend_parameters(self) -> dict[str, Any]:
        """Return the traceable PyTabKit constructor parameters."""
        return {
            "device": self.device,
            "random_state": self.random_state,
            "n_cv": self.n_cv,
            "n_refit": self.n_refit,
            "n_repeats": self.n_repeats,
            "n_ens": self.n_ens,
            "val_fraction": 0.0,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "predict_batch_size": self.predict_batch_size,
            "hidden_sizes": list(self.hidden_sizes),
            "act": self.act,
            "use_parametric_act": self.use_parametric_act,
            "opt": self.opt,
            "lr": self.lr,
            "lr_sched": self.lr_sched,
            "wd": self.wd,
            "wd_sched": self.wd_sched,
            "p_drop": self.p_drop,
            "p_drop_sched": self.p_drop_sched,
            "val_metric_name": self.val_metric_name,
            "use_ls": self.use_ls,
            "use_early_stopping": self.use_early_stopping,
            "verbosity": self.verbosity,
        }

    @staticmethod
    def _backend_class():
        try:
            installed_version = version("pytabkit")
        except PackageNotFoundError as exc:
            raise RuntimeError(
                f"Dataset B RealMLP requires pytabkit=={PYTABKIT_VERSION}."
            ) from exc
        if installed_version != PYTABKIT_VERSION:
            raise RuntimeError(
                f"Dataset B RealMLP requires pytabkit=={PYTABKIT_VERSION}; "
                f"found {installed_version}."
            )
        try:
            from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Classifier
        except ImportError as exc:
            raise RuntimeError("Cannot import PyTabKit RealMLP_TD_Classifier.") from exc
        return RealMLP_TD_Classifier

    def _validate_requested_device(self) -> None:
        if not self.device.startswith("cuda"):
            return
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("CUDA was requested but PyTorch is unavailable.") from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested for RealMLP but is unavailable; CPU fallback is prohibited."
            )
        if ":" in self.device:
            device_index = int(self.device.split(":", 1)[1])
            if device_index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"Requested RealMLP CUDA device {self.device} is unavailable."
                )

    @staticmethod
    def _validate_X(X: pd.DataFrame) -> pd.DataFrame:
        if (
            not isinstance(X, pd.DataFrame)
            or X.empty
            or tuple(X.columns) != TWO_CHANNEL_COLUMNS
        ):
            raise ValueError(
                "Dataset B RealMLP expects the non-empty 19-column native DataFrame."
            )
        if any(X[column].dtype != np.dtype("float32") for column in NUMERIC_VALUE_COLUMNS):
            raise ValueError("Dataset B RealMLP numerical columns must be float32.")
        numeric = X.loc[:, NUMERIC_VALUE_COLUMNS].to_numpy(dtype=np.float32)
        if not np.isfinite(numeric).all():
            raise ValueError("Dataset B RealMLP numerical columns contain NaN or infinity.")
        for column in CATEGORICAL_COLUMNS:
            dtype = X[column].dtype
            if not isinstance(dtype, pd.CategoricalDtype):
                raise ValueError("Dataset B RealMLP categorical columns must use category dtype.")
            if tuple(dtype.categories) != tuple(CATEGORICAL_SCHEMA[column]):
                raise ValueError(f"Dataset B RealMLP category schema differs for {column}.")
            if X[column].isna().any():
                raise ValueError(f"Dataset B RealMLP categorical column {column} contains missing codes.")
        return X

    @staticmethod
    def _validate_y(y: object, rows: int, *, require_both_classes: bool) -> np.ndarray:
        labels = np.asarray(y)
        if labels.ndim != 1 or len(labels) != rows:
            raise ValueError("Dataset B RealMLP labels must be one-dimensional and align with X.")
        if any(value not in (0, 1) for value in labels.tolist()):
            raise ValueError("Dataset B RealMLP labels must contain only binary values.")
        if require_both_classes and set(labels.tolist()) != {0, 1}:
            raise ValueError("Dataset B RealMLP fold-train requires both binary classes.")
        return labels.astype(np.int64, copy=False)

    def _validate_hyperparameters(self) -> None:
        if self.device != "cpu" and re.fullmatch(r"cuda(?::\d+)?", self.device) is None:
            raise ValueError(
                "Dataset B RealMLP device must be explicitly cpu, cuda, or cuda:<index>."
            )
        if self.allow_silent_cpu_fallback:
            raise ValueError("Dataset B RealMLP prohibits silent CPU fallback.")
        if not isinstance(self.use_best_epoch, bool):
            raise ValueError("RealMLP use_best_epoch must be boolean.")
        if self.class_weighting or self.sample_weighting:
            raise ValueError("RealMLP pilot prohibits class and sample weighting.")
        if (self.n_cv, self.n_refit, self.n_repeats, self.n_ens) != (1, 0, 1, 1):
            raise ValueError("RealMLP wrapper prohibits internal CV, refit, repeats, or ensembles.")
        for name, value in (
            ("n_epochs", self.n_epochs),
            ("batch_size", self.batch_size),
            ("predict_batch_size", self.predict_batch_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"RealMLP {name} must be a positive integer.")
        if not self.hidden_sizes or any(
            isinstance(width, bool) or not isinstance(width, int) or width <= 0
            for width in self.hidden_sizes
        ):
            raise ValueError("RealMLP hidden_sizes must contain positive integers.")
        if self.lr <= 0 or self.wd < 0 or not 0 <= self.p_drop < 1:
            raise ValueError("RealMLP learning rate, weight decay, or dropout is invalid.")
