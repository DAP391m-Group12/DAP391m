"""Canonical Locked Dataset B TabMmini model, loss, inference, and bundle contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn

from src.dataset_b.data import (
    CATEGORICAL_COLUMNS,
    NUMERIC_VALUE_COLUMNS,
    TWO_CHANNEL_COLUMNS,
)
from src.dataset_b.models.neural.tabmmini.preprocessing import (
    TabMMiniNativeBatch,
    fixed_tabmmini_category_maps,
    tabmmini_category_cardinalities,
)
from src.utils.path import CONFIG_DIR


TABM_VERSION = "0.0.3"
ENSEMBLE_MEMBERS = 32
OUTPUT_UNITS = 1
DEFAULT_CONFIG_PATH = CONFIG_DIR / "dataset_b" / "tabmmini_v1.yaml"


class TabMMiniConfigurationError(ValueError):
    """Raised when the machine-readable configuration differs from the lock."""


def _locked_config_sections() -> dict[str, Any]:
    return {
        "backend": {
            "package": "tabm",
            "package_version": TABM_VERSION,
            "constructor": "tabm.TabM.make",
        },
        "model_parameters": {
            "arch_type": "tabm-mini",
            "k": 32,
            "n_blocks": 2,
            "d_block": 512,
            "dropout": 0.1,
            "d_out": 1,
            "num_embeddings": None,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 0.002,
            "weight_decay": 0.0003,
        },
    }


def load_tabmmini_v1_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Load and validate the locked TabMmini v1 machine-readable contract."""
    path = Path(config_path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise TabMMiniConfigurationError(
            f"Cannot read TabMmini configuration: {path}"
        ) from exc
    return _validate_tabmmini_v1_config(payload)


def _validate_tabmmini_v1_config(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TabMMiniConfigurationError("TabMmini configuration must be a YAML mapping.")
    expected_header = {
        "schema_version": 1,
        "dataset": "dataset_b",
        "model": "tabmmini",
        "model_version": "tabmmini_v1",
        "status": "final_locked_configuration",
        "is_final_configuration": True,
        "hyperparameter_search": False,
    }
    if any(payload.get(key) != value for key, value in expected_header.items()):
        raise TabMMiniConfigurationError("TabMmini configuration header is invalid.")
    expected_keys = {
        *expected_header,
        "contracts",
        "backend",
        "model_parameters",
        "optimizer",
        "training",
        "input_contract",
        "output_contract",
        "pilot",
        "evaluation",
        "serialization",
        "run_policy",
    }
    if set(payload) != expected_keys:
        raise TabMMiniConfigurationError(
            "TabMmini configuration contains missing or unexpected top-level fields."
        )
    for section, expected in _locked_config_sections().items():
        if payload.get(section) != expected:
            raise TabMMiniConfigurationError(
                f"TabMmini {section} differs from the locked contract."
            )
    contracts = payload.get("contracts", {})
    if (
        contracts.get("raw_feature_count"),
        contracts.get("semantic_feature_count"),
        contracts.get("numeric_feature_count"),
        contracts.get("categorical_feature_count"),
    ) != (15, 19, 4, 15):
        raise TabMMiniConfigurationError("TabMmini 15/19 native feature contract is invalid.")
    input_contract = payload.get("input_contract", {})
    required_input = {
        "numerical_dtype": "float32",
        "categorical_dtype": "int64",
        "vocabulary_source": "authoritative_schema",
        "learn_vocabulary_from_data": False,
        "unknown_category": "fail",
        "external_scaler": False,
        "external_ohe": False,
        "num_embeddings": None,
    }
    if any(input_contract.get(key) != value for key, value in required_input.items()):
        raise TabMMiniConfigurationError("TabMmini native input contract is invalid.")
    output = payload.get("output_contract", {})
    if (
        output.get("logits_shape") != ["B", 32, 1]
        or output.get("expanded_target_shape") != ["B", 32, 1]
        or output.get("member_aggregation") != "mean_probability"
        or output.get("average_logits_before_sigmoid") is not False
    ):
        raise TabMMiniConfigurationError("TabMmini output contract is invalid.")
    training = payload.get("training", {})
    if (
        training.get("member_loss") != "weighted_bce_with_logits"
        or training.get("member_reduction") != "mean_all_elements"
        or training.get("average_logits_before_loss") is not False
        or training.get("batch_fallback_on_confirmed_cuda_oom") != [128, 64]
    ):
        raise TabMMiniConfigurationError("TabMmini training contract is invalid.")
    pilot = payload.get("pilot", {})
    if (
        pilot.get("source") != "train_only"
        or pilot.get("sample_fraction") != 0.20
        or pilot.get("fit_fraction") != 0.80
        or pilot.get("stop_fraction") != 0.20
        or pilot.get("maximum_epochs") != 100
        or pilot.get("selection_metric") != "inner_stop_weighted_bce_loss"
        or pilot.get("boundary_rule")
        != "unresolved_if_best_epoch_is_100_and_curve_is_still_improving"
    ):
        raise TabMMiniConfigurationError("TabMmini pilot contract is invalid.")
    threshold = payload.get("evaluation", {}).get("threshold", {})
    if threshold != {
        "candidates": {"start": 0.001, "stop": 0.999, "step": 0.001, "count": 999},
        "objective": "maximize_balanced_accuracy",
        "tie_break": ["minimum_absolute_distance_to_0.5", "lower_threshold"],
    }:
        raise TabMMiniConfigurationError("TabMmini threshold contract is invalid.")
    serialization = payload.get("serialization", {})
    if (
        serialization.get("state") != "model_state_dict"
        or serialization.get("include_optimizer_state") is not False
        or serialization.get("include_category_maps") is not True
        or serialization.get("include_semantic_schema") is not True
    ):
        raise TabMMiniConfigurationError("TabMmini serialization contract is invalid.")
    return payload


def compute_pos_weight(labels: object) -> torch.Tensor:
    """Compute n_denied / n_approved and reject a one-class fit population."""
    values = torch.as_tensor(labels)
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("TabMmini fit labels must be a non-empty one-dimensional vector.")
    if not torch.all((values == 0) | (values == 1)).item():
        raise ValueError("TabMmini fit labels must contain only 0 and 1.")
    n_approved = int((values == 1).sum().item())
    n_denied = int((values == 0).sum().item())
    if not n_approved or not n_denied:
        raise ValueError("TabMmini fit population must contain both classes.")
    return torch.tensor(n_denied / n_approved, dtype=torch.float32)


def expand_member_targets(targets: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Expand [B] binary targets to the exact member-logit shape [B,32,1]."""
    _validate_logits(logits)
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError("TabMmini targets must have shape [B] and align with logits.")
    if not torch.all((targets == 0) | (targets == 1)).item():
        raise ValueError("TabMmini targets must contain only 0 and 1.")
    return targets.to(device=logits.device, dtype=logits.dtype).view(-1, 1, 1).expand_as(logits)


def member_wise_weighted_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | float,
) -> torch.Tensor:
    """Average unreduced weighted BCE over samples, members, and output."""
    expanded = expand_member_targets(targets, logits)
    weight = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    if weight.numel() != 1 or not torch.isfinite(weight).all().item() or weight.item() <= 0:
        raise ValueError("TabMmini pos_weight must be one finite positive value.")
    element_losses = F.binary_cross_entropy_with_logits(
        logits,
        expanded,
        pos_weight=weight.reshape(1),
        reduction="none",
    )
    if element_losses.shape != logits.shape or not torch.isfinite(element_losses).all().item():
        raise ValueError("TabMmini member-wise BCE produced an invalid loss tensor.")
    loss = element_losses.mean()
    if loss.ndim != 0 or not torch.isfinite(loss).item():
        raise ValueError("TabMmini scalar loss must be finite.")
    return loss


def mean_member_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Apply sigmoid per member, then average probabilities into p_approved."""
    _validate_logits(logits)
    probabilities = torch.sigmoid(logits).squeeze(-1).mean(dim=1)
    if probabilities.shape != (logits.shape[0],):
        raise ValueError("TabMmini p_approved must have shape [B].")
    if not torch.isfinite(probabilities).all().item():
        raise ValueError("TabMmini probabilities contain NaN or infinity.")
    return probabilities


def _validate_logits(logits: torch.Tensor) -> None:
    if logits.ndim != 3 or tuple(logits.shape[1:]) != (ENSEMBLE_MEMBERS, OUTPUT_UNITS):
        raise ValueError("TabMmini logits must have exact shape [B, 32, 1].")
    if logits.dtype != torch.float32 or not torch.isfinite(logits).all().item():
        raise ValueError("TabMmini logits must contain finite float32 values.")


BackendFactory = Callable[..., nn.Module]


def _tabm_make(**kwargs: Any) -> nn.Module:
    try:
        installed = version("tabm")
    except PackageNotFoundError as exc:
        raise RuntimeError(f"Dataset B TabMmini requires tabm=={TABM_VERSION}.") from exc
    if installed != TABM_VERSION:
        raise RuntimeError(
            f"Dataset B TabMmini requires tabm=={TABM_VERSION}; found {installed}."
        )
    try:
        from tabm import TabM
    except ImportError as exc:
        raise RuntimeError("Cannot import tabm.TabM.") from exc
    return TabM.make(**kwargs)


class TabMMiniModel(nn.Module):
    """Thin guarded wrapper around the pinned public ``TabM.make`` interface."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any] | None = None,
        backend_factory: BackendFactory | None = None,
    ) -> None:
        super().__init__()
        payload = load_tabmmini_v1_config() if config is None else dict(config)
        self.config = _validate_tabmmini_v1_config(payload)
        parameters = dict(self.config["model_parameters"])
        make_kwargs = {
            "n_num_features": len(NUMERIC_VALUE_COLUMNS),
            "cat_cardinalities": list(tabmmini_category_cardinalities()),
            **parameters,
        }
        self.backend = (backend_factory or _tabm_make)(**make_kwargs)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        TabMMiniNativeBatch(x_num=x_num, x_cat=x_cat)
        logits = self.backend(x_num, x_cat)
        _validate_logits(logits)
        return logits

    def predict_approved_probability(
        self, x_num: torch.Tensor, x_cat: torch.Tensor
    ) -> torch.Tensor:
        return mean_member_probabilities(self(x_num, x_cat))

    def save_inference_bundle(
        self,
        path: str | Path,
        *,
        preprocessing_state: Mapping[str, Any],
        effective_batch_size: int,
        fixed_epoch: int,
        run_id: str,
        runtime_versions: Mapping[str, str] | None = None,
    ) -> None:
        """Persist inference-only state; optimizer state is intentionally excluded."""
        if not run_id or not isinstance(run_id, str):
            raise ValueError("TabMmini inference bundle requires a non-empty run_id.")
        for name, value in (("effective_batch_size", effective_batch_size), ("fixed_epoch", fixed_epoch)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"TabMmini {name} must be a positive integer.")
        payload = {
            "bundle_version": 1,
            "model_version": "tabmmini_v1",
            "run_id": run_id,
            "configuration": self.config,
            "model_state_dict": self.state_dict(),
            "semantic_schema": list(TWO_CHANNEL_COLUMNS),
            "numeric_columns": list(NUMERIC_VALUE_COLUMNS),
            "categorical_columns": list(CATEGORICAL_COLUMNS),
            "category_maps": fixed_tabmmini_category_maps(),
            "preprocessing_state": dict(preprocessing_state),
            "effective_batch_size": effective_batch_size,
            "fixed_epoch": fixed_epoch,
            "runtime_versions": dict(runtime_versions or {}),
        }
        torch.save(payload, Path(path))

    @classmethod
    def load_inference_bundle(
        cls,
        path: str | Path,
        *,
        backend_factory: BackendFactory | None = None,
        map_location: str | torch.device = "cpu",
    ) -> tuple["TabMMiniModel", dict[str, Any]]:
        """Load and validate an inference-only bundle without optimizer state."""
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        if not isinstance(payload, dict) or payload.get("bundle_version") != 1:
            raise ValueError("Invalid TabMmini inference bundle.")
        if payload.get("model_version") != "tabmmini_v1" or "optimizer_state_dict" in payload:
            raise ValueError("Invalid TabMmini inference bundle contents.")
        if payload.get("semantic_schema") != list(TWO_CHANNEL_COLUMNS):
            raise ValueError("TabMmini inference bundle semantic schema differs.")
        if payload.get("category_maps") != fixed_tabmmini_category_maps():
            raise ValueError("TabMmini inference bundle category maps differ.")
        model = cls(config=payload["configuration"], backend_factory=backend_factory)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        return model, payload
