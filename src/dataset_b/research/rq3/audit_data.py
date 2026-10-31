"""Canonical Read-only input and audit-view gates for Dataset B RQ3."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from src.utils.path import PROJECT_ROOT


DEFAULT_PROTOCOL_PATH = PROJECT_ROOT / "config" / "dataset_b" / "rq3_protocol.yaml"
RECORD_ID_PATTERN = re.compile(r"^dataset_b-(\d{9})$")


class RQ3InputError(RuntimeError):
    """Raised when a locked RQ3 input boundary is violated."""


@dataclass(frozen=True)
class ModelAuditInput:
    model_name: str
    run_id: str
    threshold: float
    predictions: pd.DataFrame


@dataclass(frozen=True)
class RQ3PreflightResult:
    audit_metadata: pd.DataFrame
    models: dict[str, ModelAuditInput]
    bootstrap_indices: np.ndarray


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RQ3InputError(message)


def load_protocol(path: Path = DEFAULT_PROTOCOL_PATH) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "RQ3 protocol must be a mapping.")
    return payload


def source_ordinals(record_ids: pd.Series) -> np.ndarray:
    ordinals: list[int] = []
    for record_id in record_ids.astype(str):
        match = RECORD_ID_PATTERN.fullmatch(record_id)
        _require(match is not None, f"Invalid Dataset B record_id: {record_id}")
        source_row_number = int(match.group(1))
        _require(source_row_number >= 1, f"Dataset B record_id must encode a positive source-row number: {record_id}")
        ordinals.append(source_row_number - 1)
    return np.asarray(ordinals, dtype=np.int64)


def _sex_mapping(protocol: Mapping[str, Any]) -> dict[str, str]:
    sex = protocol["attributes"]["applicant_sex"]
    mapping = {str(code): str(label) for label, code in sex["primary_groups"].items()}
    mapping.update(
        {str(code): str(label) for label, code in sex["coverage_only_groups"].items()}
    )
    return mapping


def build_audit_metadata(
    protocol: Mapping[str, Any], *, project_root: Path = PROJECT_ROOT
) -> pd.DataFrame:
    audit = protocol["audit_data"]
    test_path = project_root / audit["test_view"]
    raw_path = project_root / audit["raw_source"]
    test = pd.read_csv(
        test_path,
        usecols=["record_id", "lei", "action_taken", "y_approved", "applicant_sex"],
        dtype={
            "record_id": str,
            "lei": str,
            "action_taken": str,
            "y_approved": np.int8,
            "applicant_sex": str,
        },
        keep_default_na=False,
    )
    expected_rows = int(audit["test_rows"])
    _require(len(test) == expected_rows, "Controlled Test row count differs from protocol.")
    _require(test["record_id"].is_unique, "Controlled Test record_id must be unique.")
    _require(test["y_approved"].isin([0, 1]).all(), "Test y_approved must be binary.")

    ordinals = source_ordinals(test["record_id"])
    raw_metadata = pd.read_csv(
        raw_path,
        usecols=["lei", "action_taken", "applicant_sex", "derived_race"],
        dtype={
            "lei": str,
            "action_taken": str,
            "applicant_sex": str,
            "derived_race": str,
        },
        keep_default_na=False,
    )
    _require(np.all(ordinals >= 0), "record_id source ordinals must be non-negative.")
    _require(np.all(ordinals < len(raw_metadata)), "record_id source ordinal is outside raw source.")
    selected_raw = raw_metadata.iloc[ordinals].reset_index(drop=True)
    for field in ("lei", "action_taken", "applicant_sex"):
        _require(
            selected_raw[field].astype(str).tolist() == test[field].astype(str).tolist(),
            f"Raw-source {field} does not reconcile with the controlled Test view.",
        )
    derived_race = selected_raw["derived_race"]

    sex_mapping = _sex_mapping(protocol)
    sex_group = test["applicant_sex"].map(sex_mapping)
    _require(sex_group.notna().all(), "Controlled Test contains an unmapped applicant_sex code.")

    allowed_race = {
        str(protocol["attributes"]["derived_race"]["reference_group"]),
        *map(str, protocol["attributes"]["derived_race"]["primary_comparisons"]),
        *map(str, protocol["attributes"]["derived_race"]["secondary_comparisons"]),
        *map(str, protocol["attributes"]["derived_race"]["coverage_only_groups"]),
    }
    _require(set(derived_race.unique()) <= allowed_race, "Raw source contains an unmapped derived_race group.")

    output = pd.DataFrame(
        {
            "record_id": test["record_id"],
            "y_approved": test["y_approved"].astype(np.int8),
            "applicant_sex_code": test["applicant_sex"].astype(str),
            "applicant_sex_group": sex_group.astype(str),
            "derived_race": derived_race,
        }
    )
    _require(len(output) == expected_rows, "RQ3 audit join is incomplete.")
    _require(output["record_id"].is_unique, "RQ3 audit join is not one-to-one.")
    return output


def _read_threshold(path: Path, expected_run_id: str) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("run_id") == expected_run_id, "Threshold run_id differs from protocol.")
    threshold = float(payload["selected_threshold"])
    _require(np.isfinite(threshold) and 0.0 <= threshold <= 1.0, "Threshold must be finite within [0, 1].")
    return threshold


def load_model_audit_input(
    model_name: str,
    specification: Mapping[str, Any],
    audit_metadata: pd.DataFrame,
    *,
    project_root: Path = PROJECT_ROOT,
) -> ModelAuditInput:
    prediction_path = project_root / specification["predictions"]
    threshold_path = project_root / specification["threshold"]
    predictions = pd.read_csv(prediction_path)
    required = {"record_id", "y_approved", "p_approved"}
    _require(required <= set(predictions.columns), f"{model_name} prediction schema is incomplete.")
    _require(len(predictions) == len(audit_metadata), f"{model_name} prediction row count differs from Test.")
    _require(predictions["record_id"].astype(str).tolist() == audit_metadata["record_id"].tolist(), f"{model_name} ordered record_id differs from Test.")
    _require(np.array_equal(predictions["y_approved"].to_numpy(), audit_metadata["y_approved"].to_numpy()), f"{model_name} y_approved differs from Test.")
    probabilities = predictions["p_approved"].to_numpy(dtype=np.float64)
    _require(np.isfinite(probabilities).all(), f"{model_name} p_approved contains non-finite values.")
    _require(np.all((probabilities >= 0.0) & (probabilities <= 1.0)), f"{model_name} p_approved is outside [0, 1].")

    run_id = str(specification["run_id"])
    threshold = _read_threshold(threshold_path, run_id)
    derived_class = (probabilities >= threshold).astype(np.int8)
    if "predicted_approved" in predictions.columns:
        stored_class = predictions["predicted_approved"].to_numpy(dtype=np.int8)
        _require(np.array_equal(stored_class, derived_class), f"{model_name} predicted_approved differs from its frozen threshold rule.")
    output = pd.DataFrame(
        {
            "record_id": predictions["record_id"].astype(str),
            "y_approved": predictions["y_approved"].astype(np.int8),
            "p_approved": probabilities,
            "predicted_approved": derived_class,
        }
    )
    return ModelAuditInput(model_name, run_id, threshold, output)


def load_bootstrap_indices(
    protocol: Mapping[str, Any], *, project_root: Path = PROJECT_ROOT
) -> np.ndarray:
    path = project_root / protocol["uncertainty"]["indices"]
    with np.load(path, allow_pickle=False) as payload:
        _require(payload.files == ["indices"], "Bootstrap artifact must contain only the indices array.")
        indices = payload["indices"]
    expected = (int(protocol["uncertainty"]["replicates"]), int(protocol["audit_data"]["test_rows"]))
    _require(indices.shape == expected, "Bootstrap indices shape differs from protocol.")
    _require(np.issubdtype(indices.dtype, np.integer), "Bootstrap indices must use an integer dtype.")
    _require(indices.min() >= 0 and indices.max() < expected[1], "Bootstrap indices are outside the Test row range.")
    return indices


def run_preflight(
    protocol: Mapping[str, Any] | None = None, *, project_root: Path = PROJECT_ROOT
) -> RQ3PreflightResult:
    active = load_protocol() if protocol is None else dict(protocol)
    audit_metadata = build_audit_metadata(active, project_root=project_root)
    models = {
        model_name: load_model_audit_input(
            model_name, specification, audit_metadata, project_root=project_root
        )
        for model_name, specification in active["eligible_final_runs"].items()
    }
    bootstrap_indices = load_bootstrap_indices(active, project_root=project_root)
    return RQ3PreflightResult(audit_metadata, models, bootstrap_indices)
