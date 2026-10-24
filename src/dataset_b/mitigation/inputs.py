"""Read-only input gates for Dataset B RQ4a Equalized-Odds post-processing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from src.dataset_b.fairness.audit_data import source_ordinals
from src.dataset_b.fairness.metrics import DenominatorRules
from src.utils.path import PROJECT_ROOT


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "dataset_b" / "rq4a_postprocessing.yaml"
DEFAULT_RQ3_PROTOCOL_PATH = PROJECT_ROOT / "config" / "dataset_b" / "rq3_protocol.yaml"

PARTITIONS = ("validation", "test")
MODELS = ("xgboost", "realmlp")
LOCKED_G3 = ("White", "Black or African American", "Asian")
LOCKED_G6 = (
    "White",
    "Black or African American",
    "Asian",
    "American Indian or Alaska Native",
    "2 or more minority races",
    "Joint",
)
LOCKED_POSTPROCESSOR = {
    "constraints": "equalized_odds",
    "objective": "balanced_accuracy_score",
    "grid_size": 1000,
    "flip": False,
    "prefit": True,
    "predict_method": "predict_proba",
}
LOCKED_CONFIGURATIONS = {
    "RQ4a-3": ("g3", "eod3", "eod6", "rq4a3"),
    "RQ4a-6": ("g6", "eod6", "eod3", "rq4a6"),
}
LOCKED_SENSITIVITY_SEEDS = tuple(range(42, 72))
RECONCILIATION_FIELDS = ("lei", "action_taken", "applicant_sex")
LOCKED_SPLIT_VIEWS = {
    "validation": "data/dataset_b/split_v1/validation.csv",
    "test": "data/dataset_b/split_v1/test.csv",
}
LOCKED_CALIBRATION_RESAMPLING = {
    "source": "validation",
    "method": "conditional_bootstrap_with_replacement",
    "strata": ["derived_race", "y_approved"],
    "refit_postprocessor_each_replicate": True,
}
LOCKED_BOOTSTRAP = {
    "test_indices": "artifacts/dataset_b/rq1_test_bootstrap_indices.npz",
    "test_indices_dtype": "int32",
    "reuse_locked_paired_test_indices": True,
    "allow_regenerate_test_indices": False,
    "root_seed_formula": "10000 + replicate_id",
    "independent_child_streams": ["calibration_resampling", "randomized_decisions"],
    "locked_test_indices_consume_new_rng_stream": False,
    "repeated_test_occurrences_receive_independent_randomized_draws": True,
    "randomized_draw_order": "locked_test_index_order_with_decision_child_stream",
    "same_seed_and_order_reproduce_bit_for_bit": True,
}
LOCKED_ADAPTER_RULES = {
    "duplicate_lookup_record_id": "fail",
    "on_unknown_or_missing_query_record_id": "fail",
    "repeated_query_record_id_from_bootstrap": "allowed",
    "preserve_query_multiplicity_and_order": True,
    "synthetic_row_keys_allowed": False,
}
LOCKED_BASELINE_LINEAGE = {
    "rule": "p_approved_ge_frozen_threshold",
    "stored_predicted_approved_must_match_rule": True,
    "stored_predicted_denied_must_equal_1_minus_predicted_approved": True,
    "validated_frame_includes_predicted_approved": True,
    "derive_when_no_stored_hard_class": True,
}
LOCKED_DENIED_METRIC_DEFINITIONS = {
    "positive_class": "denied",
    "precision_none_when_no_predicted_denied": True,
    "recall_none_when_no_actual_denied": True,
    "f1_formula": "2 * true_denied / (actual_denied + predicted_denied)",
    "f1_none_only_when_denominator_zero": True,
}


class RQ4AInputError(RuntimeError):
    """Raised when a locked RQ4a input boundary is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RQ4AInputError(message)


def _strict_equal(actual: Any, expected: Any) -> bool:
    """Equality that also requires matching types, so YAML `1` never passes for `true`."""

    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _strict_equal(a, e) for a, e in zip(actual, expected)
        )
    return type(actual) is type(expected) and actual == expected


def _require_locked(section: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    for key, value in expected.items():
        _require(key in section, f"{label}.{key} is missing from the RQ4a config.")
        _require(_strict_equal(section[key], value), f"{label}.{key} differs from the locked contract value {value!r}.")


@dataclass(frozen=True)
class ConfigurationSpec:
    name: str
    roster_key: str
    roster: tuple[str, ...]
    target_eod: str
    audit_eod: str
    run_id_prefix: str


@dataclass(frozen=True)
class BaselineModelSpec:
    model: str
    run_id: str
    validation_predictions: Path
    test_predictions: Path
    threshold_artifact: Path
    frozen_threshold: float


@dataclass(frozen=True)
class RQ4AContract:
    g3: tuple[str, ...]
    g6: tuple[str, ...]
    rules: DenominatorRules
    postprocessor_params: dict[str, Any]
    configurations: dict[str, ConfigurationSpec]
    baseline_models: dict[str, BaselineModelSpec]
    primary_seed: int
    sensitivity_seeds: tuple[int, ...]
    replicates: int
    minimum_valid_replicates: int
    test_indices_path: Path
    raw_source: Path
    split_views: dict[str, Path]
    expected_rows: dict[str, int]
    reconciliation_fields: tuple[str, ...]


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "RQ4a config must be a mapping.")
    return payload


def validate_config(config: Mapping[str, Any], *, project_root: Path = PROJECT_ROOT) -> RQ4AContract:
    """Validate the locked RQ4a contract and return its typed view."""

    _require(config.get("repository_question") == "RQ4a", "Config is not the RQ4a contract.")
    _require(config.get("dataset") == "dataset_b", "RQ4a config must target dataset_b.")

    groups = config["groups"]
    g3, g6 = tuple(groups["g3"]), tuple(groups["g6"])
    _require(g3 == LOCKED_G3, "G3 roster differs from the locked ordered roster.")
    _require(g6 == LOCKED_G6, "G6 roster differs from the locked ordered roster.")
    _require(groups.get("fixed_roster_across_bootstrap") is True, "Roster must be fixed across bootstrap.")
    _require(groups.get("allow_drop_merge_rename_or_dynamic_shrink") is False, "Dynamic roster changes must be prohibited.")

    denominators = config["denominators"]
    rules = DenominatorRules(
        minimum_group_rows=int(denominators["minimum_group_rows"]),
        minimum_actual_approved=int(denominators["minimum_actual_approved"]),
        minimum_actual_denied=int(denominators["minimum_actual_denied"]),
    )
    _require(rules == DenominatorRules(100, 30, 30), "Denominator gates differ from the accepted RQ3 gates.")

    postprocessor = config["postprocessor"]
    params = {key: postprocessor[key] for key in LOCKED_POSTPROCESSOR}
    _require(params == LOCKED_POSTPROCESSOR, "ThresholdOptimizer parameters differ from the locked contract.")
    _require(postprocessor.get("allow_base_model_retraining") is False, "Base-model retraining must be prohibited.")
    _require(postprocessor.get("allow_base_model_reinference") is False, "Base-model reinference must be prohibited.")

    configurations: dict[str, ConfigurationSpec] = {}
    _require(set(config["configurations"]) == set(LOCKED_CONFIGURATIONS), "RQ4a configurations differ from RQ4a-3/RQ4a-6.")
    for name, (roster_key, target, audit, prefix) in LOCKED_CONFIGURATIONS.items():
        spec = config["configurations"][name]
        _require(spec["group_roster"] == roster_key, f"{name} roster key differs from {roster_key}.")
        _require(spec["target_eod"] == target and spec["audit_eod"] == audit, f"{name} target/audit EOD differs.")
        _require(spec["outside_roster_decision"] == "frozen_baseline_hard_prediction", f"{name} outside-roster rule differs.")
        _require(spec["run_id_pattern"] == f"{prefix}_postproc_<YYYYMMDD>_<NN>", f"{name} run-ID pattern differs.")
        configurations[name] = ConfigurationSpec(name, roster_key, g3 if roster_key == "g3" else g6, target, audit, prefix)

    decisions = config["randomized_decisions"]
    _require(int(decisions["primary_seed"]) == 42, "Primary randomized-decision seed must be 42.")
    _require(tuple(decisions["sensitivity_seeds"]) == LOCKED_SENSITIVITY_SEEDS, "Sensitivity seeds must be 42 through 71.")
    _require(decisions.get("select_best_seed") is False, "Best-seed selection must be prohibited.")

    bootstrap = config["bootstrap"]
    ci = bootstrap["confidence_interval"]
    _require(int(bootstrap["replicates"]) == 1000, "RQ4a bootstrap must use 1,000 replicates.")
    _require(int(bootstrap["minimum_valid_replicates"]) == 950, "Minimum valid replicates must be 950.")
    _require(
        (ci["confidence_level"], ci["method"], ci["quantile_method"]) == (0.95, "percentile", "linear"),
        "Bootstrap CI semantics differ from the locked contract.",
    )
    _require_locked(bootstrap, LOCKED_BOOTSTRAP, "bootstrap")
    _require(isinstance(bootstrap.get("calibration_resampling"), Mapping), "bootstrap.calibration_resampling is missing.")
    _require_locked(bootstrap["calibration_resampling"], LOCKED_CALIBRATION_RESAMPLING, "bootstrap.calibration_resampling")
    _require_locked(config["cached_probability_adapter"], LOCKED_ADAPTER_RULES, "cached_probability_adapter")
    _require_locked(config["baseline_lineage"], LOCKED_BASELINE_LINEAGE, "baseline_lineage")
    metrics = config["metrics"]
    _require(isinstance(metrics.get("denied_metric_definitions"), Mapping), "metrics.denied_metric_definitions is missing.")
    _require_locked(metrics["denied_metric_definitions"], LOCKED_DENIED_METRIC_DEFINITIONS, "metrics.denied_metric_definitions")
    _require(
        "empty_group_change_rate" in metrics and metrics["empty_group_change_rate"] is None,
        "metrics.empty_group_change_rate must be null.",
    )

    baseline_models: dict[str, BaselineModelSpec] = {}
    _require(set(config["baseline_models"]) == set(MODELS), "Baseline models must be exactly XGBoost and RealMLP.")
    for model in MODELS:
        spec = config["baseline_models"][model]
        threshold = float(spec["frozen_threshold"])
        _require(np.isfinite(threshold) and 0.0 <= threshold <= 1.0, f"{model} frozen threshold is invalid.")
        baseline_models[model] = BaselineModelSpec(
            model=model,
            run_id=str(spec["run_id"]),
            validation_predictions=project_root / spec["validation_predictions"],
            test_predictions=project_root / spec["test_predictions"],
            threshold_artifact=project_root / spec["threshold_artifact"],
            frozen_threshold=threshold,
        )

    audit = config["audit_join"]
    _require(tuple(audit["partitions"]) == PARTITIONS, "Audit join must cover Validation and Test.")
    fields = tuple(audit["reconciliation_fields"])
    _require(fields == RECONCILIATION_FIELDS, "Audit reconciliation fields differ from lei/action_taken/applicant_sex.")
    views = audit.get("controlled_split_views") or {}
    _require(isinstance(views, Mapping) and tuple(views) == PARTITIONS, "Controlled split views must be named explicitly for Validation and Test.")
    _require_locked(views, LOCKED_SPLIT_VIEWS, "audit_join.controlled_split_views")
    _require(_strict_equal(audit.get("derive_split_view_paths_from_test_view"), False), "Split-view paths must not be derived.")
    preflight = config["input_preflight"]

    return RQ4AContract(
        g3=g3,
        g6=g6,
        rules=rules,
        postprocessor_params=params,
        configurations=configurations,
        baseline_models=baseline_models,
        primary_seed=42,
        sensitivity_seeds=LOCKED_SENSITIVITY_SEEDS,
        replicates=1000,
        minimum_valid_replicates=950,
        test_indices_path=project_root / bootstrap["test_indices"],
        raw_source=project_root / audit["raw_source"],
        split_views={partition: project_root / views[partition] for partition in PARTITIONS},
        expected_rows={"validation": int(preflight["validation_rows"]), "test": int(preflight["test_rows"])},
        reconciliation_fields=fields,
    )


def validate_run_id(run_id: str, configuration: ConfigurationSpec) -> str:
    """Accept only `<prefix>_postproc_<YYYYMMDD>_<NN>` with a real date and NN >= 01."""

    match = re.fullmatch(rf"{configuration.run_id_prefix}_postproc_(\d{{8}})_(\d{{2}})", run_id)
    _require(match is not None, f"Run ID {run_id!r} does not match the {configuration.name} pattern.")
    try:
        datetime.strptime(match.group(1), "%Y%m%d")
    except ValueError as exc:
        raise RQ4AInputError(f"Run ID {run_id!r} has an invalid date component.") from exc
    _require(int(match.group(2)) >= 1, f"Run ID {run_id!r} sequence must start at 01.")
    return run_id


def read_frozen_threshold(path: Path, spec: BaselineModelSpec) -> float:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(payload.get("run_id") == spec.run_id, f"{spec.model} threshold run_id differs from the contract.")
    threshold = float(payload["selected_threshold"])
    _require(threshold == spec.frozen_threshold, f"{spec.model} stored threshold differs from the frozen contract value.")
    return threshold


def _stored_hard_class(predictions: pd.DataFrame, column: str, label: str) -> np.ndarray:
    values = pd.to_numeric(predictions[column], errors="coerce")
    _require(values.notna().all() and values.isin([0, 1]).all(), f"{label} stored {column} must be binary.")
    return values.to_numpy(dtype=np.int8)


def validate_frozen_predictions(
    predictions: pd.DataFrame,
    *,
    model: str,
    partition: str,
    expected_record_ids: Sequence[str],
    expected_y: Sequence[int],
    threshold: float,
) -> pd.DataFrame:
    """Fail closed on schema, ID order, label, probability, or baseline-lineage violations.

    The frozen hard class is `p_approved >= threshold`; a stored `predicted_approved`
    must match it and a stored `predicted_denied` must equal `1 - predicted_approved`.
    """

    label = f"{model} {partition}"
    _require({"record_id", "y_approved", "p_approved"} <= set(predictions.columns), f"{label} prediction schema is incomplete.")
    ids = predictions["record_id"].astype(str)
    _require(ids.is_unique, f"{label} record_id must be unique.")
    _require(len(ids) == len(expected_record_ids), f"{label} row count differs from the audit view.")
    _require(ids.tolist() == list(expected_record_ids), f"{label} ordered record_id differs from the audit view.")
    y = pd.to_numeric(predictions["y_approved"], errors="coerce")
    _require(y.notna().all() and y.isin([0, 1]).all(), f"{label} y_approved must be binary.")
    _require(np.array_equal(y.to_numpy(dtype=np.int8), np.asarray(expected_y, dtype=np.int8)), f"{label} y_approved differs from the audit view.")
    p = pd.to_numeric(predictions["p_approved"], errors="coerce").to_numpy(dtype=np.float64)
    _require(np.isfinite(p).all(), f"{label} p_approved contains missing or non-finite values.")
    _require(np.all((p >= 0.0) & (p <= 1.0)), f"{label} p_approved is outside [0, 1].")
    _require(np.isfinite(threshold) and 0.0 <= threshold <= 1.0, f"{label} frozen threshold is invalid.")
    predicted_approved = (p >= threshold).astype(np.int8)
    if "predicted_approved" in predictions.columns:
        stored = _stored_hard_class(predictions, "predicted_approved", label)
        _require(np.array_equal(stored, predicted_approved), f"{label} predicted_approved differs from p_approved >= threshold.")
    if "predicted_denied" in predictions.columns:
        stored_denied = _stored_hard_class(predictions, "predicted_denied", label)
        _require(np.array_equal(stored_denied, 1 - predicted_approved), f"{label} predicted_denied differs from 1 - predicted_approved.")
    return pd.DataFrame(
        {
            "record_id": ids.to_numpy(),
            "y_approved": y.to_numpy(dtype=np.int8),
            "p_approved": p,
            "predicted_approved": predicted_approved,
        }
    )


def load_frozen_predictions(
    path: Path,
    *,
    model: str,
    partition: str,
    expected_record_ids: Sequence[str],
    expected_y: Sequence[int],
    threshold: float,
) -> pd.DataFrame:
    """Read stored predictions with round-trip float parsing; no model is loaded."""

    frame = pd.read_csv(path, dtype={"record_id": str}, float_precision="round_trip")
    return validate_frozen_predictions(
        frame,
        model=model,
        partition=partition,
        expected_record_ids=expected_record_ids,
        expected_y=expected_y,
        threshold=threshold,
    )


def sex_mapping(rq3_protocol: Mapping[str, Any]) -> dict[str, str]:
    sex = rq3_protocol["attributes"]["applicant_sex"]
    mapping = {str(code): str(group) for group, code in sex["primary_groups"].items()}
    mapping.update({str(code): str(group) for group, code in sex["coverage_only_groups"].items()})
    return mapping


def allowed_race_groups(rq3_protocol: Mapping[str, Any]) -> set[str]:
    race = rq3_protocol["attributes"]["derived_race"]
    return {
        str(race["reference_group"]),
        *map(str, race["primary_comparisons"]),
        *map(str, race["secondary_comparisons"]),
        *map(str, race["coverage_only_groups"]),
    }


def reconcile_audit_metadata(
    view: pd.DataFrame,
    raw_metadata: pd.DataFrame,
    *,
    partition: str,
    expected_rows: int,
    sex_groups: Mapping[str, str],
    race_groups: set[str],
    reconciliation_fields: Sequence[str] = RECONCILIATION_FIELDS,
) -> pd.DataFrame:
    """Generalized RQ3 audit join for one partition (Validation or Test)."""

    _require(partition in PARTITIONS, f"Unknown partition {partition!r}.")
    _require(len(view) == expected_rows, f"Controlled {partition} row count differs from the contract.")
    _require(view["record_id"].is_unique, f"Controlled {partition} record_id must be unique.")
    _require(view["y_approved"].astype(str).isin(["0", "1"]).all(), f"Controlled {partition} y_approved must be binary.")
    try:
        ordinals = source_ordinals(view["record_id"])
    except RuntimeError as exc:
        raise RQ4AInputError(str(exc)) from exc
    _require(np.all(ordinals < len(raw_metadata)), "record_id source ordinal is outside the raw source.")
    selected = raw_metadata.iloc[ordinals].reset_index(drop=True)
    for field in reconciliation_fields:
        _require(
            selected[field].astype(str).tolist() == view[field].astype(str).tolist(),
            f"Raw-source {field} does not reconcile with the controlled {partition} view.",
        )
    derived_race = selected["derived_race"].astype(str)
    _require(derived_race.ne("").all(), f"Raw source has empty derived_race for {partition} rows.")
    _require(set(derived_race.unique()) <= race_groups, f"Raw source contains an unmapped derived_race group ({partition}).")
    sex_group = view["applicant_sex"].astype(str).map(dict(sex_groups))
    _require(sex_group.notna().all(), f"Controlled {partition} contains an unmapped applicant_sex code.")
    return pd.DataFrame(
        {
            "record_id": view["record_id"].astype(str).to_numpy(),
            "y_approved": view["y_approved"].astype(np.int8).to_numpy(),
            "applicant_sex_code": view["applicant_sex"].astype(str).to_numpy(),
            "applicant_sex_group": sex_group.astype(str).to_numpy(),
            "derived_race": derived_race.to_numpy(),
        }
    )


def split_view_path(partition: str, contract: RQ4AContract) -> Path:
    """Controlled split-view path named explicitly in the RQ4a config (never derived)."""

    _require(partition in PARTITIONS, f"Unknown partition {partition!r}.")
    return contract.split_views[partition]


def read_split_view(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        usecols=["record_id", "lei", "action_taken", "y_approved", "applicant_sex"],
        dtype={"record_id": str, "lei": str, "action_taken": str, "y_approved": str, "applicant_sex": str},
        keep_default_na=False,
    )


def read_raw_audit_source(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        usecols=["lei", "action_taken", "applicant_sex", "derived_race"],
        dtype={"lei": str, "action_taken": str, "applicant_sex": str, "derived_race": str},
        keep_default_na=False,
    )


def validate_bootstrap_indices(indices: np.ndarray, *, replicates: int, test_rows: int) -> np.ndarray:
    _require(indices.shape == (replicates, test_rows), "Locked Test bootstrap indices have an unexpected shape.")
    _require(indices.dtype == np.int32, "Locked Test bootstrap indices must have dtype exactly int32.")
    _require(int(indices.min()) >= 0 and int(indices.max()) < test_rows, "Locked Test bootstrap indices are out of range.")
    return indices


def load_bootstrap_indices(path: Path, *, replicates: int, test_rows: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        _require(payload.files == ["indices"], "Bootstrap artifact must contain only the indices array.")
        indices = payload["indices"]
    return validate_bootstrap_indices(indices, replicates=replicates, test_rows=test_rows)
