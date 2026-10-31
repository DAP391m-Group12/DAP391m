"""Canonical Dataset B RQ4a runner: orchestrates the T4 pure core into the locked artifact layout.

The runner performs no base-model fit or inference. It reads frozen `p_approved`
values, fits Fairlearn `ThresholdOptimizer` on the complete Validation partition
and on every stratified Validation bootstrap replicate, reuses the locked paired
Test indices, and writes the YAML-locked files atomically into a new run
directory that is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import fairlearn
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml
from sklearn.metrics import roc_auc_score

from src.dataset_b.research.rq4.rq4a.inputs import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_RQ3_PROTOCOL_PATH,
    MODELS,
    PARTITIONS,
    ConfigurationSpec,
    RQ4AContract,
    allowed_race_groups,
    load_bootstrap_indices,
    load_config,
    load_frozen_predictions,
    read_frozen_threshold,
    read_raw_audit_source,
    read_split_view,
    reconcile_audit_metadata,
    sex_mapping,
    split_view_path,
    validate_config,
    validate_run_id,
)
from src.dataset_b.research.rq4.rq4a.metrics import (
    EOD_INVALID_REASON,
    EODEstimate,
    ba_cost,
    balanced_accuracy,
    compute_fixed_roster_eod,
    compute_roster_group_metrics,
    decision_changes,
    delta_eod,
    denied_metrics,
    extremum_roles,
)
from src.dataset_b.research.rq4.rq4a.postprocessing import (
    CachedProbabilityEstimator,
    apply_policy,
    fit_policy,
    load_policy,
    save_policy,
)
from src.utils.path import PROJECT_ROOT


MODEL_FRAME_COLUMNS = ("record_id", "y_approved", "p_approved", "predicted_approved", "derived_race", "applicant_sex_group")
POLICIES = ("baseline", "mitigated")
SUMMARY_QUANTITIES = ("delta_target_eod", "ba_cost", "delta_audit_eod")
BA_INVALID_REASON = "single_actual_class"
REASON_ORDER = (
    "required_group_metric_invalid",
    "insufficient_valid_bootstrap_replicates",
    "target_eod_reduction_not_demonstrated",
    "ba_budget_exceeded",
    "unstable_diagnostics",
)
RQ3_NAMESPACE = Path("artifacts") / "dataset_b" / "rq3"

# Independent literal copy of the T5-owned contract surface (C2-locked). The YAML
# is compared against these values; outputs are written from these values, so an
# edited YAML can never silently change the artifact layout.
_DECISION_COLUMNS = [
    "run_id", "configuration_id", "model", "record_id", "y_approved", "p_approved", "derived_race",
    "applicant_sex_group", "in_target_roster", "baseline_decision", "mitigated_decision", "decision_changed",
]
LOCKED_OUTPUTS: dict[str, Any] = {
    "run_root": "artifacts/dataset_b/rq4/rq4a/<run_id>",
    "file_layout_status": "locked",
    "column_lists_status": "locked",
    "validation": {
        "fitted_policies_dir": "validation/fitted_policies",
        "one_policy_per_model": True,
        "decisions.csv": {
            "columns": list(_DECISION_COLUMNS),
            "mitigated_decision_seed": 42,
            "purpose": "recompute_validation_eod_and_fairness_generalization_gap_without_rerunning_fairlearn",
        },
    },
    "test_evaluation": {
        "decisions.csv": {
            "columns": list(_DECISION_COLUMNS),
            "mitigated_decision_seed": 42,
            "purpose": "recompute_test_eod_inherited_roc_auc_brier_and_sex_eod_audit",
        },
        "bootstrap_replicates.csv": {
            "columns": [
                "run_id", "configuration_id", "model", "replicate_id", "root_seed", "calibration_stream_id",
                "decision_stream_id", "baseline_ba", "mitigated_ba", "ba_cost", "ba_valid", "baseline_target_eod",
                "mitigated_target_eod", "delta_target_eod", "target_eod_valid", "baseline_audit_eod",
                "mitigated_audit_eod", "delta_audit_eod", "audit_eod_valid", "invalid_reason_codes",
            ],
            "rows_per_model": 1000,
        },
        "status.json": {
            "fields": ["run_id", "configuration_id", "model_statuses"],
            "model_statuses": {
                "keyed_by": "model",
                "models": ["xgboost", "realmlp"],
                "per_model_fields": ["execution_status", "stability_status", "primary_criterion", "reason_codes"],
            },
            "optional_run_level_status": True,
            "run_level_status_replaces_model_statuses": False,
        },
    },
    "reporting": {
        "model_summary.json": {
            "per_model_fields": [
                "execution_status", "stability_status", "primary_criterion", "reason_codes", "point_delta_target_eod",
                "delta_target_eod_ci", "point_ba_cost", "ba_cost_ci", "n_valid_delta_target_eod", "n_valid_ba_cost",
                "ba_budget_sensitivity", "fairness_generalization_gap_eod3", "fairness_generalization_gap_eod6",
                "sensitivity_mitigated_target_eod_min_max_range", "sensitivity_delta_target_eod_min_max_range",
                "sensitivity_mitigated_ba_min_max_range", "sensitivity_ba_cost_min_max_range",
            ],
        },
        "experiment_metrics.csv": {
            "columns": [
                "run_id", "configuration_id", "model", "partition", "policy", "balanced_accuracy", "eod3", "eod3_valid",
                "eod3_tpr_range", "eod3_fpr_range", "eod6", "eod6_valid", "eod6_tpr_range", "eod6_fpr_range",
                "denied_f1", "denied_precision", "denied_recall", "sex_eod", "sex_eod_valid", "sex_eod_tpr_range",
                "sex_eod_fpr_range", "roc_auc_inherited", "brier_inherited",
            ],
        },
        "group_metrics.csv": {
            "columns": [
                "run_id", "configuration_id", "model", "partition", "policy", "attribute", "group", "in_target_roster",
                "n_rows", "n_actual_approved", "n_actual_denied", "tpr", "fpr", "selection_rate", "tpr_valid",
                "fpr_valid", "selection_rate_valid", "tpr_extremum_role", "fpr_extremum_role",
                "selection_rate_extremum_role",
            ],
            "extremum_role_values": ["min", "max", "none", "not_evaluable"],
        },
        "decision_changes.csv": {
            "columns": [
                "run_id", "configuration_id", "model", "scope", "group", "n_rows", "n_changed", "change_rate",
                "approved_to_denied", "denied_to_approved",
            ],
        },
        "randomization_sensitivity.csv": {
            "columns": [
                "run_id", "configuration_id", "model", "seed", "mitigated_ba", "ba_cost", "ba_valid",
                "mitigated_target_eod", "delta_target_eod", "target_eod_valid", "mitigated_audit_eod",
                "delta_audit_eod", "audit_eod_valid", "invalid_reason_codes",
            ],
            "seeds": "42_through_71",
        },
        "bootstrap_summary.csv": {
            "columns": [
                "run_id", "configuration_id", "model", "quantity", "point_estimate", "ci_lower", "ci_upper",
                "confidence_level", "ci_method", "quantile_method", "n_replicates", "n_valid", "minimum_valid",
                "valid_gate_status",
            ],
        },
        "runtime.csv": {
            "columns": ["run_id", "stage", "started_at_utc", "finished_at_utc", "elapsed_seconds", "status"],
        },
        "artifact_index.json": {"per_file_fields": ["relative_path", "sha256", "bytes", "rows"]},
    },
}
LOCKED_STATUS_CONTRACT: dict[str, Any] = {
    "execution_status": ["completed", "failed", "infeasible"],
    "completed_run_status_json_execution_status": ["completed"],
    "failure_receipt_execution_status": ["failed", "infeasible"],
    "failed_and_infeasible_are_attempt_level_only": True,
    "stability_status": ["stable", "unstable", "not_assessed"],
    "primary_criterion": ["pass", "fail", "not_evaluable"],
    "fields_are_independent": True,
    "reason_codes": [
        "none",
        "prerequisite_rq1_paired_bootstrap_missing",
        "required_group_metric_invalid",
        "insufficient_valid_bootstrap_replicates",
        "target_eod_reduction_not_demonstrated",
        "ba_budget_exceeded",
        "unstable_diagnostics",
        "runtime_failure",
        "postprocessing_infeasible",
    ],
}
LOCKED_FAILURE_RECEIPTS: dict[str, Any] = {
    "schema_version": 1,
    "directory": "artifacts/dataset_b/rq4/rq4a/_failures",
    "output_root_override_relative_directory": "_failures",
    "filename_pattern": "<run_id>.json",
    "purpose": "retain_atomic_audit_evidence_for_started_attempts_that_do_not_publish_a_completed_run",
    "attempt_begins_after": [
        "runner_contract_validation",
        "config_validation",
        "prerequisite_validation",
        "run_id_validation",
        "output_collision_validation",
        "production_input_preflight",
        "runner_input_validation",
    ],
    "no_receipt_for_failures_before_attempt_begins": True,
    "completed_run_directory_on_failure": "absent",
    "write_mode": "atomic",
    "overwrite_existing_receipt": False,
    "run_id_consumed_when_published": True,
    "re_raise_after_receipt": True,
    "fields": [
        "schema_version",
        "run_id",
        "configuration_id",
        "execution_status",
        "reason_codes",
        "failed_stage",
        "model",
        "started_at_utc",
        "failed_at_utc",
        "runtime_provenance",
        "exception_type",
        "error_summary",
    ],
    "execution_status": {"runtime_exception": "failed", "explicit_infeasible_exception": "infeasible"},
    "reason_codes": {"runtime_exception": ["runtime_failure"], "explicit_infeasible_exception": ["postprocessing_infeasible"]},
    "infeasible_exception_class": "RQ4AInfeasibleError",
    "never_infer_infeasible_from_generic_exception": True,
    "model_null_when_failure_is_not_model_specific": True,
    "error_summary": {"source": "exception_message", "single_line": True, "maximum_characters": 1000},
    "temporary_filename_pattern": ".<run_id>.tmp-<pid>.json",
    "cleanup_partial_receipt_on_write_failure": True,
}
LOCKED_PROHIBITIONS: dict[str, Any] = {
    "prohibit_base_model_retraining": True,
    "prohibit_base_model_reinference": True,
    "prohibit_locked_test_bootstrap_regeneration": True,
    "prohibit_best_randomized_seed_selection": True,
    "prohibit_dynamic_group_roster": True,
    "prohibit_test_use_for_selection_or_refit": True,
    "prohibit_roc_auc_or_brier_postprocessing_improvement_claim": True,
    "prohibit_legal_compliance_claim": True,
    "prohibit_discrimination_claim": True,
    "prohibit_causal_claim": True,
    "prohibit_creditworthiness_or_default_risk_claim": True,
}
SENSITIVITY_SEEDS = list(range(42, 72))
LOCKED_RUNNER_SCALARS: dict[tuple[str, ...], Any] = {
    ("artifact_namespace",): "artifacts/dataset_b/rq4/rq4a",
    ("prohibit_rq3_artifact_namespace_reuse",): True,
    ("runtime", "record_actual_interpreter_and_imported_versions"): True,
    ("runtime", "policy_persistence"): "joblib_dump_compress_3_with_adapter_in_stable_importable_module",
    ("reporting_contract", "producer"): "python_code_produces_and_validates_artifacts",
    ("reporting_contract", "reporting_write_mode"): "atomic",
    ("randomized_decisions", "primary_seed"): 42,
    ("randomized_decisions", "sensitivity_seeds"): SENSITIVITY_SEEDS,
    ("randomized_decisions", "select_best_seed"): False,
    ("randomized_decisions", "pool_sensitivity_seeds_into_bootstrap_ci"): False,
    ("stability_status", "seeds"): "42_through_71",
    ("stability_status", "required_seed_count"): 30,
    ("stability_status", "reference_seed"): 42,
    ("run_identity", "one_immutable_run_id_per_configuration"): True,
    ("run_identity", "overwrite_existing_run"): False,
    ("run_identity", "models_per_run"): ["xgboost", "realmlp"],
    ("run_identity", "collision_targets"): ["completed_run_directory", "failure_receipt"],
    ("run_identity", "failure_receipt_consumes_run_id"): True,
    ("bootstrap", "save_invalid_replicates"): True,
    ("bootstrap", "save_invalid_reasons"): True,
    ("bootstrap", "invalid_replicate_retention"): "retain_all_rows_with_reason_codes",
    ("bootstrap", "allow_base_model_retraining_or_reinference"): False,
    ("cached_probability_adapter", "load_model_bundle_or_preprocessor"): False,
    ("cached_probability_adapter", "fit_or_train"): False,
    ("cached_probability_adapter", "base_model_inference"): False,
}
PROVENANCE_PACKAGES = ("numpy", "pandas", "scipy", "scikit_learn", "pyyaml", "joblib", "fairlearn")
RUN_ID_PREFIXES = {"RQ4a-3": "rq4a3", "RQ4a-6": "rq4a6"}
FAILURES_DIRNAME = "_failures"
ERROR_SUMMARY_MAX = 1000


class RQ4ARunnerError(RuntimeError):
    """Raised when the runner would break the locked RQ4a orchestration contract."""


class RQ4AInfeasibleError(RuntimeError):
    """Explicit domain signal that post-processing is infeasible for a started attempt.

    Only this exact class yields `execution_status: infeasible`; the runner never
    infers infeasibility from any other exception (including subclasses).
    """


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RQ4ARunnerError(message)


def _first_difference(actual: Any, expected: Any, path: str) -> str | None:
    """Location of the first strict mismatch (exact keys, list order, and scalar types), else None."""

    if isinstance(expected, dict):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            return path
        for key, value in expected.items():
            found = _first_difference(actual[key], value, f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return path
        for position, (a, e) in enumerate(zip(actual, expected)):
            found = _first_difference(a, e, f"{path}[{position}]")
            if found:
                return found
        return None
    return None if type(actual) is type(expected) and actual == expected else path


def validate_runner_contract(config: Mapping[str, Any]) -> None:
    """Reject any change to the T5-owned contract surface against the independent locked copy."""

    for key, expected in (
        ("outputs", LOCKED_OUTPUTS),
        ("status_contract", LOCKED_STATUS_CONTRACT),
        ("failure_receipts", LOCKED_FAILURE_RECEIPTS),
        ("prohibitions", LOCKED_PROHIBITIONS),
    ):
        found = _first_difference(config.get(key), expected, key)
        _require(found is None, f"RQ4a runner contract changed at {found}.")
    for path, expected in LOCKED_RUNNER_SCALARS.items():
        node: Any = config
        for part in path:
            _require(isinstance(node, Mapping) and part in node, f"RQ4a runner contract is missing {'.'.join(path)}.")
            node = node[part]
        found = _first_difference(node, expected, ".".join(path))
        _require(found is None, f"RQ4a runner contract changed at {found}.")


def runtime_provenance() -> dict[str, Any]:
    """Interpreter and imported package versions of the running process."""

    return {
        "sys_executable": sys.executable,
        "python_version": platform.python_version(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "pyyaml": yaml.__version__,
            "joblib": joblib.__version__,
            "fairlearn": fairlearn.__version__,
        },
    }


def validate_runtime_provenance_structure(payload: Any) -> None:
    """Environment-independent check of stored provenance: field names, package set/order, non-empty strings."""

    _require(isinstance(payload, dict) and list(payload) == ["sys_executable", "python_version", "packages"], "Runtime provenance fields differ.")
    packages = payload["packages"]
    _require(isinstance(packages, dict) and tuple(packages) == PROVENANCE_PACKAGES, "Runtime provenance package list differs.")
    values = [payload["sys_executable"], payload["python_version"], *packages.values()]
    _require(all(isinstance(v, str) and v for v in values), "Runtime provenance values must be non-empty strings.")


def verify_producer_provenance(payload: Any) -> None:
    """Producer-time check: stored provenance must equal the interpreter and modules of this process."""

    validate_runtime_provenance_structure(payload)
    _require(payload == runtime_provenance(), "Runtime provenance does not match the producing interpreter and modules.")


# --------------------------------------------------------------------------- inputs


@dataclass(frozen=True)
class ModelData:
    model: str
    threshold: float
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class RunnerInputs:
    config: Mapping[str, Any]
    contract: RQ4AContract
    configuration: ConfigurationSpec
    models: Mapping[str, ModelData]
    bootstrap_indices: np.ndarray
    sex_roster: tuple[str, ...]
    sex_report_groups: tuple[str, ...]


@dataclass(frozen=True)
class RunSettings:
    """Replicate count and valid-replicate gate; production always uses the contract values."""

    replicates: int
    minimum_valid: int

    @classmethod
    def from_contract(cls, contract: RQ4AContract) -> "RunSettings":
        return cls(contract.replicates, contract.minimum_valid_replicates)


def sex_rosters(rq3_protocol: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Sex-EOD roster = RQ3 primary sex groups (reference first); report groups add coverage-only groups."""

    sex = rq3_protocol["attributes"]["applicant_sex"]
    roster = tuple(str(g) for g in sex["primary_groups"])
    return roster, roster + tuple(str(g) for g in sex["coverage_only_groups"])


def race_report_groups(config: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(config["groups"]["g6"]) + tuple(config["groups"]["coverage_only_outside_g6"])


def validate_runner_inputs(inputs: RunnerInputs, settings: RunSettings) -> None:
    """Fail closed on model coverage, ID/label alignment, roster coverage, and index shape."""

    _require(tuple(inputs.models) == MODELS, "Runner models must be exactly XGBoost then RealMLP.")
    _require(1 <= settings.minimum_valid <= settings.replicates, "Minimum valid replicates must be within [1, replicates].")
    reference = inputs.models[MODELS[0]]
    for model, data in inputs.models.items():
        _require(data.model == model, f"Model entry {model} carries data for {data.model}.")
        for partition in PARTITIONS:
            frame = getattr(data, partition)
            _require(tuple(frame.columns) == MODEL_FRAME_COLUMNS, f"{model} {partition} frame columns differ from the runner schema.")
            _require(frame["record_id"].is_unique, f"{model} {partition} record_id must be unique.")
            ref = getattr(reference, partition)
            for column in ("record_id", "y_approved", "derived_race", "applicant_sex_group"):
                _require(
                    frame[column].tolist() == ref[column].tolist(),
                    f"{model} {partition} ordered {column} differs from {reference.model}.",
                )
            base = (frame["p_approved"].to_numpy(dtype=np.float64) >= data.threshold).astype(np.int8)
            _require(np.array_equal(base, frame["predicted_approved"].to_numpy(dtype=np.int8)), f"{model} {partition} baseline lineage differs.")
        overlap = set(data.validation["record_id"]) & set(data.test["record_id"])
        _require(not overlap, f"{model} Validation and Test share record_id values.")
    roster = inputs.configuration.roster
    present = set(reference.validation["derived_race"])
    _require(set(roster) <= present, "Every target-roster group must be present in Validation.")
    allowed = set(race_report_groups(inputs.config))
    for partition in PARTITIONS:
        _require(set(getattr(reference, partition)["derived_race"]) <= allowed, f"{partition} contains a race group outside G6 and coverage-only groups.")
        _require(set(getattr(reference, partition)["applicant_sex_group"]) <= set(inputs.sex_report_groups), f"{partition} contains an unmapped sex group.")
    indices = inputs.bootstrap_indices
    n_test = len(reference.test)
    _require(isinstance(indices, np.ndarray) and indices.dtype == np.int32, "Locked Test indices must be an int32 array.")
    _require(indices.shape == (settings.replicates, n_test), "Locked Test indices shape differs from (replicates, Test rows).")
    _require(int(indices.min()) >= 0 and int(indices.max()) < n_test, "Locked Test indices are out of range.")


def validate_prerequisites(config: Mapping[str, Any]) -> None:
    prerequisites = config.get("prerequisites")
    _require(
        isinstance(prerequisites, Mapping) and prerequisites.get("rq1_paired_bootstrap_task") == "complete",
        "prerequisite_rq1_paired_bootstrap_missing: RQ1 paired bootstrap is not complete.",
    )


def load_production_inputs(
    config: Mapping[str, Any],
    configuration_name: str,
    *,
    project_root: Path = PROJECT_ROOT,
    rq3_protocol_path: Path | None = None,
) -> RunnerInputs:
    """Re-run the input preflight as runtime validation and assemble runner inputs (read-only)."""

    validate_runner_contract(config)
    contract = validate_config(config, project_root=project_root)
    _require(configuration_name in contract.configurations, f"Unknown RQ4a configuration {configuration_name!r}.")
    validate_prerequisites(config)
    rq3_path = Path(rq3_protocol_path) if rq3_protocol_path is not None else project_root / "config" / "dataset_b" / "rq3_protocol.yaml"
    rq3_protocol = yaml.safe_load(rq3_path.read_text(encoding="utf-8"))
    sex_groups = sex_mapping(rq3_protocol)
    race_groups = allowed_race_groups(rq3_protocol)
    raw = read_raw_audit_source(contract.raw_source)
    audits = {
        partition: reconcile_audit_metadata(
            read_split_view(split_view_path(partition, contract)),
            raw,
            partition=partition,
            expected_rows=contract.expected_rows[partition],
            sex_groups=sex_groups,
            race_groups=race_groups,
            reconciliation_fields=contract.reconciliation_fields,
        )
        for partition in PARTITIONS
    }
    del raw
    models: dict[str, ModelData] = {}
    for model in MODELS:
        spec = contract.baseline_models[model]
        threshold = read_frozen_threshold(spec.threshold_artifact, spec)
        frames: dict[str, pd.DataFrame] = {}
        for partition in PARTITIONS:
            audit = audits[partition]
            path = spec.validation_predictions if partition == "validation" else spec.test_predictions
            frozen = load_frozen_predictions(
                path,
                model=model,
                partition=partition,
                expected_record_ids=audit["record_id"].tolist(),
                expected_y=audit["y_approved"].to_numpy(),
                threshold=threshold,
            )
            frames[partition] = pd.DataFrame(
                {
                    "record_id": frozen["record_id"].to_numpy(),
                    "y_approved": frozen["y_approved"].to_numpy(dtype=np.int8),
                    "p_approved": frozen["p_approved"].to_numpy(dtype=np.float64),
                    "predicted_approved": frozen["predicted_approved"].to_numpy(dtype=np.int8),
                    "derived_race": audit["derived_race"].to_numpy(),
                    "applicant_sex_group": audit["applicant_sex_group"].to_numpy(),
                }
            )
        models[model] = ModelData(model, threshold, frames["validation"], frames["test"])
    indices = load_bootstrap_indices(contract.test_indices_path, replicates=contract.replicates, test_rows=contract.expected_rows["test"])
    sex_roster, sex_report = sex_rosters(rq3_protocol)
    inputs = RunnerInputs(config, contract, contract.configurations[configuration_name], models, indices, sex_roster, sex_report)
    validate_runner_inputs(inputs, RunSettings.from_contract(contract))
    return inputs


# --------------------------------------------------------------------------- random streams


@dataclass(frozen=True)
class ReplicateStreams:
    replicate_id: int
    root_seed: int
    calibration_rng: np.random.Generator
    decision_seed: int
    calibration_stream_id: str
    decision_stream_id: str


def replicate_streams(replicate_id: int) -> ReplicateStreams:
    """Root seed 10000 + replicate_id; independent SeedSequence children for resampling and decisions."""

    root = 10000 + int(replicate_id)
    calibration, decision = np.random.SeedSequence(root).spawn(2)
    decision_seed = int(decision.generate_state(1, dtype=np.uint32)[0])
    return ReplicateStreams(
        replicate_id=int(replicate_id),
        root_seed=root,
        calibration_rng=np.random.default_rng(calibration),
        decision_seed=decision_seed,
        calibration_stream_id=f"SeedSequence({root}).spawn(2)[0]->PCG64",
        decision_stream_id=f"SeedSequence({root}).spawn(2)[1]->RandomState({decision_seed})",
    )


def stratified_resample(derived_race: Sequence[str], y_approved: Sequence[int], rng: np.random.Generator) -> np.ndarray:
    """Conditional bootstrap with replacement within each (derived_race, y_approved) stratum.

    Strata are visited in sorted (race, label) order; each stratum keeps its size.
    Returns positional indices into the complete Validation partition.
    """

    race = np.asarray(derived_race).astype(str)
    y = np.asarray(y_approved).astype(np.int64)
    keys = sorted(set(zip(race.tolist(), y.tolist())))
    parts = []
    for group, label in keys:
        positions = np.flatnonzero((race == group) & (y == label))
        parts.append(positions[rng.integers(0, len(positions), size=len(positions))])
    return np.concatenate(parts)


# --------------------------------------------------------------------------- evaluation


def _eod(y: np.ndarray, p: np.ndarray, decisions: np.ndarray, groups: np.ndarray, roster: Sequence[str], rules) -> EODEstimate:
    return compute_fixed_roster_eod(compute_roster_group_metrics(y, p, decisions, groups, roster, rules=rules), roster)


@dataclass(frozen=True)
class PairedEvaluation:
    baseline_ba: float | None
    mitigated_ba: float | None
    baseline_target: EODEstimate
    mitigated_target: EODEstimate
    baseline_audit: EODEstimate
    mitigated_audit: EODEstimate

    @property
    def ba_valid(self) -> bool:
        return self.baseline_ba is not None and self.mitigated_ba is not None

    @property
    def ba_cost(self) -> float | None:
        return ba_cost(self.baseline_ba, self.mitigated_ba)

    @property
    def target_valid(self) -> bool:
        return self.baseline_target.valid and self.mitigated_target.valid

    @property
    def audit_valid(self) -> bool:
        return self.baseline_audit.valid and self.mitigated_audit.valid

    @property
    def delta_target(self) -> float | None:
        return delta_eod(self.baseline_target, self.mitigated_target)

    @property
    def delta_audit(self) -> float | None:
        return delta_eod(self.baseline_audit, self.mitigated_audit)

    def reason_codes(self) -> str:
        codes = []
        if not self.ba_valid:
            codes.append(f"ba:{BA_INVALID_REASON}")
        if not self.target_valid:
            codes.append(f"target_eod:{EOD_INVALID_REASON}")
        if not self.audit_valid:
            codes.append(f"audit_eod:{EOD_INVALID_REASON}")
        return ";".join(codes) if codes else "none"


def paired_evaluation(
    y: np.ndarray,
    p: np.ndarray,
    groups: np.ndarray,
    baseline: np.ndarray,
    mitigated: np.ndarray,
    configuration: ConfigurationSpec,
    contract: RQ4AContract,
) -> PairedEvaluation:
    target_roster = contract.g3 if configuration.target_eod == "eod3" else contract.g6
    audit_roster = contract.g3 if configuration.audit_eod == "eod3" else contract.g6
    rules = contract.rules
    return PairedEvaluation(
        baseline_ba=balanced_accuracy(y, baseline),
        mitigated_ba=balanced_accuracy(y, mitigated),
        baseline_target=_eod(y, p, baseline, groups, target_roster, rules),
        mitigated_target=_eod(y, p, mitigated, groups, target_roster, rules),
        baseline_audit=_eod(y, p, baseline, groups, audit_roster, rules),
        mitigated_audit=_eod(y, p, mitigated, groups, audit_roster, rules),
    )


def _arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "record_id": frame["record_id"].to_numpy().astype(str),
        "y": frame["y_approved"].to_numpy(dtype=np.int8),
        "p": frame["p_approved"].to_numpy(dtype=np.float64),
        "base": frame["predicted_approved"].to_numpy(dtype=np.int8),
        "race": frame["derived_race"].to_numpy().astype(str),
        "sex": frame["applicant_sex_group"].to_numpy().astype(str),
    }


def _adapter(data: ModelData) -> CachedProbabilityEstimator:
    lookup = pd.concat([data.validation, data.test], ignore_index=True)
    return CachedProbabilityEstimator(pd.Series(lookup["p_approved"].to_numpy(dtype=np.float64), index=lookup["record_id"].astype(str)))


def bootstrap_model(
    data: ModelData,
    adapter: CachedProbabilityEstimator,
    inputs: RunnerInputs,
    settings: RunSettings,
    run_id: str,
) -> pd.DataFrame:
    """One row per replicate (valid and invalid), refitting the policy on each Validation resample."""

    configuration, contract = inputs.configuration, inputs.contract
    roster = configuration.roster
    val, test = _arrays(data.validation), _arrays(data.test)
    rows = []
    for replicate_id in range(settings.replicates):
        streams = replicate_streams(replicate_id)
        resample = stratified_resample(val["race"], val["y"], streams.calibration_rng)
        policy = fit_policy(adapter, val["record_id"][resample], val["y"][resample], val["race"][resample], roster)
        rows_idx = inputs.bootstrap_indices[replicate_id]
        mitigated, _ = apply_policy(
            policy, test["record_id"][rows_idx], test["race"][rows_idx], test["base"][rows_idx], roster, seed=streams.decision_seed
        )
        evaluation = paired_evaluation(
            test["y"][rows_idx], test["p"][rows_idx], test["race"][rows_idx], test["base"][rows_idx], mitigated, configuration, contract
        )
        rows.append(
            {
                "run_id": run_id,
                "configuration_id": configuration.name,
                "model": data.model,
                "replicate_id": replicate_id,
                "root_seed": streams.root_seed,
                "calibration_stream_id": streams.calibration_stream_id,
                "decision_stream_id": streams.decision_stream_id,
                "baseline_ba": evaluation.baseline_ba,
                "mitigated_ba": evaluation.mitigated_ba,
                "ba_cost": evaluation.ba_cost,
                "ba_valid": evaluation.ba_valid,
                "baseline_target_eod": evaluation.baseline_target.value,
                "mitigated_target_eod": evaluation.mitigated_target.value,
                "delta_target_eod": evaluation.delta_target,
                "target_eod_valid": evaluation.target_valid,
                "baseline_audit_eod": evaluation.baseline_audit.value,
                "mitigated_audit_eod": evaluation.mitigated_audit.value,
                "delta_audit_eod": evaluation.delta_audit,
                "audit_eod_valid": evaluation.audit_valid,
                "invalid_reason_codes": evaluation.reason_codes(),
            }
        )
    return pd.DataFrame(rows)


def percentile_interval(values: Sequence[float], confidence: float) -> tuple[float, float]:
    lower_q = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(np.asarray(values, dtype=np.float64), [lower_q, 1.0 - lower_q], method="linear")
    return float(lower), float(upper)


def summarize_quantity(
    values: pd.Series, valid: pd.Series, point: float | None, *, minimum_valid: int, confidence: float
) -> dict[str, Any]:
    """Percentile CI over the valid values of one quantity; null when below the minimum-valid gate."""

    kept = values[valid.astype(bool)].astype(float).to_numpy()
    gate = len(kept) >= minimum_valid
    lower, upper = percentile_interval(kept, confidence) if gate else (None, None)
    return {
        "point_estimate": point,
        "ci_lower": lower,
        "ci_upper": upper,
        "n_replicates": int(len(values)),
        "n_valid": int(len(kept)),
        "minimum_valid": int(minimum_valid),
        "valid_gate_status": "pass" if gate else "fail",
    }


def primary_criterion(
    *,
    point_target_valid: bool,
    point_delta_eod: float | None,
    delta_eod_ci_upper: float | None,
    n_valid_delta_eod: int,
    point_ba_cost: float | None,
    ba_cost_ci_upper: float | None,
    n_valid_ba_cost: int,
    minimum_valid: int,
    ba_budget: float,
) -> tuple[str, list[str]]:
    """Locked four-condition criterion; returns (pass|fail|not_evaluable, reason codes)."""

    reasons: list[str] = []
    if not point_target_valid:
        reasons.append("required_group_metric_invalid")
    if n_valid_delta_eod < minimum_valid or n_valid_ba_cost < minimum_valid:
        reasons.append("insufficient_valid_bootstrap_replicates")
    if reasons:
        return "not_evaluable", reasons
    _require(
        None not in (point_delta_eod, delta_eod_ci_upper, point_ba_cost, ba_cost_ci_upper),
        "Criterion inputs are undefined although the validity gates passed.",
    )
    if point_delta_eod >= 0 or delta_eod_ci_upper >= 0:
        reasons.append("target_eod_reduction_not_demonstrated")
    if point_ba_cost > ba_budget or ba_cost_ci_upper > ba_budget:
        reasons.append("ba_budget_exceeded")
    return ("fail", reasons) if reasons else ("pass", [])


def stability_status(sensitivity: pd.DataFrame, *, seeds: Sequence[int], reference_seed: int, ba_budget: float) -> str:
    """stable / unstable / not_assessed over the 30 locked seeds of one model and configuration."""

    if sorted(sensitivity["seed"].astype(int).tolist()) != sorted(int(s) for s in seeds):
        return "not_assessed"
    valid = sensitivity["target_eod_valid"].astype(bool) & sensitivity["ba_valid"].astype(bool)
    if not valid.all():
        return "not_assessed"
    indexed = sensitivity.set_index(sensitivity["seed"].astype(int))
    eod_ind = indexed["delta_target_eod"].astype(float) < 0
    ba_ind = indexed["ba_cost"].astype(float) <= ba_budget
    same = (eod_ind == eod_ind[reference_seed]) & (ba_ind == ba_ind[reference_seed])
    return "stable" if bool(same.all()) else "unstable"


def _ordered_reasons(codes: Sequence[str]) -> list[str]:
    unique = set(codes)
    ordered = [code for code in REASON_ORDER if code in unique]
    return ordered if ordered else ["none"]


def _min_max_range(values: pd.Series, valid: pd.Series) -> dict[str, float | None]:
    kept = values[valid.astype(bool)].astype(float)
    if kept.empty:
        return {"min": None, "max": None, "range": None}
    return {"min": float(kept.min()), "max": float(kept.max()), "range": float(kept.max() - kept.min())}


def _inherited(y: np.ndarray, p: np.ndarray) -> tuple[float | None, float]:
    roc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    return roc, float(np.mean(np.square(p - y), dtype=np.float64))


def _experiment_row(
    common: Mapping[str, Any],
    partition: str,
    policy: str,
    arrays: Mapping[str, np.ndarray],
    decisions: np.ndarray,
    contract: RQ4AContract,
    sex_roster: Sequence[str],
) -> tuple[dict[str, Any], dict[str, EODEstimate]]:
    y, p = arrays["y"], arrays["p"]
    eods = {
        "eod3": _eod(y, p, decisions, arrays["race"], contract.g3, contract.rules),
        "eod6": _eod(y, p, decisions, arrays["race"], contract.g6, contract.rules),
        "sex_eod": _eod(y, p, decisions, arrays["sex"], sex_roster, contract.rules),
    }
    roc, brier = _inherited(y, p)
    row = {**common, "partition": partition, "policy": policy, "balanced_accuracy": balanced_accuracy(y, decisions)}
    for name, estimate in eods.items():
        row.update(
            {
                name: estimate.value,
                f"{name}_valid": estimate.valid,
                f"{name}_tpr_range": estimate.tpr_range,
                f"{name}_fpr_range": estimate.fpr_range,
            }
        )
    row.update(denied_metrics(y, decisions))
    row.update({"roc_auc_inherited": roc, "brier_inherited": brier})
    return row, eods


def _group_rows(
    common: Mapping[str, Any],
    partition: str,
    policy: str,
    attribute: str,
    labels: np.ndarray,
    report_groups: Sequence[str],
    arrays: Mapping[str, np.ndarray],
    decisions: np.ndarray,
    target_roster: Sequence[str],
    contract: RQ4AContract,
) -> list[dict[str, Any]]:
    metrics = compute_roster_group_metrics(arrays["y"], arrays["p"], decisions, labels, report_groups, rules=contract.rules)
    roles = {name: extremum_roles(metrics, metric) for name, metric in (("tpr", "tpr_approved"), ("fpr", "fpr_approved"), ("selection_rate", "selection_rate"))}
    rows = []
    for group in report_groups:
        m = metrics[group]
        tpr, fpr, sr = m["tpr_approved"], m["fpr_approved"], m["selection_rate"]
        rows.append(
            {
                **common,
                "partition": partition,
                "policy": policy,
                "attribute": attribute,
                "group": group,
                "in_target_roster": attribute == "derived_race" and group in target_roster,
                "n_rows": tpr.group_rows,
                "n_actual_approved": tpr.actual_approved,
                "n_actual_denied": tpr.actual_denied,
                "tpr": tpr.value,
                "fpr": fpr.value,
                "selection_rate": sr.value,
                "tpr_valid": tpr.is_defined,
                "fpr_valid": fpr.is_defined,
                "selection_rate_valid": sr.is_defined,
                "tpr_extremum_role": roles["tpr"][group],
                "fpr_extremum_role": roles["fpr"][group],
                "selection_rate_extremum_role": roles["selection_rate"][group],
            }
        )
    return rows


def _decision_frame(common: Mapping[str, Any], arrays: Mapping[str, np.ndarray], mitigated: np.ndarray, in_roster: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            **{key: value for key, value in common.items()},
            "record_id": arrays["record_id"],
            "y_approved": arrays["y"],
            "p_approved": arrays["p"],
            "derived_race": arrays["race"],
            "applicant_sex_group": arrays["sex"],
            "in_target_roster": in_roster,
            "baseline_decision": arrays["base"],
            "mitigated_decision": mitigated,
            "decision_changed": arrays["base"] != mitigated,
        }
    )


@dataclass
class ModelResult:
    validation_decisions: pd.DataFrame
    test_decisions: pd.DataFrame
    replicates: pd.DataFrame
    status: dict[str, Any]
    summary: dict[str, Any]
    experiment: list[dict[str, Any]]
    groups: list[dict[str, Any]]
    changes: list[dict[str, Any]]
    sensitivity: pd.DataFrame
    bootstrap_summary: list[dict[str, Any]]
    policy: Any = field(repr=False)


class _Runtime:
    """runtime.csv rows plus the current attempt stage/model for failure receipts."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.rows: list[dict[str, Any]] = []
        self.current_stage = "attempt_setup"
        self.current_model: str | None = None

    def mark(self, stage: str, model: str | None = None) -> None:
        self.current_stage, self.current_model = stage, model

    @contextmanager
    def stage(self, name: str, *, model: str | None = None) -> Iterator[None]:
        self.mark(name, model)
        name = f"{model}:{name}" if model else name
        started = datetime.now(timezone.utc)
        clock = time.perf_counter()
        status = "failed"
        try:
            yield
            status = "completed"
        finally:
            self.rows.append(
                {
                    "run_id": self.run_id,
                    "stage": name,
                    "started_at_utc": started.isoformat(),
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": time.perf_counter() - clock,
                    "status": status,
                }
            )


def run_model(data: ModelData, inputs: RunnerInputs, settings: RunSettings, run_id: str, runtime: _Runtime) -> ModelResult:
    # Receipt-only stage labels (not runtime.csv rows) for work outside the timed stages.
    runtime.mark("model_setup", data.model)
    config, contract, configuration = inputs.config, inputs.contract, inputs.configuration
    roster = configuration.roster
    primary_seed = contract.primary_seed
    budget = float(config["primary_criterion"]["ba_budget_primary"])
    confidence = float(config["bootstrap"]["confidence_interval"]["confidence_level"])
    common = {"run_id": run_id, "configuration_id": configuration.name, "model": data.model}
    adapter = _adapter(data)
    val, test = _arrays(data.validation), _arrays(data.test)

    with runtime.stage("fit_complete_validation", model=data.model):
        policy = fit_policy(adapter, val["record_id"], val["y"], val["race"], roster)

    with runtime.stage("point_estimates", model=data.model):
        decisions = {}
        for partition, arrays in (("validation", val), ("test", test)):
            mitigated, in_roster = apply_policy(policy, arrays["record_id"], arrays["race"], arrays["base"], roster, seed=primary_seed)
            decisions[partition] = (mitigated, in_roster)
        experiment, groups, eods = [], [], {}
        for partition, arrays in (("validation", val), ("test", test)):
            for policy_name, chosen in (("baseline", arrays["base"]), ("mitigated", decisions[partition][0])):
                row, eods[(partition, policy_name)] = _experiment_row(common, partition, policy_name, arrays, chosen, contract, inputs.sex_roster)
                experiment.append(row)
                groups += _group_rows(common, partition, policy_name, "derived_race", arrays["race"], race_report_groups(config), arrays, chosen, roster, contract)
                groups += _group_rows(common, partition, policy_name, "applicant_sex", arrays["sex"], inputs.sex_report_groups, arrays, chosen, roster, contract)
        point = paired_evaluation(test["y"], test["p"], test["race"], test["base"], decisions["test"][0], configuration, contract)
        changes = [{**common, **row} for row in decision_changes(test["base"], decisions["test"][0], test["race"], race_report_groups(config))]
        validation_decisions = _decision_frame(common, val, *decisions["validation"])
        test_decisions = _decision_frame(common, test, *decisions["test"])

    with runtime.stage("randomization_sensitivity", model=data.model):
        sensitivity_rows = []
        for seed in contract.sensitivity_seeds:
            mitigated, _ = apply_policy(policy, test["record_id"], test["race"], test["base"], roster, seed=seed)
            evaluation = paired_evaluation(test["y"], test["p"], test["race"], test["base"], mitigated, configuration, contract)
            sensitivity_rows.append(
                {
                    **common,
                    "seed": int(seed),
                    "mitigated_ba": evaluation.mitigated_ba,
                    "ba_cost": evaluation.ba_cost,
                    "ba_valid": evaluation.ba_valid,
                    "mitigated_target_eod": evaluation.mitigated_target.value,
                    "delta_target_eod": evaluation.delta_target,
                    "target_eod_valid": evaluation.target_valid,
                    "mitigated_audit_eod": evaluation.mitigated_audit.value,
                    "delta_audit_eod": evaluation.delta_audit,
                    "audit_eod_valid": evaluation.audit_valid,
                    "invalid_reason_codes": evaluation.reason_codes(),
                }
            )
        sensitivity = pd.DataFrame(sensitivity_rows)

    with runtime.stage("bootstrap", model=data.model):
        replicates = bootstrap_model(data, adapter, inputs, settings, run_id)

    runtime.mark("summary_and_criterion", data.model)
    summaries = {
        "delta_target_eod": summarize_quantity(replicates["delta_target_eod"], replicates["target_eod_valid"], point.delta_target, minimum_valid=settings.minimum_valid, confidence=confidence),
        "ba_cost": summarize_quantity(replicates["ba_cost"], replicates["ba_valid"], point.ba_cost, minimum_valid=settings.minimum_valid, confidence=confidence),
        "delta_audit_eod": summarize_quantity(replicates["delta_audit_eod"], replicates["audit_eod_valid"], point.delta_audit, minimum_valid=settings.minimum_valid, confidence=confidence),
    }
    ci = config["bootstrap"]["confidence_interval"]
    bootstrap_summary = [
        {
            **common,
            "quantity": quantity,
            **summaries[quantity],
            "confidence_level": confidence,
            "ci_method": ci["method"],
            "quantile_method": ci["quantile_method"],
        }
        for quantity in SUMMARY_QUANTITIES
    ]

    criterion, criterion_reasons = primary_criterion(
        point_target_valid=point.target_valid,
        point_delta_eod=point.delta_target,
        delta_eod_ci_upper=summaries["delta_target_eod"]["ci_upper"],
        n_valid_delta_eod=summaries["delta_target_eod"]["n_valid"],
        point_ba_cost=point.ba_cost,
        ba_cost_ci_upper=summaries["ba_cost"]["ci_upper"],
        n_valid_ba_cost=summaries["ba_cost"]["n_valid"],
        minimum_valid=settings.minimum_valid,
        ba_budget=budget,
    )
    stability = stability_status(sensitivity, seeds=contract.sensitivity_seeds, reference_seed=primary_seed, ba_budget=budget)
    reasons = _ordered_reasons(criterion_reasons + (["unstable_diagnostics"] if stability == "unstable" else []))
    status = {"execution_status": "completed", "stability_status": stability, "primary_criterion": criterion, "reason_codes": reasons}

    budget_sensitivity = {}
    for alt in config["primary_criterion"]["ba_budget_sensitivity_reporting_only"]:
        upper = summaries["ba_cost"]["ci_upper"]
        budget_sensitivity[str(alt)] = {
            "point_ba_cost_le_budget": None if point.ba_cost is None else bool(point.ba_cost <= float(alt)),
            "ba_cost_ci_upper_le_budget": None if upper is None else bool(upper <= float(alt)),
        }

    def gap(name: str, policy_name: str) -> float | None:
        test_eod, val_eod = eods[("test", policy_name)][name], eods[("validation", policy_name)][name]
        return float(test_eod.value) - float(val_eod.value) if test_eod.valid and val_eod.valid else None

    summary = {
        **status,
        "point_delta_target_eod": point.delta_target,
        "delta_target_eod_ci": [summaries["delta_target_eod"]["ci_lower"], summaries["delta_target_eod"]["ci_upper"]],
        "point_ba_cost": point.ba_cost,
        "ba_cost_ci": [summaries["ba_cost"]["ci_lower"], summaries["ba_cost"]["ci_upper"]],
        "n_valid_delta_target_eod": summaries["delta_target_eod"]["n_valid"],
        "n_valid_ba_cost": summaries["ba_cost"]["n_valid"],
        "ba_budget_sensitivity": budget_sensitivity,
        "fairness_generalization_gap_eod3": {p: gap("eod3", p) for p in POLICIES},
        "fairness_generalization_gap_eod6": {p: gap("eod6", p) for p in POLICIES},
        "sensitivity_mitigated_target_eod_min_max_range": _min_max_range(sensitivity["mitigated_target_eod"], sensitivity["target_eod_valid"]),
        "sensitivity_delta_target_eod_min_max_range": _min_max_range(sensitivity["delta_target_eod"], sensitivity["target_eod_valid"]),
        "sensitivity_mitigated_ba_min_max_range": _min_max_range(sensitivity["mitigated_ba"], sensitivity["ba_valid"]),
        "sensitivity_ba_cost_min_max_range": _min_max_range(sensitivity["ba_cost"], sensitivity["ba_valid"]),
    }
    return ModelResult(validation_decisions, test_decisions, replicates, status, summary, experiment, groups, changes, sensitivity, bootstrap_summary, policy)


# --------------------------------------------------------------------------- outputs


def _columns(section: str, filename: str) -> list[str]:
    return list(LOCKED_OUTPUTS[section][filename]["columns"])


def _write_csv(path: Path, frame: pd.DataFrame, columns: Sequence[str]) -> None:
    _require(list(frame.columns) == list(columns), f"{path.name} columns differ from the locked YAML column list.")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, allow_nan=False) + "\n", encoding="utf-8")


def _frame(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(list(rows)).reindex(columns=list(columns)) if rows else pd.DataFrame(columns=list(columns))


def resolve_run_root(output_root: Path, run_id: str, *, project_root: Path = PROJECT_ROOT) -> Path:
    """`<output_root>/<run_id>`; never inside the accepted RQ3 namespace and never an existing path."""

    run_root = (Path(output_root) / run_id).resolve()
    rq3 = (Path(project_root) / RQ3_NAMESPACE).resolve()
    _require(run_root != rq3 and rq3 not in run_root.parents, "RQ4a outputs must never be written under the RQ3 namespace.")
    _require(not run_root.exists(), f"RQ4a run directory already exists; overwrite is prohibited: {run_root}")
    return run_root


def failure_receipt_path(output_root: Path, run_id: str) -> Path:
    """`<output_root>/_failures/<run_id>.json` (the default root gives the locked receipt directory)."""

    return Path(output_root).resolve() / FAILURES_DIRNAME / f"{run_id}.json"


def check_run_id_available(output_root: Path, run_id: str, *, project_root: Path = PROJECT_ROOT) -> Path:
    """Collision gate: neither a completed-run directory nor a failure receipt may exist for the run ID."""

    run_root = resolve_run_root(output_root, run_id, project_root=project_root)
    receipt = failure_receipt_path(output_root, run_id)
    _require(not receipt.exists(), f"RQ4a run ID was consumed by a failure receipt; rerun with the next sequence: {receipt}")
    return run_root


def error_summary(exc: BaseException) -> str:
    """Exception message collapsed to one line and truncated to the locked maximum."""

    return " ".join(str(exc).split())[:ERROR_SUMMARY_MAX]


def _qualified_name(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


INFEASIBLE_EXCEPTION_TYPE = _qualified_name(RQ4AInfeasibleError)


def build_failure_receipt(
    *,
    run_id: str,
    configuration_id: str,
    exc: BaseException,
    failed_stage: str,
    model: str | None,
    started_at: datetime,
    failed_at: datetime,
) -> dict[str, Any]:
    kind = "explicit_infeasible_exception" if type(exc) is RQ4AInfeasibleError else "runtime_exception"
    return {
        "schema_version": LOCKED_FAILURE_RECEIPTS["schema_version"],
        "run_id": run_id,
        "configuration_id": configuration_id,
        "execution_status": LOCKED_FAILURE_RECEIPTS["execution_status"][kind],
        "reason_codes": list(LOCKED_FAILURE_RECEIPTS["reason_codes"][kind]),
        "failed_stage": failed_stage,
        "model": model,
        "started_at_utc": started_at.isoformat(),
        "failed_at_utc": failed_at.isoformat(),
        "runtime_provenance": runtime_provenance(),
        "exception_type": _qualified_name(type(exc)),
        "error_summary": error_summary(exc),
    }


def _utc_timestamp(value: Any, field: str) -> datetime:
    _require(isinstance(value, str), f"Failure receipt {field} must be a string.")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise RQ4ARunnerError(f"Failure receipt {field} is not an ISO-8601 timestamp.") from exc
    _require(parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0, f"Failure receipt {field} must be UTC.")
    return parsed


def validate_failure_receipt(payload: Any, *, verify_producer: bool = False) -> None:
    """Validate a receipt against the locked literals; provenance is structural unless verifying the producer."""

    _require(isinstance(payload, dict) and list(payload) == LOCKED_FAILURE_RECEIPTS["fields"], "Failure receipt fields differ from the locked order.")
    _require(type(payload["schema_version"]) is int and payload["schema_version"] == 1, "Failure receipt schema_version must be 1.")
    configuration_id, run_id = payload["configuration_id"], payload["run_id"]
    _require(configuration_id in RUN_ID_PREFIXES, "Failure receipt configuration_id is not RQ4a-3 or RQ4a-6.")
    match = re.fullmatch(rf"{RUN_ID_PREFIXES[configuration_id]}_postproc_(\d{{8}})_(\d{{2}})", run_id) if isinstance(run_id, str) else None
    _require(match is not None and int(match.group(2)) >= 1, "Failure receipt run_id does not match its configuration pattern.")
    try:
        datetime.strptime(match.group(1), "%Y%m%d")
    except ValueError as exc:
        raise RQ4ARunnerError("Failure receipt run_id has an invalid date component.") from exc
    pairs = {
        (LOCKED_FAILURE_RECEIPTS["execution_status"][kind], tuple(LOCKED_FAILURE_RECEIPTS["reason_codes"][kind]))
        for kind in ("runtime_exception", "explicit_infeasible_exception")
    }
    reasons = payload["reason_codes"]
    _require(
        isinstance(payload["execution_status"], str)
        and isinstance(reasons, list)
        and all(isinstance(code, str) for code in reasons)
        and (payload["execution_status"], tuple(reasons)) in pairs,
        "Failure receipt status/reason pair is not locked.",
    )
    _require(payload["execution_status"] in LOCKED_STATUS_CONTRACT["failure_receipt_execution_status"], "Failure receipt execution_status is not a locked value.")
    _require(set(reasons) <= set(LOCKED_STATUS_CONTRACT["reason_codes"]), "Failure receipt reason code is not locked.")
    exception_type = payload["exception_type"]
    _require(isinstance(exception_type, str) and exception_type, "Failure receipt exception_type must be a non-empty string.")
    _require(
        (payload["execution_status"] == "infeasible") == (exception_type == INFEASIBLE_EXCEPTION_TYPE),
        "Failure receipt infeasible status must come only from RQ4AInfeasibleError.",
    )
    _require(isinstance(payload["failed_stage"], str) and payload["failed_stage"], "Failure receipt failed_stage must be a non-empty string.")
    _require(payload["model"] is None or payload["model"] in MODELS, "Failure receipt model must be null or a locked model.")
    started = _utc_timestamp(payload["started_at_utc"], "started_at_utc")
    failed = _utc_timestamp(payload["failed_at_utc"], "failed_at_utc")
    _require(failed >= started, "Failure receipt failed_at_utc precedes started_at_utc.")
    if verify_producer:
        verify_producer_provenance(payload["runtime_provenance"])
    else:
        validate_runtime_provenance_structure(payload["runtime_provenance"])
    summary = payload["error_summary"]
    _require(isinstance(summary, str) and "\n" not in summary and "\r" not in summary, "Failure receipt error_summary must be one line.")
    _require(len(summary) <= ERROR_SUMMARY_MAX, "Failure receipt error_summary exceeds 1,000 characters.")


def _write_receipt_file(path: Path, payload: Mapping[str, Any]) -> None:
    _write_json(path, payload)


def publish_failure_receipt(output_root: Path, receipt: Mapping[str, Any]) -> Path:
    """Atomically publish `<output_root>/_failures/<run_id>.json`; never overwrite; no partial file on failure."""

    validate_failure_receipt(dict(receipt), verify_producer=True)
    final = failure_receipt_path(output_root, receipt["run_id"])
    temporary = final.with_name(f".{receipt['run_id']}.tmp-{os.getpid()}.json")
    _require(not final.exists(), f"Refusing to overwrite an existing failure receipt: {final}")
    _require(not temporary.exists(), f"Failure-receipt temporary file already exists: {temporary}")
    final.parent.mkdir(parents=True, exist_ok=True)
    try:
        _write_receipt_file(temporary, receipt)
        validate_failure_receipt(json.loads(temporary.read_text(encoding="utf-8")), verify_producer=True)
        _require(not final.exists(), f"Failure receipt appeared during publication: {final}")
        temporary.rename(final)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return final


def expected_files(models: Sequence[str]) -> set[str]:
    """Locked file set derived from the independent layout copy (one policy directory per model)."""

    files = {
        f"{section}/{name}"
        for section in ("validation", "test_evaluation", "reporting")
        for name in LOCKED_OUTPUTS[section]
        if name.endswith((".csv", ".json"))
    }
    policies = LOCKED_OUTPUTS["validation"]["fitted_policies_dir"]
    for model in models:
        files |= {f"{policies}/{model}/policy.joblib", f"{policies}/{model}/policy_interpolation.json"}
    return files


def _relative_files(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}


def _csv_rows(path: Path) -> int | None:
    if path.suffix != ".csv":
        return None
    return int(len(pd.read_csv(path, dtype=str, keep_default_na=False)))


def build_file_entries(root: Path) -> list[dict[str, Any]]:
    """Per-file index entries (locked fields) recomputed from the files on disk."""

    files = []
    for relative in sorted(_relative_files(root) - {"reporting/artifact_index.json"}):
        path = root / relative
        payload = path.read_bytes()
        files.append({"relative_path": relative, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload), "rows": _csv_rows(path)})
    return files


def build_artifact_index(root: Path, provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Artifact index with the given producer provenance; provenance is never regenerated here."""

    return {"runtime_provenance": dict(provenance), "files": build_file_entries(root)}


def validate_run_directory(
    root: Path, *, replicates: int, rows: Mapping[str, int], verify_producer: bool = False
) -> dict[str, Any]:
    """Validate layout, columns, row counts, statuses, provenance, and index against the locked copy.

    Stored provenance is checked structurally and kept as recorded, so a later audit
    may run under another interpreter. `verify_producer=True` (used by the producing
    run before publication) also requires it to equal the current process.
    """

    root = Path(root)
    _require(_relative_files(root) == expected_files(MODELS), "RQ4a output file set differs from the locked layout.")
    for section in ("validation", "test_evaluation", "reporting"):
        for filename, spec in LOCKED_OUTPUTS[section].items():
            if filename.endswith(".csv"):
                header = pd.read_csv(root / section / filename, nrows=0).columns.tolist()
                _require(header == list(spec["columns"]), f"{section}/{filename} header differs from the locked columns.")
    for section, partition in (("validation", "validation"), ("test_evaluation", "test")):
        frame = pd.read_csv(root / section / "decisions.csv", dtype={"record_id": str})
        _require(frame.groupby("model").size().to_dict() == {m: rows[partition] for m in MODELS}, f"{section}/decisions.csv row counts differ.")
    reps = pd.read_csv(root / "test_evaluation" / "bootstrap_replicates.csv")
    for model in MODELS:
        ids = reps.loc[reps["model"] == model, "replicate_id"].tolist()
        _require(ids == list(range(replicates)), f"{model} bootstrap replicates are incomplete or out of order.")
    sens = pd.read_csv(root / "reporting" / "randomization_sensitivity.csv")
    for model in MODELS:
        _require(sens.loc[sens["model"] == model, "seed"].tolist() == SENSITIVITY_SEEDS, f"{model} sensitivity seeds differ from 42..71.")
    status = json.loads((root / "test_evaluation" / "status.json").read_text(encoding="utf-8"))
    status_spec = LOCKED_OUTPUTS["test_evaluation"]["status.json"]
    _require(list(status) == status_spec["fields"], "status.json fields differ from the locked fields.")
    _require(list(status["model_statuses"]) == status_spec["model_statuses"]["models"], "status.json model_statuses must be keyed by XGBoost and RealMLP.")
    for model, entry in status["model_statuses"].items():
        _require(list(entry) == status_spec["model_statuses"]["per_model_fields"], f"{model} status fields differ.")
        _require(
            entry["execution_status"] in LOCKED_STATUS_CONTRACT["completed_run_status_json_execution_status"],
            f"{model} completed-run execution_status is not a locked value.",
        )
        _require(entry["stability_status"] in LOCKED_STATUS_CONTRACT["stability_status"], f"{model} stability_status is not a locked value.")
        _require(entry["primary_criterion"] in LOCKED_STATUS_CONTRACT["primary_criterion"], f"{model} primary_criterion is not a locked value.")
        _require(set(entry["reason_codes"]) <= set(LOCKED_STATUS_CONTRACT["reason_codes"]), f"{model} reason_codes contain an unlocked code.")
    summary = json.loads((root / "reporting" / "model_summary.json").read_text(encoding="utf-8"))
    per_model = LOCKED_OUTPUTS["reporting"]["model_summary.json"]["per_model_fields"]
    for model in MODELS:
        _require(list(summary["model_summaries"][model]) == per_model, f"{model} model_summary fields differ from the locked fields.")
    index = json.loads((root / "reporting" / "artifact_index.json").read_text(encoding="utf-8"))
    _require(isinstance(index, dict) and list(index) == ["runtime_provenance", "files"], "artifact_index.json top-level keys differ.")
    if verify_producer:
        verify_producer_provenance(index["runtime_provenance"])
    else:
        validate_runtime_provenance_structure(index["runtime_provenance"])
    per_file = LOCKED_OUTPUTS["reporting"]["artifact_index.json"]["per_file_fields"]
    _require(all(list(entry) == per_file for entry in index["files"]), "artifact_index.json per-file fields differ.")
    _require(index["files"] == build_file_entries(root), "artifact_index.json file entries do not match the written files.")
    return {"status": "PASS", "files": len(index["files"]) + 1}


def execute_run(
    inputs: RunnerInputs,
    run_id: str,
    output_root: Path,
    *,
    settings: RunSettings | None = None,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Run both models for one configuration and publish the run directory atomically.

    Read-only gates run first and publish nothing on failure. After they pass the
    attempt begins: a later exception removes the temporary run directory, publishes
    one failure receipt under `<output_root>/_failures/`, and is re-raised.
    """

    validate_runner_contract(inputs.config)
    validate_config(inputs.config, project_root=project_root)
    validate_prerequisites(inputs.config)
    settings = settings or RunSettings.from_contract(inputs.contract)
    validate_run_id(run_id, inputs.configuration)
    run_root = check_run_id_available(output_root, run_id, project_root=project_root)
    validate_runner_inputs(inputs, settings)
    temporary = run_root.with_name(f".{run_root.name}.tmp-{os.getpid()}")
    _require(not temporary.exists(), f"RQ4a temporary directory already exists: {temporary}")

    runtime = _Runtime(run_id)
    started_at = datetime.now(timezone.utc)
    try:
        temporary.mkdir(parents=True)
        results = {model: run_model(inputs.models[model], inputs, settings, run_id, runtime) for model in MODELS}
        with runtime.stage("policy_persistence_and_reload"):
            for model, result in results.items():
                runtime.mark("policy_persistence_and_reload", model)
                directory = temporary / "validation" / "fitted_policies" / model
                save_policy(result.policy, directory)
                reloaded = load_policy(directory)
                roster = inputs.configuration.roster
                for frame in (result.validation_decisions, result.test_decisions):
                    again, _ = apply_policy(
                        reloaded, frame["record_id"].to_numpy(), frame["derived_race"].to_numpy(), frame["baseline_decision"].to_numpy(), roster, seed=inputs.contract.primary_seed
                    )
                    _require(np.array_equal(again, frame["mitigated_decision"].to_numpy()), f"{model} reloaded policy does not reproduce seed-42 decisions.")
        with runtime.stage("write_outputs"):
            decision_columns = _columns("validation", "decisions.csv")
            _write_csv(temporary / "validation" / "decisions.csv", pd.concat([r.validation_decisions for r in results.values()], ignore_index=True), decision_columns)
            _write_csv(temporary / "test_evaluation" / "decisions.csv", pd.concat([r.test_decisions for r in results.values()], ignore_index=True), _columns("test_evaluation", "decisions.csv"))
            _write_csv(
                temporary / "test_evaluation" / "bootstrap_replicates.csv",
                pd.concat([r.replicates for r in results.values()], ignore_index=True),
                _columns("test_evaluation", "bootstrap_replicates.csv"),
            )
            _write_json(
                temporary / "test_evaluation" / "status.json",
                {"run_id": run_id, "configuration_id": inputs.configuration.name, "model_statuses": {m: r.status for m, r in results.items()}},
            )
            _write_json(
                temporary / "reporting" / "model_summary.json",
                {"run_id": run_id, "configuration_id": inputs.configuration.name, "model_summaries": {m: r.summary for m, r in results.items()}},
            )
            for filename, attribute in (
                ("experiment_metrics.csv", "experiment"),
                ("group_metrics.csv", "groups"),
                ("decision_changes.csv", "changes"),
                ("bootstrap_summary.csv", "bootstrap_summary"),
            ):
                columns = _columns("reporting", filename)
                _write_csv(temporary / "reporting" / filename, _frame([row for r in results.values() for row in getattr(r, attribute)], columns), columns)
            _write_csv(
                temporary / "reporting" / "randomization_sensitivity.csv",
                pd.concat([r.sensitivity for r in results.values()], ignore_index=True),
                _columns("reporting", "randomization_sensitivity.csv"),
            )
        runtime.mark("finalize_and_validate_outputs")
        _write_csv(temporary / "reporting" / "runtime.csv", _frame(runtime.rows, _columns("reporting", "runtime.csv")), _columns("reporting", "runtime.csv"))
        _write_json(temporary / "reporting" / "artifact_index.json", build_artifact_index(temporary, runtime_provenance()))
        rows = {partition: len(getattr(inputs.models[MODELS[0]], partition)) for partition in PARTITIONS}
        validation = validate_run_directory(temporary, replicates=settings.replicates, rows=rows, verify_producer=True)
        runtime.mark("atomic_publication")
        _require(not run_root.exists(), f"RQ4a run directory appeared during execution: {run_root}")
        temporary.replace(run_root)
        return {**validation, "run_root": str(run_root), "model_statuses": {m: r.status for m, r in results.items()}}
    except Exception as exc:
        failed_at = datetime.now(timezone.utc)
        shutil.rmtree(temporary, ignore_errors=True)
        receipt = build_failure_receipt(
            run_id=run_id,
            configuration_id=inputs.configuration.name,
            exc=exc,
            failed_stage=runtime.current_stage,
            model=runtime.current_model,
            started_at=started_at,
            failed_at=failed_at,
        )
        try:
            publish_failure_receipt(output_root, receipt)
        except Exception as receipt_error:
            exc.add_note(f"RQ4a failure receipt was not published: {type(receipt_error).__name__}: {error_summary(receipt_error)}")
        raise


def run_rq4a(
    configuration_name: str,
    run_id: str,
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    project_root: Path = PROJECT_ROOT,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Production entry point: locked contract settings, real inputs, default namespace unless overridden."""

    config = load_config(config_path)
    validate_runner_contract(config)
    contract = validate_config(config, project_root=project_root)
    _require(configuration_name in contract.configurations, f"Unknown RQ4a configuration {configuration_name!r}.")
    validate_prerequisites(config)
    validate_run_id(run_id, contract.configurations[configuration_name])
    root = Path(output_root) if output_root is not None else project_root / config["artifact_namespace"]
    check_run_id_available(root, run_id, project_root=project_root)
    inputs = load_production_inputs(config, configuration_name, project_root=project_root, rq3_protocol_path=DEFAULT_RQ3_PROTOCOL_PATH)
    return execute_run(inputs, run_id, root, settings=RunSettings.from_contract(inputs.contract), project_root=project_root)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Dataset B RQ4a Equalized-Odds post-processing runner")
    parser.add_argument("--configuration", required=True, choices=["RQ4a-3", "RQ4a-6"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args(argv)
    print(json.dumps(run_rq4a(args.configuration, args.run_id, output_root=args.output_root), indent=2))


# Preserve qualified exception identifiers recorded in failure receipts.
RQ4ARunnerError.__module__ = "src.dataset_b.mitigation.runner"
RQ4AInfeasibleError.__module__ = "src.dataset_b.mitigation.runner"


if __name__ == "__main__":
    main()
