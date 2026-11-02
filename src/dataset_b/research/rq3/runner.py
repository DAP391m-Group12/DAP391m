"""Canonical Sequential artifact runner for the frozen Dataset B RQ3 audit."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from src.dataset_b.research.rq3.audit_data import (
    DEFAULT_PROTOCOL_PATH,
    ModelAuditInput,
    RQ3PreflightResult,
    load_protocol,
    run_preflight,
)
from src.dataset_b.research.rq3.metrics import (
    DenominatorRules,
    GapEstimate,
    MetricEstimate,
    RangeEstimate,
    compute_global_race_ranges,
    compute_group_metrics,
    compute_pairwise_gaps,
)
from src.utils.path import PROJECT_ROOT


class RQ3RunnerError(RuntimeError):
    """Raised when an RQ3 execution or artifact gate fails."""


IDENTITY_COLUMNS = (
    "result_type",
    "model",
    "attribute",
    "group",
    "target_group",
    "reference_group",
    "metric",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RQ3RunnerError(message)


def _rules(protocol: Mapping[str, Any]) -> DenominatorRules:
    values = protocol["denominators"]
    return DenominatorRules(
        minimum_group_rows=int(values["minimum_group_rows"]),
        minimum_actual_approved=int(values["minimum_actual_approved"]),
        minimum_actual_denied=int(values["minimum_actual_denied"]),
    )


def _attribute_groups(protocol: Mapping[str, Any], attribute: str) -> list[tuple[str, str]]:
    specification = protocol["attributes"][attribute]
    rows = [(str(specification["reference_group"]), "reference")]
    rows.extend((str(group), "primary") for group in specification.get("primary_comparisons", []))
    rows.extend((str(group), "secondary") for group in specification.get("secondary_comparisons", []))
    rows.extend((str(group), "coverage_only") for group in specification.get("coverage_only_groups", []))
    return rows


def _group_column(attribute: str) -> str:
    return "applicant_sex_group" if attribute == "applicant_sex" else "derived_race"


def _metric_rows(
    model: ModelAuditInput,
    audit: pd.DataFrame,
    protocol: Mapping[str, Any],
    sample_indices: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, MetricEstimate]]]]:
    predictions = model.predictions
    if sample_indices is None:
        y = predictions["y_approved"].to_numpy(dtype=np.int8)
        p = predictions["p_approved"].to_numpy(dtype=np.float64)
        predicted = predictions["predicted_approved"].to_numpy(dtype=np.int8)
        selected_audit = audit
    else:
        y = predictions["y_approved"].to_numpy(dtype=np.int8)[sample_indices]
        p = predictions["p_approved"].to_numpy(dtype=np.float64)[sample_indices]
        predicted = predictions["predicted_approved"].to_numpy(dtype=np.int8)[sample_indices]
        selected_audit = audit.iloc[sample_indices].reset_index(drop=True)

    rules = _rules(protocol)
    rows: list[dict[str, Any]] = []
    by_attribute: dict[str, dict[str, dict[str, MetricEstimate]]] = {}
    for attribute in ("applicant_sex", "derived_race"):
        group_column = _group_column(attribute)
        by_group: dict[str, dict[str, MetricEstimate]] = {}
        for group, role in _attribute_groups(protocol, attribute):
            mask = selected_audit[group_column].to_numpy(dtype=str) == group
            if not np.any(mask):
                raise RQ3RunnerError(f"No rows found for {attribute} group {group}.")
            metrics = compute_group_metrics(y[mask], p[mask], predicted[mask], rules=rules)
            by_group[group] = metrics
            for metric_name, estimate in metrics.items():
                value = estimate.value
                status = estimate.status
                if role == "coverage_only":
                    value = None
                    status = "coverage_only"
                rows.append(
                    {
                        "result_type": "group",
                        "model": model.model_name,
                        "run_id": model.run_id,
                        "threshold": model.threshold,
                        "attribute": attribute,
                        "group": group,
                        "group_role": role,
                        "target_group": "",
                        "reference_group": "",
                        "metric": metric_name,
                        "value": value,
                        "status": status,
                        "denominator": estimate.denominator,
                        "group_rows": estimate.group_rows,
                        "actual_approved": estimate.actual_approved,
                        "actual_denied": estimate.actual_denied,
                    }
                )
        by_attribute[attribute] = by_group
    return rows, by_attribute


def _gap_rows(
    model: ModelAuditInput,
    protocol: Mapping[str, Any],
    group_metrics: Mapping[str, Mapping[str, Mapping[str, MetricEstimate]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for attribute in ("applicant_sex", "derived_race"):
        specification = protocol["attributes"][attribute]
        reference = str(specification["reference_group"])
        targets = list(map(str, specification.get("primary_comparisons", [])))
        targets.extend(map(str, specification.get("secondary_comparisons", [])))
        for target in targets:
            gaps = compute_pairwise_gaps(
                group_metrics[attribute][target], group_metrics[attribute][reference]
            )
            for metric_name, estimate in gaps.items():
                rows.append(
                    {
                        "result_type": "pairwise",
                        "model": model.model_name,
                        "run_id": model.run_id,
                        "threshold": model.threshold,
                        "attribute": attribute,
                        "group": "",
                        "group_role": "",
                        "target_group": target,
                        "reference_group": reference,
                        "metric": metric_name,
                        "value": estimate.value,
                        "status": estimate.status,
                        "target_value": estimate.target_value,
                        "reference_value": estimate.reference_value,
                        "groups_used": "",
                    }
                )

    race = protocol["attributes"]["derived_race"]
    eligible_names = [str(race["reference_group"])]
    eligible_names.extend(map(str, race.get("primary_comparisons", [])))
    eligible_names.extend(map(str, race.get("secondary_comparisons", [])))
    eligible = {name: group_metrics["derived_race"][name] for name in eligible_names}
    ranges = compute_global_race_ranges(eligible)
    for metric_name, estimate in ranges.items():
        rows.append(
            {
                "result_type": "global_range",
                "model": model.model_name,
                "run_id": model.run_id,
                "threshold": model.threshold,
                "attribute": "derived_race",
                "group": "",
                "group_role": "",
                "target_group": "",
                "reference_group": str(race["reference_group"]),
                "metric": metric_name,
                "value": estimate.value,
                "status": estimate.status,
                "target_value": None,
                "reference_value": None,
                "groups_used": "|".join(estimate.groups_used),
            }
        )
    return rows


def _identity(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(column, "")) for column in IDENTITY_COLUMNS)


def _append_distribution(
    distributions: dict[tuple[str, ...], list[float]], rows: list[dict[str, Any]]
) -> None:
    for row in rows:
        value = row.get("value")
        distributions.setdefault(_identity(row), []).append(
            np.nan if value is None else float(value)
        )


def _bootstrap_summary(
    point_rows: list[dict[str, Any]],
    distributions: Mapping[tuple[str, ...], list[float]],
    protocol: Mapping[str, Any],
) -> pd.DataFrame:
    minimum_valid = int(protocol["uncertainty"]["minimum_valid_replicates"])
    confidence = float(protocol["uncertainty"]["confidence_level"])
    lower_q = (1.0 - confidence) / 2.0
    upper_q = 1.0 - lower_q
    rows: list[dict[str, Any]] = []
    for point in point_rows:
        values = np.asarray(distributions.get(_identity(point), []), dtype=np.float64)
        valid = values[np.isfinite(values)]
        available = len(valid) >= minimum_valid
        lower = float(np.quantile(valid, lower_q, method="linear")) if available else None
        upper = float(np.quantile(valid, upper_q, method="linear")) if available else None
        rows.append(
            {
                **{column: point.get(column, "") for column in IDENTITY_COLUMNS},
                "run_id": point["run_id"],
                "point_estimate": point.get("value"),
                "ci_lower": lower,
                "ci_upper": upper,
                "confidence_level": confidence,
                "valid_replicates": int(len(valid)),
                "invalid_replicates": int(len(values) - len(valid)),
                "replicate_count": int(len(values)),
                "ci_status": "defined" if available else "undefined_insufficient_valid_replicates",
            }
        )
    return pd.DataFrame(rows)


def analyze_model(
    model: ModelAuditInput,
    audit: pd.DataFrame,
    bootstrap_indices: np.ndarray,
    protocol: Mapping[str, Any],
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    start = time.perf_counter()
    group_rows, group_metrics = _metric_rows(model, audit, protocol)
    gap_rows = _gap_rows(model, protocol, group_metrics)
    point_seconds = time.perf_counter() - start

    bootstrap_start = time.perf_counter()
    distributions: dict[tuple[str, ...], list[float]] = {}
    for replicate_id, indices in enumerate(bootstrap_indices):
        replicate_group_rows, replicate_metrics = _metric_rows(
            model, audit, protocol, sample_indices=indices
        )
        replicate_gap_rows = _gap_rows(model, protocol, replicate_metrics)
        _append_distribution(distributions, replicate_group_rows + replicate_gap_rows)
        if progress is not None and (replicate_id + 1) % 100 == 0:
            progress(
                f"RQ3 model={model.model_name} bootstrap={replicate_id + 1}/{len(bootstrap_indices)}"
            )
    bootstrap = _bootstrap_summary(group_rows + gap_rows, distributions, protocol)
    bootstrap_seconds = time.perf_counter() - bootstrap_start
    timing = {
        "point_metric_seconds": point_seconds,
        "bootstrap_seconds": bootstrap_seconds,
        "total_seconds": time.perf_counter() - start,
    }
    return pd.DataFrame(group_rows), pd.DataFrame(gap_rows), bootstrap, timing


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def validate_output_directory(
    output_directory: Path,
    protocol: Mapping[str, Any],
    *,
    expected_rows: int | None = None,
) -> dict[str, Any]:
    expected_files = set(protocol["output"]["files"])
    observed_files = {path.name for path in output_directory.iterdir() if path.is_file()}
    _require(observed_files == expected_files, "RQ3 output file set differs from protocol.")
    audit = pd.read_csv(output_directory / "audit_metadata.csv")
    groups = pd.read_csv(output_directory / "group_metrics.csv")
    gaps = pd.read_csv(output_directory / "gap_metrics.csv")
    bootstrap = pd.read_csv(output_directory / "bootstrap_summary.csv")
    runtime = pd.read_csv(output_directory / "runtime.csv")
    status = json.loads((output_directory / "status.json").read_text(encoding="utf-8"))
    required_rows = int(protocol["audit_data"]["test_rows"]) if expected_rows is None else expected_rows
    _require(len(audit) == required_rows and audit["record_id"].is_unique, "RQ3 audit metadata row gate failed.")
    _require(not groups.empty and not gaps.empty and not bootstrap.empty, "RQ3 metric outputs must not be empty.")
    _require(set(runtime["model"]) == set(protocol["eligible_final_runs"]), "RQ3 runtime model coverage failed.")
    _require(len(bootstrap) == len(groups) + len(gaps), "RQ3 bootstrap summary does not cover every point result.")
    _require(not any(column.startswith("p_denied") for column in audit.columns), "Denied-oriented fields are forbidden in RQ3 audit metadata.")
    _require(status.get("status") == "PASS", "RQ3 status.json is not PASS.")
    return {
        "status": "PASS",
        "audit_rows": int(len(audit)),
        "group_metric_rows": int(len(groups)),
        "gap_metric_rows": int(len(gaps)),
        "bootstrap_summary_rows": int(len(bootstrap)),
        "model_count": int(len(runtime)),
    }


def run_from_preflight(
    preflight: RQ3PreflightResult,
    protocol: Mapping[str, Any],
    output_directory: Path,
    *,
    progress: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    _require(not output_directory.exists(), f"RQ3 output directory already exists: {output_directory}")
    temporary = output_directory.with_name(f".{output_directory.name}.tmp-{os.getpid()}")
    _require(not temporary.exists(), f"RQ3 temporary directory already exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        group_frames: list[pd.DataFrame] = []
        gap_frames: list[pd.DataFrame] = []
        bootstrap_frames: list[pd.DataFrame] = []
        runtime_rows: list[dict[str, Any]] = []
        model_status: dict[str, str] = {}
        for model_name in protocol["eligible_final_runs"]:
            if progress is not None:
                progress(f"RQ3 model={model_name} stage=start")
            groups, gaps, bootstrap, timing = analyze_model(
                preflight.models[model_name],
                preflight.audit_metadata,
                preflight.bootstrap_indices,
                protocol,
                progress=progress,
            )
            group_frames.append(groups)
            gap_frames.append(gaps)
            bootstrap_frames.append(bootstrap)
            runtime_rows.append({"model": model_name, **timing, "status": "PASS"})
            model_status[model_name] = "PASS"
            if progress is not None:
                progress(f"RQ3 model={model_name} stage=complete")

        _write_csv(temporary / "audit_metadata.csv", preflight.audit_metadata)
        _write_csv(temporary / "group_metrics.csv", pd.concat(group_frames, ignore_index=True))
        _write_csv(temporary / "gap_metrics.csv", pd.concat(gap_frames, ignore_index=True))
        _write_csv(temporary / "bootstrap_summary.csv", pd.concat(bootstrap_frames, ignore_index=True))
        _write_csv(temporary / "runtime.csv", pd.DataFrame(runtime_rows))
        status_payload = {
            "status": "PASS",
            "run_id": protocol["output"]["run_id"],
            "model_status": model_status,
            "model_fit_calls": 0,
            "prediction_calls": 0,
            "bootstrap_indices_generated": False,
            "target_orientation": "Approved-positive",
        }
        (temporary / "status.json").write_text(
            json.dumps(status_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        validation = validate_output_directory(
            temporary, protocol, expected_rows=len(preflight.audit_metadata)
        )
        temporary.replace(output_directory)
        return validation
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def run_rq3(
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
    *,
    project_root: Path = PROJECT_ROOT,
    progress: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    preflight = run_preflight(protocol, project_root=project_root)
    output_directory = project_root / protocol["output"]["root"]
    return run_from_preflight(preflight, protocol, output_directory, progress=progress)


if __name__ == "__main__":
    run_rq3()
