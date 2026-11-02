"""Canonical Final run-scoped Validation, threshold, and Test evaluation for RealMLP."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from src.dataset_b.models.neural.plain_mlp.test_evaluation import (
    METRIC_NAMES,
    _calibration_curve,
    _metrics,
)
from src.dataset_b.models.neural.realmlp.final_contract import (
    EXPECTED_ROWS,
    RealMLPFinalContractError,
    RealMLPFinalPaths,
    base_manifest,
    begin_stage,
    load_quality_gated_partition,
    publish_stage,
    read_json,
    transform_native_preprocessing,
    validate_manifest,
    validate_probabilities,
    write_json,
)
from src.dataset_b.models.neural.realmlp.final_training import validate_realmlp_full_train
from src.utils.path import PROJECT_ROOT


DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "dataset_b" / "realmlp"
DEFAULT_BOOTSTRAP_INDICES = (
    PROJECT_ROOT / "artifacts" / "dataset_b" / "rq1_test_bootstrap_indices.npz"
)


class RealMLPFinalEvaluationError(RealMLPFinalContractError):
    """Raised when final RealMLP holdout evaluation violates its boundary."""


def run_realmlp_validation(
    run_id: str, *, artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT
) -> dict[str, Any]:
    paths = RealMLPFinalPaths.for_run(run_id, artifact_root=artifact_root)
    validate_realmlp_full_train(paths)
    temporary, destination = begin_stage(paths, "validation")
    bundle = _load_bundle(paths)
    gated = load_quality_gated_partition("validation")
    prepared = transform_native_preprocessing(bundle["preprocessing"], gated)
    probabilities = validate_probabilities(
        bundle["estimator"].predict_proba(prepared.X), len(prepared.X)
    )
    _write_predictions(
        temporary / "validation_predictions.csv",
        prepared.record_ids,
        prepared.y,
        probabilities,
    )
    manifest = {
        **base_manifest(run_id, "validation"),
        "rows": len(prepared.X),
        "unique_record_ids": len(set(prepared.record_ids)),
        "quality_gate": gated.quality_audit.as_dict(),
        "execution": {
            "model_fit_calls": 0,
            "preprocessing_fit_calls": 0,
            "preprocessing_transform_calls": 1,
            "predict_proba_calls": 1,
        },
    }
    write_json(temporary / "manifest.json", manifest)
    publish_stage(temporary, destination)
    return validate_realmlp_validation(paths)


def select_realmlp_threshold(
    run_id: str, *, artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT
) -> dict[str, Any]:
    paths = RealMLPFinalPaths.for_run(run_id, artifact_root=artifact_root)
    validate_realmlp_validation(paths)
    temporary, destination = begin_stage(paths, "threshold")
    frame = pd.read_csv(paths.stage("validation") / "validation_predictions.csv", dtype={"record_id": str})
    labels = frame["y_approved"].to_numpy(dtype=np.int8)
    probabilities = frame["p_approved"].to_numpy(dtype=np.float64)
    candidates: list[dict[str, float]] = []
    for value in range(1, 1000):
        threshold = value / 1000
        score = float(balanced_accuracy_score(labels, probabilities >= threshold))
        candidates.append({"threshold": threshold, "balanced_accuracy": score})
    best = max(row["balanced_accuracy"] for row in candidates)
    selected = min(
        (row for row in candidates if row["balanced_accuracy"] == best),
        key=lambda row: (abs(row["threshold"] - 0.5), row["threshold"]),
    )
    pd.DataFrame(candidates).to_csv(temporary / "threshold_candidates.csv", index=False)
    result = {
        **base_manifest(run_id, "threshold"),
        "selected_threshold": selected["threshold"],
        "balanced_accuracy": selected["balanced_accuracy"],
        "candidate_count": len(candidates),
        "objective": "maximize_validation_balanced_accuracy",
        "tie_break": ["minimum_absolute_distance_to_0.5", "lower_threshold"],
        "execution": {"model_fit_calls": 0, "predict_proba_calls": 0},
    }
    write_json(temporary / "threshold_selection.json", result)
    write_json(temporary / "manifest.json", result)
    publish_stage(temporary, destination)
    return validate_realmlp_threshold(paths)


def run_realmlp_test_evaluation(
    run_id: str,
    *,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    bootstrap_indices_path: str | Path = DEFAULT_BOOTSTRAP_INDICES,
) -> dict[str, Any]:
    paths = RealMLPFinalPaths.for_run(run_id, artifact_root=artifact_root)
    validate_realmlp_full_train(paths)
    threshold_manifest = validate_realmlp_threshold(paths)
    temporary, destination = begin_stage(paths, "test_evaluation")
    threshold = float(threshold_manifest["selected_threshold"])
    bundle = _load_bundle(paths)
    gated = load_quality_gated_partition("test")
    prepared = transform_native_preprocessing(bundle["preprocessing"], gated)
    started = time.perf_counter()
    probabilities = validate_probabilities(
        bundle["estimator"].predict_proba(prepared.X), len(prepared.X)
    )
    inference_seconds = time.perf_counter() - started
    _write_predictions(
        temporary / "test_predictions.csv",
        prepared.record_ids,
        prepared.y,
        probabilities,
        threshold=threshold,
    )
    p_approved = probabilities[:, 1]
    point, invalid, counts = _metrics(prepared.y, p_approved, threshold)
    if invalid:
        raise RealMLPFinalEvaluationError(f"RealMLP point metrics are invalid: {invalid}")
    indices = _load_locked_bootstrap(bootstrap_indices_path, len(prepared.X))
    bootstrap_rows: list[dict[str, Any]] = []
    for replicate_id, positions in enumerate(indices):
        values, problems, _ = _metrics(
            prepared.y[positions], p_approved[positions], threshold
        )
        bootstrap_rows.append(
            {
                "replicate_id": replicate_id,
                **values,
                "invalid_metrics": json.dumps(problems, ensure_ascii=False, sort_keys=True),
            }
        )
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(temporary / "rq1_bootstrap_metrics.csv", index=False)
    intervals: dict[str, dict[str, Any]] = {}
    for name in METRIC_NAMES:
        values = bootstrap[name].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        intervals[name] = {
            "lower_2_5": float(np.quantile(finite, 0.025, method="linear")) if len(finite) else None,
            "upper_97_5": float(np.quantile(finite, 0.975, method="linear")) if len(finite) else None,
            "valid_replicates": int(len(finite)),
            "invalid_replicates": int(len(values) - len(finite)),
        }
    result = {
        **base_manifest(run_id, "test_evaluation"),
        "rows": len(prepared.X),
        "unique_record_ids": len(set(prepared.record_ids)),
        "threshold": threshold,
        "point_metrics": point,
        "confidence_intervals_95_percentile": intervals,
        "counts": counts,
        "calibration_curve": _calibration_curve(prepared.y, p_approved),
        "bootstrap_replicates": len(bootstrap),
        "bootstrap_indices": str(Path(bootstrap_indices_path)),
        "test_inference_seconds": inference_seconds,
        "quality_gate": gated.quality_audit.as_dict(),
        "execution": {
            "model_fit_calls": 0,
            "preprocessing_fit_calls": 0,
            "preprocessing_transform_calls": 1,
            "predict_proba_calls": 1,
            "bootstrap_indices_regenerated": False,
        },
    }
    write_json(temporary / "rq1_test_metrics.json", result)
    write_json(temporary / "manifest.json", result)
    publish_stage(temporary, destination)
    return validate_realmlp_test_evaluation(paths)


def validate_realmlp_validation(paths: RealMLPFinalPaths) -> dict[str, Any]:
    root = paths.stage("validation")
    payload = validate_manifest(
        read_json(root / "manifest.json"), run_id=paths.run_id, stage="validation"
    )
    frame = pd.read_csv(root / "validation_predictions.csv", dtype={"record_id": str})
    if (
        len(frame) != EXPECTED_ROWS["validation"]
        or frame["record_id"].nunique() != EXPECTED_ROWS["validation"]
        or payload.get("execution", {}).get("model_fit_calls") != 0
        or payload.get("execution", {}).get("predict_proba_calls") != 1
    ):
        raise RealMLPFinalEvaluationError("RealMLP Validation artifact is invalid.")
    return payload


def validate_realmlp_threshold(paths: RealMLPFinalPaths) -> dict[str, Any]:
    payload = validate_manifest(
        read_json(paths.stage("threshold") / "manifest.json"),
        run_id=paths.run_id,
        stage="threshold",
    )
    value = payload.get("selected_threshold")
    if not isinstance(value, (int, float)) or not 0.0 < float(value) < 1.0:
        raise RealMLPFinalEvaluationError("RealMLP Validation threshold is invalid.")
    return payload


def validate_realmlp_test_evaluation(paths: RealMLPFinalPaths) -> dict[str, Any]:
    root = paths.stage("test_evaluation")
    payload = validate_manifest(
        read_json(root / "manifest.json"), run_id=paths.run_id, stage="test_evaluation"
    )
    frame = pd.read_csv(root / "test_predictions.csv", dtype={"record_id": str})
    bootstrap = pd.read_csv(root / "rq1_bootstrap_metrics.csv")
    if (
        len(frame) != EXPECTED_ROWS["test"]
        or frame["record_id"].nunique() != EXPECTED_ROWS["test"]
        or len(bootstrap) != 1_000
        or payload.get("threshold") != validate_realmlp_threshold(paths).get("selected_threshold")
        or payload.get("execution", {}).get("model_fit_calls") != 0
        or payload.get("execution", {}).get("predict_proba_calls") != 1
    ):
        raise RealMLPFinalEvaluationError("RealMLP Test artifact is invalid.")
    return payload


def _load_bundle(paths: RealMLPFinalPaths) -> dict[str, Any]:
    bundle = joblib.load(paths.stage("full_train") / "fitted_model.joblib")
    if not isinstance(bundle, dict) or set(bundle) != {"estimator", "preprocessing", "scope"}:
        raise RealMLPFinalEvaluationError("RealMLP fitted bundle is invalid.")
    return bundle


def _load_locked_bootstrap(path: str | Path, expected_rows: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        indices = payload["indices"]
    if (
        indices.shape != (1_000, expected_rows)
        or indices.dtype != np.int32
        or np.any((indices < 0) | (indices >= expected_rows))
    ):
        raise RealMLPFinalEvaluationError("Locked bootstrap indices are invalid.")
    return indices


def _write_predictions(
    path: Path,
    record_ids: Sequence[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float | None = None,
) -> None:
    columns = ["record_id", "y_approved", "p_denied", "p_approved"]
    if threshold is not None:
        columns.extend(("predicted_approved", "predicted_denied"))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for record_id, label, row in zip(record_ids, labels, probabilities, strict=True):
            values: list[Any] = [record_id, int(label), repr(float(row[0])), repr(float(row[1]))]
            if threshold is not None:
                approved = int(row[1] >= threshold)
                values.extend((approved, 1 - approved))
            writer.writerow(values)
