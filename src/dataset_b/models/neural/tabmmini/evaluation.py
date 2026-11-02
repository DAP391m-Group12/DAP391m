"""Canonical Run-scoped holdout evaluation interfaces for the locked TabMmini v1 route.

The entrypoints in this module receive an already-fitted preprocessor and model.
They deliberately expose only ``transform`` and prediction operations: holdout
labels never enter fitting, weighting, vocabulary construction, or model state.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from src.dataset_b.models.neural.plain_mlp.test_evaluation import (
    METRIC_NAMES,
    _calibration_curve,
    _metrics,
)
from src.utils.path import PROJECT_ROOT


DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "dataset_b" / "tabmmini"
DEFAULT_BOOTSTRAP_INDICES = (
    PROJECT_ROOT / "artifacts" / "dataset_b" / "rq1_test_bootstrap_indices.npz"
)
EXPECTED_BOOTSTRAP_REPLICATES = 1_000
THRESHOLD_GRID = np.arange(1, 1_000, dtype=np.float64) / 1_000.0
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class TabMMiniFinalEvaluationError(RuntimeError):
    """Raised when a TabMmini holdout or evaluation boundary is violated."""


@dataclass(frozen=True)
class TabMMiniFinalPaths:
    run_id: str
    artifact_root: Path

    @classmethod
    def for_run(
        cls, run_id: str, *, artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT
    ) -> "TabMMiniFinalPaths":
        if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
            raise ValueError("TabMmini run_id must be non-empty and path-safe.")
        return cls(run_id=run_id, artifact_root=Path(artifact_root))

    @property
    def working_root(self) -> Path:
        return self.artifact_root / "working" / self.run_id

    @property
    def final_root(self) -> Path:
        return self.artifact_root / "final" / self.run_id

    def stage(self, name: str, *, published: bool = False) -> Path:
        return (self.final_root if published else self.working_root) / name


@dataclass(frozen=True)
class TabMMiniHoldoutPopulation:
    X: Any
    y_approved: np.ndarray
    record_ids: tuple[str, ...]


PredictionRunner = Callable[[Any, Any, int], np.ndarray]


def as_two_class_probabilities(p_approved: Sequence[float] | np.ndarray) -> np.ndarray:
    """Derive evaluator columns ``(p_denied, p_approved)`` from the canonical vector."""
    approved = _validate_canonical_probabilities(p_approved)
    matrix = np.column_stack((np.float32(1.0) - approved, approved)).astype(
        np.float32, copy=False
    )
    if matrix.shape != (len(approved), 2) or not np.isfinite(matrix).all():
        raise TabMMiniFinalEvaluationError("Derived TabMmini probability matrix is invalid.")
    return matrix


def select_shared_threshold(
    labels: Sequence[int] | np.ndarray,
    p_approved: Sequence[float] | np.ndarray,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Apply the shared 999-candidate Balanced Accuracy and tie-break policy."""
    y = _validate_binary_labels(labels)
    probabilities = _validate_canonical_probabilities(p_approved, expected_rows=len(y))
    candidates = [
        {
            "threshold": float(threshold),
            "balanced_accuracy": float(
                balanced_accuracy_score(y, probabilities >= threshold)
            ),
        }
        for threshold in THRESHOLD_GRID
    ]
    best = max(row["balanced_accuracy"] for row in candidates)
    selected = min(
        (row for row in candidates if row["balanced_accuracy"] == best),
        key=lambda row: (abs(row["threshold"] - 0.5), row["threshold"]),
    )
    return selected, candidates


def run_tabmmini_validation(
    run_id: str,
    population: TabMMiniHoldoutPopulation,
    *,
    fitted_run_id: str,
    preprocessor: Any,
    model: Any,
    prediction_runner: PredictionRunner,
    effective_batch_size: int,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
) -> dict[str, Any]:
    """Transform and predict ordered Validation exactly once without fitting."""
    paths = TabMMiniFinalPaths.for_run(run_id, artifact_root=artifact_root)
    return _run_holdout_prediction(
        paths,
        "validation",
        population,
        fitted_run_id=fitted_run_id,
        preprocessor=preprocessor,
        model=model,
        prediction_runner=prediction_runner,
        effective_batch_size=effective_batch_size,
    )


def select_tabmmini_threshold(
    run_id: str, *, artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT
) -> dict[str, Any]:
    """Select and persist the shared threshold from the sole Validation prediction."""
    paths = TabMMiniFinalPaths.for_run(run_id, artifact_root=artifact_root)
    validation = validate_tabmmini_prediction_stage(paths, "validation")
    validation_root = paths.stage("validation")
    labels, probabilities, _ = _read_prediction_npz(
        validation_root / "validation_predictions.npz"
    )
    selected, candidates = select_shared_threshold(labels, probabilities)
    candidates_path = validation_root / "threshold_candidates.csv"
    selection_path = validation_root / "threshold_selection.json"
    manifest_path = validation_root / "manifest.json"
    working_candidates = validation_root / ".threshold_candidates.csv.working"
    working_selection = validation_root / ".threshold_selection.json.working"
    working_manifest = validation_root / ".manifest.json.working"
    guarded_paths = (
        candidates_path,
        selection_path,
        working_candidates,
        working_selection,
        working_manifest,
    )
    if any(path.exists() for path in guarded_paths):
        raise FileExistsError(
            "TabMmini Validation threshold is already locked or has a preserved attempt."
        )
    payload = {
        "schema_version": 1,
        "model": "tabmmini",
        "dataset": "dataset_b",
        "run_id": run_id,
        "stage": "validation_threshold_lock",
        "status": "LOCKED",
        "source_partition": validation["stage"],
        "source_prediction_file": "validation_predictions.npz",
        "canonical_probability": "p_approved",
        "selected_threshold": selected["threshold"],
        "balanced_accuracy": selected["balanced_accuracy"],
        "candidate_grid": {
            "start": 0.001,
            "stop": 0.999,
            "step": 0.001,
            "count": 999,
        },
        "objective": "maximize_balanced_accuracy",
        "tie_break": [
            "minimum_absolute_distance_to_0.5",
            "lower_threshold",
        ],
        "frozen_before_test": True,
        "execution": {
            "model_fit_calls": 0,
            "preprocessing_fit_calls": 0,
            "threshold_prediction_calls": 0,
            "validation_inference_calls": 1,
        },
    }
    locked_manifest = {
        **validation,
        "candidate_count": 999,
        "selected_threshold": selected["threshold"],
        "threshold_file": selection_path.name,
        "threshold_status": "LOCKED",
        "test_read": False,
    }
    pd.DataFrame(candidates, columns=("threshold", "balanced_accuracy")).to_csv(
        working_candidates, index=False
    )
    _write_json(working_selection, payload)
    _write_json(working_manifest, locked_manifest)
    os.replace(working_candidates, candidates_path)
    os.replace(working_selection, selection_path)
    os.replace(working_manifest, manifest_path)
    return validate_tabmmini_threshold(paths)


def run_tabmmini_test_evaluation(
    run_id: str,
    population: TabMMiniHoldoutPopulation,
    *,
    fitted_run_id: str,
    preprocessor: Any,
    model: Any,
    prediction_runner: PredictionRunner,
    effective_batch_size: int,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    bootstrap_indices_path: str | Path = DEFAULT_BOOTSTRAP_INDICES,
) -> dict[str, Any]:
    """Predict Test once, reuse locked indices, and compute the ten RQ1 metrics."""
    paths = TabMMiniFinalPaths.for_run(run_id, artifact_root=artifact_root)
    if fitted_run_id != run_id:
        raise TabMMiniFinalEvaluationError(
            "TabMmini fitted bundle and Test run_id lineage differ."
        )
    validate_tabmmini_prediction_stage(paths, "validation")
    threshold_payload = validate_tabmmini_threshold(paths)
    checked = _validate_population(population)
    threshold = float(threshold_payload["selected_threshold"])
    indices = load_locked_bootstrap_indices(bootstrap_indices_path, len(checked.y_approved))
    destination = paths.stage("test_evaluation")

    def writer(working: Path) -> dict[str, Any]:
        started = time.perf_counter()
        native = preprocessor.transform(checked.X)
        p_approved = _validate_canonical_probabilities(
            prediction_runner(model, native, effective_batch_size),
            expected_rows=len(checked.y_approved),
        )
        inference_seconds = time.perf_counter() - started
        _write_predictions(working, "test", checked, p_approved)
        point, invalid, counts = _metrics(
            checked.y_approved, p_approved.astype(np.float64), threshold
        )
        if invalid:
            raise TabMMiniFinalEvaluationError(
                f"TabMmini point metrics are invalid: {invalid}"
            )
        bootstrap_rows: list[dict[str, Any]] = []
        for replicate_id, positions in enumerate(indices):
            values, problems, _ = _metrics(
                checked.y_approved[positions],
                p_approved[positions].astype(np.float64),
                threshold,
            )
            bootstrap_rows.append(
                {
                    "replicate_id": replicate_id,
                    **values,
                    "invalid_metrics": json.dumps(
                        problems, ensure_ascii=False, sort_keys=True
                    ),
                }
            )
        bootstrap = pd.DataFrame(
            bootstrap_rows,
            columns=("replicate_id", *METRIC_NAMES, "invalid_metrics"),
        )
        bootstrap.to_csv(working / "rq1_bootstrap_metrics.csv", index=False)
        intervals: dict[str, dict[str, Any]] = {}
        for name in METRIC_NAMES:
            values = bootstrap[name].to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            intervals[name] = {
                "lower_2_5": float(np.quantile(finite, 0.025, method="linear"))
                if len(finite)
                else None,
                "upper_97_5": float(np.quantile(finite, 0.975, method="linear"))
                if len(finite)
                else None,
                "valid_replicates": int(len(finite)),
                "invalid_replicates": int(len(values) - len(finite)),
            }
        payload = {
            "schema_version": 1,
            "model_key": "tabmmini",
            "run_id": run_id,
            "stage": "test_evaluation",
            "status": "COMPLETE",
            "rows": len(checked.y_approved),
            "unique_record_ids": len(set(checked.record_ids)),
            "threshold": threshold,
            "canonical_probability": {
                "name": "p_approved",
                "dtype": "float32",
                "shape": [len(p_approved)],
            },
            "point_metrics": point,
            "confidence_intervals_95_percentile": intervals,
            "counts": counts,
            "calibration_curve": _calibration_curve(
                checked.y_approved, p_approved.astype(np.float64)
            ),
            "bootstrap_replicates": len(indices),
            "bootstrap_indices": str(Path(bootstrap_indices_path).resolve()),
            "test_inference_seconds": inference_seconds,
            "execution": {
                "model_fit_calls": 0,
                "preprocessing_fit_calls": 0,
                "preprocessing_transform_calls": 1,
                "test_inference_calls": 1,
                "bootstrap_fit_calls": 0,
                "bootstrap_model_calls": 0,
                "bootstrap_indices_regenerated": False,
            },
        }
        _write_json(working / "rq1_test_metrics.json", payload)
        return payload

    _publish_stage(destination, writer)
    return validate_tabmmini_test_evaluation(paths)


def validate_tabmmini_prediction_stage(
    paths: TabMMiniFinalPaths, partition: str, *, published: bool = False
) -> dict[str, Any]:
    if partition not in {"validation", "test"}:
        raise ValueError("TabMmini prediction partition must be validation or test.")
    stage = "test_evaluation" if partition == "test" else "validation"
    root = paths.stage(stage, published=published)
    manifest = _read_json(root / "manifest.json")
    labels, probabilities, record_ids = _read_prediction_npz(
        root / f"{partition}_predictions.npz"
    )
    execution = manifest.get("execution", {})
    inference_calls = execution.get(
        "test_inference_calls" if partition == "test" else "prediction_calls"
    )
    if (
        manifest.get("run_id") != paths.run_id
        or manifest.get("stage") != stage
        or manifest.get("rows") != len(labels)
        or manifest.get("unique_record_ids") != len(set(record_ids))
        or len(set(record_ids)) != len(record_ids)
        or execution.get("model_fit_calls") != 0
        or execution.get("preprocessing_fit_calls") != 0
        or execution.get("preprocessing_transform_calls") != 1
        or inference_calls != 1
    ):
        raise TabMMiniFinalEvaluationError(
            f"TabMmini {partition} prediction artifact is invalid."
        )
    _validate_canonical_probabilities(probabilities, expected_rows=len(labels))
    return manifest


def validate_tabmmini_threshold(
    paths: TabMMiniFinalPaths, *, published: bool = False
) -> dict[str, Any]:
    root = paths.stage("validation", published=published)
    manifest = _read_json(root / "manifest.json")
    payload = _read_json(root / "threshold_selection.json")
    candidates = pd.read_csv(root / "threshold_candidates.csv")
    if (
        payload.get("run_id") != paths.run_id
        or payload.get("stage") != "validation_threshold_lock"
        or payload.get("status") != "LOCKED"
        or payload.get("source_partition") != "validation"
        or payload.get("canonical_probability") != "p_approved"
        or payload.get("frozen_before_test") is not True
        or payload.get("candidate_grid")
        != {"start": 0.001, "stop": 0.999, "step": 0.001, "count": 999}
        or payload.get("objective") != "maximize_balanced_accuracy"
        or payload.get("tie_break")
        != ["minimum_absolute_distance_to_0.5", "lower_threshold"]
        or len(candidates) != 999
        or not np.array_equal(
            candidates["threshold"].to_numpy(dtype=np.float64), THRESHOLD_GRID
        )
        or manifest.get("threshold_status") != "LOCKED"
        or manifest.get("threshold_file") != "threshold_selection.json"
        or manifest.get("candidate_count") != 999
        or manifest.get("selected_threshold") != payload.get("selected_threshold")
    ):
        raise TabMMiniFinalEvaluationError("TabMmini threshold artifact is invalid.")
    return payload


def validate_tabmmini_test_evaluation(
    paths: TabMMiniFinalPaths, *, published: bool = False
) -> dict[str, Any]:
    root = paths.stage("test_evaluation", published=published)
    manifest = validate_tabmmini_prediction_stage(paths, "test", published=published)
    metrics = _read_json(root / "rq1_test_metrics.json")
    bootstrap = pd.read_csv(root / "rq1_bootstrap_metrics.csv")
    if (
        metrics.get("run_id") != paths.run_id
        or metrics.get("threshold") != manifest.get("threshold")
        or set(metrics.get("point_metrics", {})) != set(METRIC_NAMES)
        or len(bootstrap) != EXPECTED_BOOTSTRAP_REPLICATES
        or tuple(bootstrap.columns)
        != ("replicate_id", *METRIC_NAMES, "invalid_metrics")
        or metrics.get("execution", {}).get("bootstrap_indices_regenerated") is not False
    ):
        raise TabMMiniFinalEvaluationError("TabMmini Test evaluation artifact is invalid.")
    return metrics


def load_locked_bootstrap_indices(path: str | Path, expected_rows: int) -> np.ndarray:
    """Load the existing shared index matrix; this function has no generation path."""
    try:
        with np.load(Path(path), allow_pickle=False) as payload:
            indices = np.asarray(payload["indices"])
    except (OSError, ValueError, KeyError) as exc:
        raise TabMMiniFinalEvaluationError(
            "Cannot read the existing locked bootstrap indices."
        ) from exc
    if (
        indices.shape != (EXPECTED_BOOTSTRAP_REPLICATES, expected_rows)
        or indices.dtype != np.int32
        or np.any((indices < 0) | (indices >= expected_rows))
    ):
        raise TabMMiniFinalEvaluationError("Locked bootstrap indices are invalid.")
    return indices


def _run_holdout_prediction(
    paths: TabMMiniFinalPaths,
    partition: str,
    population: TabMMiniHoldoutPopulation,
    *,
    fitted_run_id: str,
    preprocessor: Any,
    model: Any,
    prediction_runner: PredictionRunner,
    effective_batch_size: int,
) -> dict[str, Any]:
    if partition != "validation":
        raise ValueError("This interface is reserved for Validation prediction.")
    if fitted_run_id != paths.run_id:
        raise TabMMiniFinalEvaluationError(
            "TabMmini fitted bundle and Validation run_id lineage differ."
        )
    checked = _validate_population(population)
    if isinstance(effective_batch_size, bool) or effective_batch_size not in {64, 128, 256}:
        raise ValueError("TabMmini holdout inference requires the smoke-locked batch size.")

    def writer(working: Path) -> dict[str, Any]:
        native = preprocessor.transform(checked.X)
        probabilities = _validate_canonical_probabilities(
            prediction_runner(model, native, effective_batch_size),
            expected_rows=len(checked.y_approved),
        )
        _write_predictions(working, partition, checked, probabilities)
        return {
            "schema_version": 1,
            "model_key": "tabmmini",
            "run_id": paths.run_id,
            "stage": partition,
            "status": "COMPLETE",
            "rows": len(checked.y_approved),
            "unique_record_ids": len(set(checked.record_ids)),
            "canonical_probability": {
                "name": "p_approved",
                "dtype": "float32",
                "shape": [len(probabilities)],
            },
            "execution": {
                "model_fit_calls": 0,
                "preprocessing_fit_calls": 0,
                "preprocessing_transform_calls": 1,
                "prediction_calls": 1,
            },
        }

    _publish_stage(paths.stage(partition), writer)
    return validate_tabmmini_prediction_stage(paths, partition)


def _validate_population(population: TabMMiniHoldoutPopulation) -> TabMMiniHoldoutPopulation:
    labels = _validate_binary_labels(population.y_approved)
    record_ids = tuple(str(value) for value in population.record_ids)
    try:
        rows = len(population.X)
    except TypeError as exc:
        raise ValueError("TabMmini holdout X must have a row count.") from exc
    if (
        rows != len(labels)
        or len(record_ids) != len(labels)
        or any(not value for value in record_ids)
        or len(set(record_ids)) != len(record_ids)
    ):
        raise ValueError("TabMmini holdout X, labels, and ordered record IDs must align.")
    return TabMMiniHoldoutPopulation(population.X, labels, record_ids)


def _validate_binary_labels(labels: Sequence[int] | np.ndarray) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 1 or not len(values) or set(np.unique(values).tolist()) != {0, 1}:
        raise ValueError("TabMmini evaluation population must contain both binary classes.")
    return values.astype(np.int8, copy=False)


def _validate_canonical_probabilities(
    values: Sequence[float] | np.ndarray, *, expected_rows: int | None = None
) -> np.ndarray:
    probabilities = np.asarray(values)
    if (
        probabilities.ndim != 1
        or (expected_rows is not None and len(probabilities) != expected_rows)
        or probabilities.dtype != np.float32
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0.0) | (probabilities > 1.0))
    ):
        raise TabMMiniFinalEvaluationError(
            "Canonical TabMmini p_approved must be finite float32 with shape [n]."
        )
    return probabilities


def _write_predictions(
    directory: Path,
    partition: str,
    population: TabMMiniHoldoutPopulation,
    probabilities: np.ndarray,
) -> None:
    np.savez_compressed(
        directory / f"{partition}_predictions.npz",
        record_ids=np.asarray(population.record_ids, dtype=str),
        y_approved=population.y_approved.astype(np.int8, copy=False),
        p_approved=probabilities,
    )
    pd.DataFrame(
        {
            "record_id": population.record_ids,
            "y_approved": population.y_approved,
            "p_approved": probabilities,
        }
    ).to_csv(directory / f"{partition}_predictions.csv", index=False)


def _read_prediction_npz(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            labels = np.asarray(payload["y_approved"])
            probabilities = np.asarray(payload["p_approved"])
            record_ids = tuple(str(value) for value in payload["record_ids"].tolist())
    except (OSError, ValueError, KeyError) as exc:
        raise TabMMiniFinalEvaluationError(f"Cannot read prediction artifact: {path}") from exc
    labels = _validate_binary_labels(labels)
    _validate_canonical_probabilities(probabilities, expected_rows=len(labels))
    if len(record_ids) != len(labels) or len(set(record_ids)) != len(record_ids):
        raise TabMMiniFinalEvaluationError("Prediction record IDs are invalid.")
    return labels, probabilities, record_ids


def _publish_stage(
    destination: Path, writer: Callable[[Path], Mapping[str, Any]]
) -> None:
    working = destination.with_name(destination.name + ".working")
    if destination.exists() or working.exists():
        raise FileExistsError(
            f"TabMmini stage already exists or has a preserved attempt: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    working.mkdir()
    payload = dict(writer(working))
    _write_json(working / "manifest.json", payload)
    if _read_json(working / "manifest.json") != payload:
        raise TabMMiniFinalEvaluationError("TabMmini stage manifest reopen failed.")
    os.replace(working, destination)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TabMMiniFinalEvaluationError(f"Cannot read JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise TabMMiniFinalEvaluationError(f"JSON artifact is not an object: {path}")
    return payload
