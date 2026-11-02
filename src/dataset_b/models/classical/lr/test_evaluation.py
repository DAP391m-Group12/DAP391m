"""Run and validate LR-09 Test/RQ1 evaluation from frozen LR artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from scipy.optimize import brentq, minimize, root
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, roc_auc_score

from src.dataset_b.models.classical.lr.validation import (
    DEFAULT_OUTPUT_DIR as DEFAULT_VALIDATION_DIR,
    RUN_ID,
    validate_lr_validation_threshold,
)
from src.dataset_b.data import DatasetBInputQualityGate, DatasetBSplitLoader
from src.utils.path import CONFIG_DIR, PROJECT_ROOT


CONTRACT_ID = "LR-09"
EXPECTED_TEST_ROWS = 43_996
BOOTSTRAP_REPLICATES = 1_000
CALIBRATION_SLOPE_DECIMALS = 6
BOOTSTRAP_RECOMPUTATION_ATOL = 1e-14
DEFAULT_RUN_DIR = PROJECT_ROOT / "artifacts/dataset_b/lr/working" / RUN_ID
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR / "test"
DEFAULT_SPLIT_MANIFEST_PATH = PROJECT_ROOT / "artifacts/dataset_b/split_v1.json"
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "data/dataset_b/split_v1"
DEFAULT_FEATURE_CONTRACT_PATH = CONFIG_DIR / "dataset_b/feature_set_v1.yaml"
DEFAULT_MODEL_PROTOCOL_PATH = CONFIG_DIR / "dataset_b/model_protocol_v1.yaml"
DEFAULT_BOOTSTRAP_INDICES_PATH = PROJECT_ROOT / "artifacts/dataset_b/rq1_test_bootstrap_indices.npz"
DEFAULT_BOOTSTRAP_MANIFEST_PATH = PROJECT_ROOT / "artifacts/dataset_b/rq1_test_bootstrap_manifest.json"
HISTORICAL_LR_DIR = PROJECT_ROOT / "artifacts/dataset_b/lr_baseline_v1"

PREDICTION_COLUMNS = (
    "record_id",
    "y_approved",
    "p_approved",
    "p_denied",
    "predicted_approved",
    "predicted_denied",
)
METRIC_NAMES = (
    "roc_auc_approved",
    "average_precision_denied",
    "precision_denied",
    "recall_denied",
    "f1_denied",
    "recall_approved",
    "balanced_accuracy",
    "brier_approved",
    "calibration_intercept",
    "calibration_slope",
)
BOOTSTRAP_COLUMNS = ("replicate_id", *METRIC_NAMES, "invalid_metrics")
PAYLOAD_FILES = {
    "test_predictions.csv",
    "rq1_test_metrics.json",
    "rq1_bootstrap_metrics.csv",
}
FINAL_FILES = PAYLOAD_FILES | {"run_manifest.json", "COMPLETED"}


class LRTestEvaluationError(RuntimeError):
    """Raised when LR-09 violates its frozen Test boundary."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LRTestEvaluationError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LRTestEvaluationError(f"Cannot read JSON artifact: {path}") from exc
    _require(isinstance(payload, dict), f"JSON artifact must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise LRTestEvaluationError(f"Cannot hash required artifact: {path}") from exc
    return digest.hexdigest()


def _sha256_ids(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _resolved_paths(run_dir: Path, output_dir: Path) -> dict[str, Path]:
    root = run_dir.resolve(strict=True)
    destination = output_dir.resolve(strict=False)
    expected = (root / "test").resolve(strict=False)
    _require(destination == expected, "LR-09 output must be current run_dir/test.")
    _require(destination.parent == root, "LR-09 output escaped the current run directory.")
    _require(HISTORICAL_LR_DIR.resolve(strict=False) not in destination.parents, "Historical LR output is prohibited.")
    paths = {
        "run_dir": root,
        "output_dir": destination,
        "temporary_dir": destination.with_name(destination.name + ".tmp"),
        "training_manifest": root / "manifest.json",
        "fitted_pipeline": root / "checkpoints/fitted_pipeline.joblib",
        "validation_dir": root / "validation",
        "validation_manifest": root / "validation/manifest.json",
        "threshold": root / "validation/threshold_selection.json",
    }
    for name in ("training_manifest", "fitted_pipeline", "validation_manifest", "threshold"):
        _require(paths[name].is_file(), f"Missing frozen LR source: {paths[name]}")
    return paths


def _validate_frozen_lineage(paths: Mapping[str, Path]) -> tuple[dict[str, Any], dict[str, Any], float]:
    training = _read_json(paths["training_manifest"])
    _require(training.get("status") == "PASS" and training.get("contract_id") == "LR-07", "LR-07 source is not frozen PASS.")
    _require(training.get("run_id") == RUN_ID and training.get("model_key") == "lr", "LR-07 lineage differs.")
    _require(training.get("schemas", {}).get("encoded_feature_count") == 62, "LR-07 source is not 62-feature LR.")
    _require(training.get("validation_accessed") is False and training.get("test_accessed") is False, "LR-07 source crossed a holdout boundary.")
    hashes = training.get("artifact_sha256", {})
    _require(hashes.get("checkpoints/fitted_pipeline.joblib") == _sha256_file(paths["fitted_pipeline"]), "Frozen pipeline hash differs from LR-07 manifest.")

    validation = validate_lr_validation_threshold(paths["validation_dir"])
    selection = _read_json(paths["threshold"])
    _require(validation.get("run_id") == selection.get("run_id") == RUN_ID, "LR-08 run_id lineage differs.")
    _require(validation.get("status") == selection.get("status") == "PASS", "LR-08 is not frozen PASS.")
    _require(selection.get("contract_id") == "LR-08", "Frozen threshold is not LR-08.")
    threshold = float(selection.get("selected_threshold"))
    _require(0.0 < threshold < 1.0, "Frozen Validation threshold is invalid.")
    _require(validation.get("selected_threshold") == threshold, "LR-08 threshold lineage differs.")
    _require(selection.get("source_prediction_sha256") == validation.get("artifact_sha256", {}).get("validation_predictions.csv"), "LR-08 threshold is not tied to frozen predictions.")
    _require(selection.get("lineage_sha256", {}).get("fitted_pipeline_sha256") == _sha256_file(paths["fitted_pipeline"]), "LR-08 threshold uses another fitted pipeline.")
    return training, selection, threshold


def _load_frozen_pipeline(path: Path) -> Any:
    try:
        pipeline = joblib.load(path)
    except Exception as exc:
        raise LRTestEvaluationError("Cannot load frozen LR full-Train pipeline.") from exc
    _require(tuple(getattr(pipeline, "named_steps", {})) == ("preprocess", "estimator"), "Frozen LR pipeline steps differ.")
    _require(np.array_equal(getattr(pipeline.named_steps["estimator"], "classes_", None), np.asarray([0, 1])), "Frozen LR class order differs.")
    return pipeline


def _predict_test_once(pipeline: Any, frame: Any) -> np.ndarray:
    probabilities = np.asarray(pipeline.predict_proba(frame), dtype=np.float64)
    _require(probabilities.shape == (len(frame), 2), "Test probability matrix has the wrong shape.")
    _require(np.isfinite(probabilities).all(), "Test probabilities are not finite.")
    _require(np.all((probabilities >= 0.0) & (probabilities <= 1.0)), "Test probabilities are outside [0, 1].")
    _require(np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-12), "Test probability rows do not sum to one.")
    return probabilities[:, 1]


def _calibration(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float, dict[str, str]]:
    logits = logit(np.clip(probabilities, 1e-15, 1 - 1e-15))
    targets = labels.astype(np.float64)
    invalid: dict[str, str] = {}
    intercept = float("nan")
    slope = float("nan")
    try:
        intercept = float(brentq(lambda value: float(np.mean(expit(logits + value) - targets)), -80, 80))
    except (ArithmeticError, ValueError, RuntimeError) as exc:
        invalid["calibration_intercept"] = str(exc)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = theta[0] + theta[1] * logits
        error = expit(eta) - targets
        return float(np.mean(np.logaddexp(0, eta) - targets * eta)), np.asarray([np.mean(error), np.mean(error * logits)], dtype=np.float64)

    try:
        fit = minimize(objective, [0.0, 1.0], method="L-BFGS-B", jac=True, options={"maxiter": 100, "ftol": 1e-12, "gtol": 1e-9})
        theta = fit.x
        if not fit.success or not np.isfinite(theta).all():
            theta = root(lambda values: objective(values)[1], theta if np.isfinite(theta).all() else [0.0, 1.0], method="hybr", tol=1e-10).x
        if not np.isfinite(theta).all() or np.max(np.abs(objective(theta)[1])) > 1e-7:
            raise ArithmeticError("Calibration score equations did not converge.")
        slope = float(theta[1])
    except (ArithmeticError, ValueError, RuntimeError) as exc:
        invalid["calibration_slope"] = str(exc)
    return intercept, slope, invalid


def _metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> tuple[dict[str, float], dict[str, str], dict[str, int]]:
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predicted = probabilities >= threshold
    approved = labels == 1
    denied = ~approved
    tp, fn = int(np.count_nonzero(approved & predicted)), int(np.count_nonzero(approved & ~predicted))
    tn, fp = int(np.count_nonzero(denied & ~predicted)), int(np.count_nonzero(denied & predicted))
    n_approved, n_denied = tp + fn, tn + fp
    if n_approved == 0 or n_denied == 0:
        raise ArithmeticError("Both true classes are required.")
    values = {
        "roc_auc_approved": float(roc_auc_score(labels, probabilities)),
        "average_precision_denied": float(average_precision_score(1 - labels, 1 - probabilities)),
        "precision_denied": tn / (tn + fn) if tn + fn else float("nan"),
        "recall_denied": tn / n_denied,
        "f1_denied": 2 * tn / (2 * tn + fn + fp),
        "recall_approved": tp / n_approved,
        "balanced_accuracy": (tp / n_approved + tn / n_denied) / 2,
        "brier_approved": float(np.mean((probabilities - labels) ** 2)),
    }
    values["calibration_intercept"], values["calibration_slope"], invalid = _calibration(labels, probabilities)
    if not np.isfinite(values["precision_denied"]):
        invalid["precision_denied"] = "no_predicted_denied"
    for name, value in values.items():
        if not np.isfinite(value):
            invalid.setdefault(name, "nonfinite_metric")
    return values, invalid, {"tp_approved": tp, "fn_approved": fn, "tn_denied": tn, "fp_denied": fp, "n_approved": n_approved, "n_denied": n_denied}


def _calibration_curve(labels: np.ndarray, probabilities: np.ndarray) -> list[dict[str, Any]]:
    edges = np.unique(np.quantile(probabilities, np.linspace(0, 1, 11)))
    if len(edges) < 2:
        raise ArithmeticError("Calibration curve needs at least two unique edges.")
    bin_ids = np.searchsorted(edges[1:-1], probabilities, side="right")
    rows = []
    for index in range(len(edges) - 1):
        mask = bin_ids == index
        if mask.any():
            rows.append({"bin": index, "lower_edge": float(edges[index]), "upper_edge": float(edges[index + 1]), "count": int(mask.sum()), "mean_p_approved": float(probabilities[mask].mean()), "observed_approval_rate": float(labels[mask].mean())})
    _require(sum(row["count"] for row in rows) == len(labels), "Calibration bins do not cover Test.")
    return rows


def _load_bootstrap(indices_path: Path, manifest_path: Path, labels: np.ndarray, ids: Sequence[str]) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        with np.load(indices_path, allow_pickle=False) as loaded:
            indices = loaded["indices"]
    except Exception as exc:
        raise LRTestEvaluationError("Cannot read shared bootstrap indices.") from exc
    manifest = _read_json(manifest_path)
    n_denied = int(np.count_nonzero(labels == 0))
    _require(_sha256_file(indices_path) == manifest.get("indices_sha256"), "Shared bootstrap index hash differs.")
    _require(manifest.get("test_record_id_sha256") == _sha256_ids(ids), "Test IDs differ from shared bootstrap contract.")
    _require(indices.shape == (BOOTSTRAP_REPLICATES, EXPECTED_TEST_ROWS) and indices.dtype == np.int32, "Shared bootstrap shape or dtype differs.")
    _require(np.all((indices >= 0) & (indices < EXPECTED_TEST_ROWS)), "Shared bootstrap indices are outside Test bounds.")
    _require(np.all(labels[indices[:, :n_denied]] == 0) and np.all(labels[indices[:, n_denied:]] == 1), "Shared bootstrap class stratification differs.")
    return indices, manifest


def _write_predictions(path: Path, ids: Sequence[str], labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(PREDICTION_COLUMNS)
        for record_id, label, probability in zip(ids, labels, probabilities, strict=True):
            predicted = int(probability >= threshold)
            writer.writerow((record_id, int(label), repr(float(probability)), repr(float(1.0 - probability)), predicted, 1 - predicted))


def _read_predictions(path: Path) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            _require(tuple(reader.fieldnames or ()) == PREDICTION_COLUMNS, "Test prediction columns differ.")
            rows = list(reader)
        ids = tuple(row["record_id"] for row in rows)
        labels = np.asarray([int(row["y_approved"]) for row in rows], dtype=np.int8)
        approved = np.asarray([float(row["p_approved"]) for row in rows], dtype=np.float64)
        denied = np.asarray([float(row["p_denied"]) for row in rows], dtype=np.float64)
        predicted = np.asarray([int(row["predicted_approved"]) for row in rows], dtype=np.int8)
    except (OSError, KeyError, ValueError) as exc:
        raise LRTestEvaluationError("Cannot read Test prediction artifact.") from exc
    return ids, labels, approved, denied, predicted


def _bootstrap_rows(labels: np.ndarray, probabilities: np.ndarray, threshold: float, indices: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for replicate_id, row_indices in enumerate(indices):
        values, invalid, _ = _metrics(labels[row_indices], probabilities[row_indices], threshold)
        rows.append({"replicate_id": replicate_id, **values, "invalid_metrics": json.dumps(invalid, ensure_ascii=False, sort_keys=True)})
    return rows


def _write_bootstrap(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BOOTSTRAP_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_bootstrap(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            _require(tuple(reader.fieldnames or ()) == BOOTSTRAP_COLUMNS, "Bootstrap columns differ.")
            return [{"replicate_id": int(row["replicate_id"]), **{name: float(row[name]) for name in METRIC_NAMES}, "invalid_metrics": row["invalid_metrics"]} for row in reader]
    except (OSError, KeyError, ValueError) as exc:
        raise LRTestEvaluationError("Cannot read bootstrap metric artifact.") from exc


def _intervals(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    result: dict[str, dict[str, float | int | None]] = {}
    for name in METRIC_NAMES:
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        valid = values[np.isfinite(values)]
        result[name] = {
            "lower_2_5": float(np.quantile(valid, 0.025, method="linear")) if len(valid) else None,
            "upper_97_5": float(np.quantile(valid, 0.975, method="linear")) if len(valid) else None,
            "valid_replicates": int(len(valid)),
            "invalid_replicates": int(len(values) - len(valid)),
        }
    return result


def _bootstrap_metric_matches(name: str, stored: float, recomputed: float) -> bool:
    """Apply the locked numerical comparison policy to one bootstrap value."""

    if not np.isfinite(stored) or not np.isfinite(recomputed):
        return bool(np.isfinite(stored) == np.isfinite(recomputed))
    if name == "calibration_slope":
        return bool(round(stored, CALIBRATION_SLOPE_DECIMALS) == round(recomputed, CALIBRATION_SLOPE_DECIMALS))
    return bool(np.isclose(stored, recomputed, rtol=0.0, atol=BOOTSTRAP_RECOMPUTATION_ATOL))


def _lineage(paths: Mapping[str, Path], ids: Sequence[str], *, split_manifest_path: Path, split_dir: Path, feature_contract_path: Path, model_protocol_path: Path, bootstrap_indices_path: Path, bootstrap_manifest_path: Path) -> dict[str, str]:
    return {
        "training_manifest_sha256": _sha256_file(paths["training_manifest"]),
        "fitted_pipeline_sha256": _sha256_file(paths["fitted_pipeline"]),
        "validation_manifest_sha256": _sha256_file(paths["validation_manifest"]),
        "validation_threshold_sha256": _sha256_file(paths["threshold"]),
        "split_manifest_sha256": _sha256_file(split_manifest_path),
        "test_source_sha256": _sha256_file(split_dir / "test.csv"),
        "feature_contract_sha256": _sha256_file(feature_contract_path),
        "model_protocol_sha256": _sha256_file(model_protocol_path),
        "bootstrap_manifest_sha256": _sha256_file(bootstrap_manifest_path),
        "bootstrap_indices_sha256": _sha256_file(bootstrap_indices_path),
        "test_record_ids_sha256": _sha256_ids(ids),
    }


def _validate_artifact(directory: Path, *, expected_ids: Sequence[str] | None = None, expected_labels: np.ndarray | None = None, expected_lineage: Mapping[str, str] | None = None, expected_bootstrap_indices: np.ndarray | None = None) -> dict[str, Any]:
    _require(directory.is_dir(), f"LR-09 artifact directory is missing: {directory}")
    _require({path.name for path in directory.iterdir() if path.is_file()} == FINAL_FILES and not any(path.is_dir() for path in directory.iterdir()), "LR-09 artifact set differs.")
    _require((directory / "COMPLETED").read_text(encoding="utf-8") == f"{CONTRACT_ID} PASS\n", "LR-09 completion marker differs.")
    manifest, metrics = _read_json(directory / "run_manifest.json"), _read_json(directory / "rq1_test_metrics.json")
    hashes = manifest.get("artifact_sha256")
    _require(isinstance(hashes, dict) and set(hashes) == PAYLOAD_FILES, "LR-09 artifact hash coverage differs.")
    for name, expected_hash in hashes.items():
        _require(_sha256_file(directory / name) == expected_hash, f"LR-09 artifact hash mismatch: {name}")
    ids, labels, probabilities, denied, predicted = _read_predictions(directory / "test_predictions.csv")
    _require(len(ids) == len(set(ids)) == EXPECTED_TEST_ROWS, "LR-09 Test ID count differs.")
    _require(set(np.unique(labels).tolist()) == {0, 1}, "LR-09 Test labels are invalid.")
    _require(np.isfinite(probabilities).all() and np.all((probabilities >= 0.0) & (probabilities <= 1.0)), "LR-09 probabilities are invalid.")
    threshold = float(metrics.get("threshold"))
    _require(np.allclose(denied, 1.0 - probabilities, rtol=0.0, atol=1e-15), "LR-09 denied probabilities differ.")
    _require(np.array_equal(predicted, (probabilities >= threshold).astype(np.int8)), "LR-09 prediction labels differ from frozen threshold.")
    point, invalid, counts = _metrics(labels, probabilities, threshold)
    _require(not invalid and counts == metrics.get("counts"), "LR-09 point metrics are invalid.")
    _require(set(metrics.get("point_metrics", {})) == set(point) and all(np.isclose(point[name], metrics["point_metrics"][name], rtol=0.0, atol=1e-12) for name in point), "LR-09 point metrics are not reproducible.")
    _require(metrics.get("calibration_curve") == _calibration_curve(labels, probabilities), "LR-09 calibration curve is not reproducible.")
    rows = _read_bootstrap(directory / "rq1_bootstrap_metrics.csv")
    _require(len(rows) == BOOTSTRAP_REPLICATES and [row["replicate_id"] for row in rows] == list(range(BOOTSTRAP_REPLICATES)), "LR-09 bootstrap replicate IDs differ.")
    _require(metrics.get("confidence_intervals_95_percentile") == _intervals(rows), "LR-09 bootstrap confidence intervals differ.")
    if expected_bootstrap_indices is not None:
        recomputed_rows = _bootstrap_rows(labels, probabilities, threshold, expected_bootstrap_indices)
        _require(len(recomputed_rows) == len(rows), "LR-09 bootstrap recomputation row count differs.")
        for stored, recomputed in zip(rows, recomputed_rows, strict=True):
            _require(stored["replicate_id"] == recomputed["replicate_id"], "LR-09 bootstrap replicate ID differs on recomputation.")
            _require(stored["invalid_metrics"] == recomputed["invalid_metrics"], "LR-09 bootstrap invalid payload differs on recomputation.")
            for name in METRIC_NAMES:
                _require(_bootstrap_metric_matches(name, stored[name], recomputed[name]), f"LR-09 bootstrap metric differs on recomputation: {name}.")
    _require(manifest.get("contract_id") == metrics.get("contract_id") == CONTRACT_ID and manifest.get("status") == metrics.get("status") == "PASS", "LR-09 status or contract differs.")
    _require(manifest.get("run_id") == metrics.get("run_id") == RUN_ID, "LR-09 run_id lineage differs.")
    _require(manifest.get("rows") == metrics.get("rows") == EXPECTED_TEST_ROWS, "LR-09 row count differs.")
    _require(manifest.get("threshold") == metrics.get("threshold"), "LR-09 threshold differs.")
    _require(manifest.get("lineage_sha256") == metrics.get("lineage_sha256"), "LR-09 lineage payload differs.")
    _require(metrics.get("prediction_sha256") == _sha256_file(directory / "test_predictions.csv") and metrics.get("bootstrap_replicates_sha256") == _sha256_file(directory / "rq1_bootstrap_metrics.csv"), "LR-09 payload hash differs.")
    _require(metrics.get("execution") == {"model_fit_calls": 0, "preprocessing_fit_calls": 0, "predict_proba_calls": 1}, "LR-09 execution boundary differs.")
    _require(metrics.get("data_boundary") == {"source_partition": "test", "model_fit_performed": False, "preprocessing_fit_performed": False, "threshold_changed": False, "shared_bootstrap_regenerated": False}, "LR-09 data boundary differs.")
    if expected_ids is not None:
        _require(tuple(ids) == tuple(str(value) for value in expected_ids), "LR-09 Test ID order differs from source.")
        _require(expected_labels is not None and np.array_equal(labels, expected_labels), "LR-09 Test labels differ from source.")
        _require(expected_lineage is not None and manifest.get("lineage_sha256") == dict(expected_lineage), "LR-09 source lineage differs.")
    return manifest


def run_lr_test_evaluation(output_dir: str | Path = DEFAULT_OUTPUT_DIR, *, run_dir: str | Path = DEFAULT_RUN_DIR, split_manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST_PATH, split_dir: str | Path = DEFAULT_SPLIT_DIR, feature_contract_path: str | Path = DEFAULT_FEATURE_CONTRACT_PATH, model_protocol_path: str | Path = DEFAULT_MODEL_PROTOCOL_PATH, bootstrap_indices_path: str | Path = DEFAULT_BOOTSTRAP_INDICES_PATH, bootstrap_manifest_path: str | Path = DEFAULT_BOOTSTRAP_MANIFEST_PATH) -> dict[str, Any]:
    """Predict ordered Test once, then freeze LR RQ1 metrics and bootstrap CIs."""

    paths = _resolved_paths(Path(run_dir), Path(output_dir))
    if paths["output_dir"].exists():
        return {**_validate_artifact(paths["output_dir"]), "resume_action": "SKIPPED_VALID_COMPLETED"}
    _require(not paths["temporary_dir"].exists(), f"Temporary directory was preserved: {paths['temporary_dir']}")
    training, _selection, threshold = _validate_frozen_lineage(paths)
    pipeline = _load_frozen_pipeline(paths["fitted_pipeline"])
    loader = DatasetBSplitLoader(manifest_path=split_manifest_path, split_dir=split_dir, feature_contract_path=feature_contract_path)
    gated = DatasetBInputQualityGate().apply(loader.load("test"))
    ids, labels = tuple(gated.record_ids.astype(str).tolist()), gated.y.to_numpy(dtype=np.int8)
    _require(len(ids) == len(set(ids)) == EXPECTED_TEST_ROWS and set(np.unique(labels).tolist()) == {0, 1}, "Test quality gate changed the required Test population.")
    indices, bootstrap_manifest = _load_bootstrap(Path(bootstrap_indices_path), Path(bootstrap_manifest_path), labels, ids)
    started = time.perf_counter()
    probabilities = _predict_test_once(pipeline, gated.X.reset_index(drop=True))
    inference_seconds = time.perf_counter() - started
    point, invalid, counts = _metrics(labels, probabilities, threshold)
    _require(not invalid, f"LR-09 Test point metrics are invalid: {invalid}")
    bootstrap = _bootstrap_rows(labels, probabilities, threshold, indices)
    intervals = _intervals(bootstrap)
    lineage = _lineage(paths, ids, split_manifest_path=Path(split_manifest_path), split_dir=Path(split_dir), feature_contract_path=Path(feature_contract_path), model_protocol_path=Path(model_protocol_path), bootstrap_indices_path=Path(bootstrap_indices_path), bootstrap_manifest_path=Path(bootstrap_manifest_path))
    paths["temporary_dir"].mkdir(parents=False)
    _write_predictions(paths["temporary_dir"] / "test_predictions.csv", ids, labels, probabilities, threshold)
    _write_bootstrap(paths["temporary_dir"] / "rq1_bootstrap_metrics.csv", bootstrap)
    metrics = {
        "schema_version": 1, "contract_id": CONTRACT_ID, "status": "PASS", "run_id": RUN_ID, "model_key": "lr", "feature_version": "feature_set_v1", "split_version": "split_v1", "preprocessing_version": training["preprocessing_version"], "partition": "test", "rows": len(ids), "threshold": threshold,
        "prediction_file": "test_predictions.csv", "prediction_sha256": _sha256_file(paths["temporary_dir"] / "test_predictions.csv"), "bootstrap_indices_file": Path(bootstrap_indices_path).name, "bootstrap_indices_sha256": lineage["bootstrap_indices_sha256"], "bootstrap_replicates_file": "rq1_bootstrap_metrics.csv", "bootstrap_replicates_sha256": _sha256_file(paths["temporary_dir"] / "rq1_bootstrap_metrics.csv"), "counts": counts, "point_metrics": point, "confidence_intervals_95_percentile": intervals, "calibration_curve": _calibration_curve(labels, probabilities),
        "computational_metrics_seconds": {"full_train_fit": float(_read_json(paths["run_dir"] / "full_train_result.json")["fit_seconds"]), "test_inference_including_preprocessing": inference_seconds}, "quality_audit": gated.quality_audit.as_dict(), "lineage_sha256": lineage,
        "execution": {"model_fit_calls": 0, "preprocessing_fit_calls": 0, "predict_proba_calls": 1}, "data_boundary": {"source_partition": "test", "model_fit_performed": False, "preprocessing_fit_performed": False, "threshold_changed": False, "shared_bootstrap_regenerated": False}, "frozen_validation_threshold": {"selected_threshold": threshold, "artifact_sha256": _sha256_file(paths["threshold"]), "validation_manifest_sha256": _sha256_file(paths["validation_manifest"])}, "bootstrap_contract": {"seed": bootstrap_manifest.get("seed"), "replicates": bootstrap_manifest.get("replicates"), "sampling": bootstrap_manifest.get("sampling")},
    }
    _write_json(paths["temporary_dir"] / "rq1_test_metrics.json", metrics)
    manifest = {"schema_version": 1, "contract_id": CONTRACT_ID, "status": "PASS", "run_id": RUN_ID, "model_key": "lr", "feature_version": "feature_set_v1", "split_version": "split_v1", "rows": len(ids), "unique_record_ids": len(set(ids)), "source_order_preserved": True, "threshold": threshold, "lineage_sha256": lineage, "artifact_sha256": {name: _sha256_file(paths["temporary_dir"] / name) for name in sorted(PAYLOAD_FILES)}, "prohibited_outputs_written": {"historical_lr_baseline_v1": False, "legacy_12_feature_artifacts": False, "validation_artifacts": False, "cv_or_full_train_artifacts": False}}
    _write_json(paths["temporary_dir"] / "run_manifest.json", manifest)
    (paths["temporary_dir"] / "COMPLETED").write_text(f"{CONTRACT_ID} PASS\n", encoding="utf-8")
    _validate_artifact(paths["temporary_dir"], expected_ids=ids, expected_labels=labels, expected_lineage=lineage, expected_bootstrap_indices=indices)
    os.replace(paths["temporary_dir"], paths["output_dir"])
    return _validate_artifact(paths["output_dir"], expected_ids=ids, expected_labels=labels, expected_lineage=lineage, expected_bootstrap_indices=indices)


def validate_lr_test_evaluation(output_dir: str | Path = DEFAULT_OUTPUT_DIR, *, run_dir: str | Path = DEFAULT_RUN_DIR, split_manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST_PATH, split_dir: str | Path = DEFAULT_SPLIT_DIR, feature_contract_path: str | Path = DEFAULT_FEATURE_CONTRACT_PATH, model_protocol_path: str | Path = DEFAULT_MODEL_PROTOCOL_PATH, bootstrap_indices_path: str | Path = DEFAULT_BOOTSTRAP_INDICES_PATH, bootstrap_manifest_path: str | Path = DEFAULT_BOOTSTRAP_MANIFEST_PATH) -> dict[str, Any]:
    """Deep validate saved LR-09 artifacts against Test source without inference."""

    paths = _resolved_paths(Path(run_dir), Path(output_dir))
    _training, _selection, _threshold = _validate_frozen_lineage(paths)
    loader = DatasetBSplitLoader(manifest_path=split_manifest_path, split_dir=split_dir, feature_contract_path=feature_contract_path)
    gated = DatasetBInputQualityGate().apply(loader.load("test"))
    ids, labels = tuple(gated.record_ids.astype(str).tolist()), gated.y.to_numpy(dtype=np.int8)
    indices, _bootstrap_manifest = _load_bootstrap(Path(bootstrap_indices_path), Path(bootstrap_manifest_path), labels, ids)
    lineage = _lineage(paths, ids, split_manifest_path=Path(split_manifest_path), split_dir=Path(split_dir), feature_contract_path=Path(feature_contract_path), model_protocol_path=Path(model_protocol_path), bootstrap_indices_path=Path(bootstrap_indices_path), bootstrap_manifest_path=Path(bootstrap_manifest_path))
    return _validate_artifact(paths["output_dir"], expected_ids=ids, expected_labels=labels, expected_lineage=lineage, expected_bootstrap_indices=indices)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or validate LR-09 Test/RQ1 artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "validate"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
        sub.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        value = run_lr_test_evaluation(args.output_dir, run_dir=args.run_dir) if args.command == "run" else validate_lr_test_evaluation(args.output_dir, run_dir=args.run_dir)
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return 0
    except LRTestEvaluationError as exc:
        print(f"LR-09 failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
