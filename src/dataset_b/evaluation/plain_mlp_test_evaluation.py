"""Evaluate frozen PlainMLP state on Test with the shared bootstrap indices."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize, root
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, roc_auc_score

from src.dataset_b.evaluation.plain_mlp_validation_threshold import (
    DEFAULT_VALIDATION_DIR,
    validate_plain_mlp_validation_threshold,
)
from src.dataset_b.preprocessing import DatasetBSplitLoader
from src.dataset_b.runtime.plain_mlp_full_train import (
    DEFAULT_FEATURE_CONTRACT_PATH,
    DEFAULT_SPLIT_DIR,
    DEFAULT_SPLIT_MANIFEST_PATH,
    DEFAULT_OUTPUT_DIR as DEFAULT_FULL_TRAIN_DIR,
    _read_json as _read_full_train_json,
    _sha256_file,
    _sha256_ids,
    load_plain_mlp_bundle,
    validate_plain_mlp_full_train,
)
from src.utils.path import CONFIG_DIR, PROJECT_ROOT


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts/dataset_b/plain_mlp_v1/test_evaluation"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "dataset_b/plain_mlp_v1.yaml"
DEFAULT_BOOTSTRAP_INDICES = PROJECT_ROOT / "artifacts/dataset_b/rq1_test_bootstrap_indices.npz"
DEFAULT_BOOTSTRAP_MANIFEST = PROJECT_ROOT / "artifacts/dataset_b/rq1_test_bootstrap_manifest.json"
EXPECTED_TEST_ROWS = 43_996
BOOTSTRAP_REPLICATES = 1_000
CONTRACT_ID = "PMLP-05"
PREDICTION_COLUMNS = (
    "record_id", "y_approved", "p_approved", "p_denied",
    "predicted_approved", "predicted_denied",
)
METRIC_NAMES = (
    "roc_auc_approved", "average_precision_denied", "precision_denied",
    "recall_denied", "f1_denied", "recall_approved", "balanced_accuracy",
    "brier_approved", "calibration_intercept", "calibration_slope",
)
BOOTSTRAP_COLUMNS = ("replicate_id", *METRIC_NAMES, "invalid_metrics")
PAYLOAD_FILES = {
    "test_predictions.csv", "rq1_test_metrics.json", "rq1_bootstrap_metrics.csv"
}
FINAL_FILES = PAYLOAD_FILES | {"run_manifest.json", "COMPLETED"}


class PlainMLPTestEvaluationError(RuntimeError):
    """Raised when frozen PlainMLP Test evaluation violates its contract."""


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlainMLPTestEvaluationError(f"Cannot read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise PlainMLPTestEvaluationError(f"JSON artifact must contain an object: {path}")
    return value


def _write_json(path: str | Path, value: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _calibration(
    labels: np.ndarray, probabilities: np.ndarray
) -> tuple[float, float, dict[str, str]]:
    logits = logit(np.clip(probabilities, 1e-15, 1 - 1e-15))
    targets = labels.astype(np.float64)
    invalid: dict[str, str] = {}
    intercept = float("nan")
    slope = float("nan")
    try:
        intercept = float(
            brentq(lambda value: float(np.mean(expit(logits + value) - targets)), -80, 80)
        )
    except (ArithmeticError, ValueError, RuntimeError) as exc:
        invalid["calibration_intercept"] = str(exc)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = theta[0] + theta[1] * logits
        error = expit(eta) - targets
        loss = float(np.mean(np.logaddexp(0, eta) - targets * eta))
        gradient = np.asarray([np.mean(error), np.mean(error * logits)], dtype=np.float64)
        return loss, gradient

    try:
        fit = minimize(
            objective, [0.0, 1.0], method="L-BFGS-B", jac=True,
            options={"maxiter": 100, "ftol": 1e-12, "gtol": 1e-9},
        )
        theta = fit.x
        if not fit.success or not np.isfinite(theta).all():
            theta = root(
                lambda values: objective(values)[1],
                theta if np.isfinite(theta).all() else [0.0, 1.0],
                method="hybr",
                tol=1e-10,
            ).x
        if not np.isfinite(theta).all() or np.max(np.abs(objective(theta)[1])) > 1e-7:
            raise ArithmeticError("Calibration score equations did not converge.")
        slope = float(theta[1])
    except (ArithmeticError, ValueError, RuntimeError) as exc:
        invalid["calibration_slope"] = str(exc)
    return intercept, slope, invalid


def _metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> tuple[dict[str, float], dict[str, str], dict[str, int]]:
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predicted = probabilities >= threshold
    approved = labels == 1
    denied = ~approved
    tp = int(np.count_nonzero(approved & predicted))
    fn = int(np.count_nonzero(approved & ~predicted))
    tn = int(np.count_nonzero(denied & ~predicted))
    fp = int(np.count_nonzero(denied & predicted))
    n_approved, n_denied = tp + fn, tn + fp
    if n_approved == 0 or n_denied == 0:
        raise ArithmeticError("Both true classes are required.")
    values = {
        "roc_auc_approved": float(roc_auc_score(labels, probabilities)),
        "average_precision_denied": float(
            average_precision_score(1 - labels, 1 - probabilities)
        ),
        "precision_denied": tn / (tn + fn) if tn + fn else float("nan"),
        "recall_denied": tn / n_denied,
        "f1_denied": 2 * tn / (2 * tn + fn + fp),
        "recall_approved": tp / n_approved,
        "balanced_accuracy": (tp / n_approved + tn / n_denied) / 2,
        "brier_approved": float(np.mean((probabilities - labels) ** 2)),
    }
    values["calibration_intercept"], values["calibration_slope"], invalid = _calibration(
        labels, probabilities
    )
    if not np.isfinite(values["precision_denied"]):
        invalid["precision_denied"] = "no_predicted_denied"
    for name, value in values.items():
        if not np.isfinite(value):
            invalid.setdefault(name, "nonfinite_metric")
    counts = {
        "tp_approved": tp, "fn_approved": fn, "tn_denied": tn, "fp_denied": fp,
        "n_approved": n_approved, "n_denied": n_denied,
    }
    return values, invalid, counts


def _calibration_curve(labels: np.ndarray, probabilities: np.ndarray) -> list[dict[str, Any]]:
    edges = np.unique(np.quantile(probabilities, np.linspace(0, 1, 11)))
    if len(edges) < 2:
        raise ArithmeticError("Calibration curve needs at least two unique edges.")
    bin_ids = np.searchsorted(edges[1:-1], probabilities, side="right")
    rows = []
    for index in range(len(edges) - 1):
        mask = bin_ids == index
        if mask.any():
            rows.append(
                {
                    "bin": index,
                    "lower_edge": float(edges[index]),
                    "upper_edge": float(edges[index + 1]),
                    "count": int(mask.sum()),
                    "mean_p_approved": float(probabilities[mask].mean()),
                    "observed_approval_rate": float(labels[mask].mean()),
                }
            )
    if sum(row["count"] for row in rows) != len(labels):
        raise ArithmeticError("Calibration bins do not cover Test.")
    return rows


def _methods() -> dict[str, str]:
    return {
        "roc_auc_approved": "roc_auc_score(y_approved, p_approved)",
        "average_precision_denied": "average_precision_score(1-y_approved, 1-p_approved)",
        "predicted_approved": "p_approved >= locked Validation threshold",
        "brier_approved": "mean((p_approved-y_approved)^2)",
        "calibration_curve": "10 Test-probability quantile bins; duplicate edges collapsed",
        "calibration_intercept": "logistic offset recalibration with slope fixed at 1",
        "calibration_slope": "unpenalized logistic recalibration with free intercept and slope",
        "bootstrap": "1,000 shared stratified Test row-index replicates; no refit; percentile CI",
    }


def _paths(
    full_train_dir: str | Path,
    validation_dir: str | Path,
    split_manifest_path: str | Path,
    split_dir: str | Path,
    feature_contract_path: str | Path,
    config_path: str | Path,
    bootstrap_indices_path: str | Path,
    bootstrap_manifest_path: str | Path,
) -> dict[str, Path]:
    return {
        "full_train_dir": Path(full_train_dir),
        "validation_dir": Path(validation_dir),
        "split_manifest_path": Path(split_manifest_path),
        "split_dir": Path(split_dir),
        "feature_contract_path": Path(feature_contract_path),
        "config_path": Path(config_path),
        "bootstrap_indices_path": Path(bootstrap_indices_path),
        "bootstrap_manifest_path": Path(bootstrap_manifest_path),
    }


def _lineage(paths: dict[str, Path], ids: Sequence[str]) -> dict[str, str]:
    source_paths = {
        "full_train_manifest_sha256": paths["full_train_dir"] / "manifest.json",
        "fitted_bundle_sha256": paths["full_train_dir"] / "checkpoints/fitted_bundle.joblib",
        "validation_threshold_sha256": paths["validation_dir"] / "threshold_selection/threshold_selection.json",
        "split_manifest_sha256": paths["split_manifest_path"],
        "test_source_sha256": paths["split_dir"] / "test.csv",
        "feature_contract_sha256": paths["feature_contract_path"],
        "plain_mlp_config_sha256": paths["config_path"],
        "bootstrap_manifest_sha256": paths["bootstrap_manifest_path"],
        "bootstrap_indices_sha256": paths["bootstrap_indices_path"],
    }
    return {
        **{name: _sha256_file(path) for name, path in source_paths.items()},
        "test_record_ids_sha256": _sha256_ids(ids),
    }


def _validate_bootstrap(
    paths: dict[str, Path], labels: np.ndarray, ids: Sequence[str]
) -> np.ndarray:
    try:
        with np.load(paths["bootstrap_indices_path"], allow_pickle=False) as value:
            indices = value["indices"]
    except Exception as exc:
        raise PlainMLPTestEvaluationError("Cannot read shared bootstrap indices.") from exc
    manifest = _read_json(paths["bootstrap_manifest_path"])
    denied = int(np.count_nonzero(labels == 0))
    if (
        _sha256_file(paths["bootstrap_indices_path"]) != manifest.get("indices_sha256")
        or manifest.get("test_record_id_sha256") != _sha256_ids(ids)
        or indices.shape != (BOOTSTRAP_REPLICATES, EXPECTED_TEST_ROWS)
        or indices.dtype != np.int32
        or not np.all((indices >= 0) & (indices < EXPECTED_TEST_ROWS))
        or not np.all(labels[indices[:, :denied]] == 0)
        or not np.all(labels[indices[:, denied:]] == 1)
    ):
        raise PlainMLPTestEvaluationError("Shared bootstrap indices violate the locked contract.")
    return indices


def run_plain_mlp_test_evaluation(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    full_train_dir: str | Path = DEFAULT_FULL_TRAIN_DIR,
    validation_dir: str | Path = DEFAULT_VALIDATION_DIR,
    split_manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST_PATH,
    split_dir: str | Path = DEFAULT_SPLIT_DIR,
    feature_contract_path: str | Path = DEFAULT_FEATURE_CONTRACT_PATH,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    bootstrap_indices_path: str | Path = DEFAULT_BOOTSTRAP_INDICES,
    bootstrap_manifest_path: str | Path = DEFAULT_BOOTSTRAP_MANIFEST,
) -> dict[str, Any]:
    """Predict Test once, compute RQ1 metrics, and reuse the frozen bootstrap."""
    destination = Path(output_dir)
    paths = _paths(
        full_train_dir, validation_dir, split_manifest_path, split_dir,
        feature_contract_path, config_path, bootstrap_indices_path, bootstrap_manifest_path,
    )
    if destination.exists():
        result = validate_plain_mlp_test_evaluation(destination, **paths)
        return {**result, "resume_action": "SKIPPED_VALID_COMPLETED"}
    working = destination.with_name(destination.name + ".tmp")
    if working.exists():
        raise PlainMLPTestEvaluationError(
            f"Temporary directory already exists and was preserved: {working}"
        )
    try:
        full_manifest = validate_plain_mlp_full_train(paths["full_train_dir"])
        validate_plain_mlp_validation_threshold(paths["validation_dir"])
        selection_path = paths["validation_dir"] / "threshold_selection/threshold_selection.json"
        selection = _read_json(selection_path)
        threshold = selection.get("selected_threshold")
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not 0 < threshold < 1:
            raise PlainMLPTestEvaluationError("Locked Validation threshold is invalid.")
        loader = DatasetBSplitLoader(
            manifest_path=paths["split_manifest_path"],
            split_dir=paths["split_dir"],
            feature_contract_path=paths["feature_contract_path"],
        )
        partition = loader.load("test")
        ids = tuple(partition.record_ids.astype(str).tolist())
        labels = partition.y.to_numpy(dtype=np.int8)
        if (
            len(ids) != EXPECTED_TEST_ROWS
            or len(set(ids)) != EXPECTED_TEST_ROWS
            or labels.shape != (EXPECTED_TEST_ROWS,)
            or set(np.unique(labels).tolist()) != {0, 1}
        ):
            raise PlainMLPTestEvaluationError("Test IDs or labels violate split_v1.")
        indices = _validate_bootstrap(paths, labels, ids)
        preprocessing, estimator, _ = load_plain_mlp_bundle(
            paths["full_train_dir"] / "checkpoints/fitted_bundle.joblib"
        )
        started = time.perf_counter()
        transformed = preprocessing.transform(partition.X.reset_index(drop=True))
        probability_matrix = np.asarray(estimator.predict_proba(transformed), dtype=np.float64)
        inference_seconds = time.perf_counter() - started
        positive = np.flatnonzero(np.asarray(estimator.classes_) == 1)
        if probability_matrix.shape != (EXPECTED_TEST_ROWS, 2) or positive.shape != (1,):
            raise PlainMLPTestEvaluationError("Test probability matrix or class order is invalid.")
        probabilities = probability_matrix[:, int(positive[0])]
        if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
            raise PlainMLPTestEvaluationError("Test probabilities are invalid.")
        point, invalid, counts = _metrics(labels, probabilities, float(threshold))
        if invalid:
            raise PlainMLPTestEvaluationError(f"Point metrics contain invalid values: {invalid}")
        curve = _calibration_curve(labels, probabilities)
        bootstrap_rows = []
        for replicate, row_indices in enumerate(indices):
            values, problems, _ = _metrics(
                labels[row_indices], probabilities[row_indices], float(threshold)
            )
            bootstrap_rows.append(
                {
                    "replicate_id": replicate,
                    **values,
                    "invalid_metrics": json.dumps(problems, ensure_ascii=False, sort_keys=True),
                }
            )
        bootstrap = pd.DataFrame(bootstrap_rows, columns=BOOTSTRAP_COLUMNS)
        intervals = {}
        for name in METRIC_NAMES:
            values = bootstrap[name].to_numpy(dtype=np.float64)
            valid = values[np.isfinite(values)]
            intervals[name] = {
                "lower_2_5": float(np.quantile(valid, 0.025, method="linear")) if len(valid) else None,
                "upper_97_5": float(np.quantile(valid, 0.975, method="linear")) if len(valid) else None,
                "valid_replicates": int(len(valid)),
                "invalid_replicates": int(len(values) - len(valid)),
            }

        destination.parent.mkdir(parents=True, exist_ok=True)
        working.mkdir()
        predictions = pd.DataFrame(
            {
                "record_id": ids,
                "y_approved": labels,
                "p_approved": probabilities,
                "p_denied": 1 - probabilities,
                "predicted_approved": (probabilities >= threshold).astype(np.int8),
                "predicted_denied": (probabilities < threshold).astype(np.int8),
            }
        )
        prediction_path = working / "test_predictions.csv"
        prediction_path.write_text(
            predictions.to_csv(index=False, float_format="%.17g", lineterminator="\n"),
            encoding="utf-8", newline="",
        )
        bootstrap_path = working / "rq1_bootstrap_metrics.csv"
        bootstrap_path.write_text(
            bootstrap.to_csv(index=False, float_format="%.17g", lineterminator="\n"),
            encoding="utf-8", newline="",
        )
        lineage = _lineage(paths, ids)
        result = {
            "schema_version": 1,
            "contract_id": CONTRACT_ID,
            "status": "PASS",
            "evaluation_status": "complete_rq1_evaluation_experimental_feature_v0_2",
            "model_version": "plain_mlp_v1",
            "feature_version": "temporary_training_lock_v0_2",
            "split_version": "split_v1",
            "preprocessing_version": "dataset_b_neural_preprocessing_v1",
            "partition": "test",
            "rows": len(labels),
            "threshold": float(threshold),
            "prediction_sha256": _sha256_file(prediction_path),
            "bootstrap_replicates_sha256": _sha256_file(bootstrap_path),
            "counts": counts,
            "point_metrics": point,
            "confidence_intervals_95_percentile": intervals,
            "calibration_curve": curve,
            "computational_metrics_seconds": {
                "full_train_runner_elapsed": _read_full_train_json(
                    paths["full_train_dir"] / "full_train_result.json"
                )["execution"]["elapsed_seconds"],
                "test_inference_including_preprocessing": inference_seconds,
            },
            "methods": _methods(),
            "lineage_sha256": lineage,
            "execution": {
                "model_fit_calls": 0,
                "preprocessing_fit_calls": 0,
                "preprocessing_transform_calls": 1,
                "predict_proba_calls": 1,
            },
            "data_boundary": {
                "source_partition": "test",
                "model_fit_performed": False,
                "preprocessing_fit_performed": False,
                "threshold_changed": False,
                "shared_bootstrap_regenerated": False,
            },
            "full_train_source_status": full_manifest.get("status"),
        }
        _write_json(working / "rq1_test_metrics.json", result)
        manifest = {
            "schema_version": 1,
            "contract_id": CONTRACT_ID,
            "status": "PASS",
            "rows": len(labels),
            "threshold": float(threshold),
            "lineage_sha256": lineage,
            "artifact_sha256": {
                name: _sha256_file(working / name) for name in sorted(PAYLOAD_FILES)
            },
        }
        _write_json(working / "run_manifest.json", manifest)
        (working / "COMPLETED").write_text(f"{CONTRACT_ID} PASS\n", encoding="utf-8")
        validate_plain_mlp_test_evaluation(working, **paths)
        os.replace(working, destination)
        return validate_plain_mlp_test_evaluation(destination, **paths)
    except PlainMLPTestEvaluationError:
        raise
    except Exception as exc:
        raise PlainMLPTestEvaluationError(f"Cannot publish PlainMLP Test artifacts: {exc}") from exc


def validate_plain_mlp_test_evaluation(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    **kwargs: Any,
) -> dict[str, Any]:
    """Validate saved Test predictions, metrics, bootstrap rows, and lineage."""
    paths = _paths(
        kwargs.get("full_train_dir", DEFAULT_FULL_TRAIN_DIR),
        kwargs.get("validation_dir", DEFAULT_VALIDATION_DIR),
        kwargs.get("split_manifest_path", DEFAULT_SPLIT_MANIFEST_PATH),
        kwargs.get("split_dir", DEFAULT_SPLIT_DIR),
        kwargs.get("feature_contract_path", DEFAULT_FEATURE_CONTRACT_PATH),
        kwargs.get("config_path", DEFAULT_CONFIG_PATH),
        kwargs.get("bootstrap_indices_path", DEFAULT_BOOTSTRAP_INDICES),
        kwargs.get("bootstrap_manifest_path", DEFAULT_BOOTSTRAP_MANIFEST),
    )
    directory = Path(output_dir)
    validate_plain_mlp_full_train(paths["full_train_dir"])
    validate_plain_mlp_validation_threshold(paths["validation_dir"])
    actual = {path.name for path in directory.iterdir() if path.is_file()} if directory.is_dir() else set()
    if actual != FINAL_FILES:
        raise PlainMLPTestEvaluationError("Test artifact set is invalid.")
    if (directory / "COMPLETED").read_text(encoding="utf-8") != f"{CONTRACT_ID} PASS\n":
        raise PlainMLPTestEvaluationError("Test completion marker is invalid.")
    manifest = _read_json(directory / "run_manifest.json")
    result = _read_json(directory / "rq1_test_metrics.json")
    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, dict) or set(hashes) != PAYLOAD_FILES:
        raise PlainMLPTestEvaluationError("Test artifact hash coverage is invalid.")
    for name, expected in hashes.items():
        if _sha256_file(directory / name) != expected:
            raise PlainMLPTestEvaluationError(f"Test artifact hash mismatch: {name}")

    selection = _read_json(paths["validation_dir"] / "threshold_selection/threshold_selection.json")
    threshold = float(selection["selected_threshold"])
    frame = pd.read_csv(directory / "test_predictions.csv", dtype={"record_id": str})
    if (
        tuple(frame.columns) != PREDICTION_COLUMNS
        or len(frame) != EXPECTED_TEST_ROWS
        or not frame.record_id.is_unique
    ):
        raise PlainMLPTestEvaluationError("Test prediction structure is invalid.")
    labels = frame.y_approved.to_numpy(dtype=np.int8)
    probabilities = frame.p_approved.to_numpy(dtype=np.float64)
    if (
        set(np.unique(labels).tolist()) != {0, 1}
        or not np.isfinite(probabilities).all()
        or not np.allclose(frame.p_denied, 1 - probabilities, rtol=0.0, atol=1e-15)
        or not np.array_equal(frame.predicted_approved, (probabilities >= threshold).astype(np.int8))
        or not np.array_equal(frame.predicted_denied, (probabilities < threshold).astype(np.int8))
    ):
        raise PlainMLPTestEvaluationError("Test probabilities or threshold labels are invalid.")
    point, invalid, counts = _metrics(labels, probabilities, threshold)
    stored_point = result.get("point_metrics", {})
    if (
        invalid
        or counts != result.get("counts")
        or set(point) != set(stored_point)
        or any(not np.isclose(point[name], stored_point[name], rtol=0, atol=1e-12) for name in point)
    ):
        raise PlainMLPTestEvaluationError("Stored Test point metrics are not reproducible.")
    bootstrap = pd.read_csv(directory / "rq1_bootstrap_metrics.csv")
    if len(bootstrap) != BOOTSTRAP_REPLICATES or tuple(bootstrap.columns) != BOOTSTRAP_COLUMNS:
        raise PlainMLPTestEvaluationError("Bootstrap metric table is invalid.")
    for name in METRIC_NAMES:
        values = bootstrap[name].to_numpy(dtype=np.float64)
        valid = values[np.isfinite(values)]
        stored = result["confidence_intervals_95_percentile"][name]
        low = float(np.quantile(valid, 0.025, method="linear")) if len(valid) else None
        high = float(np.quantile(valid, 0.975, method="linear")) if len(valid) else None
        if (
            stored.get("valid_replicates") != len(valid)
            or stored.get("invalid_replicates") != len(values) - len(valid)
            or (low is None) != (stored.get("lower_2_5") is None)
            or (low is not None and not np.isclose(low, stored["lower_2_5"], rtol=0, atol=1e-14))
            or (high is not None and not np.isclose(high, stored["upper_97_5"], rtol=0, atol=1e-14))
        ):
            raise PlainMLPTestEvaluationError(f"Bootstrap interval is invalid: {name}")

    loader = DatasetBSplitLoader(
        manifest_path=paths["split_manifest_path"],
        split_dir=paths["split_dir"],
        feature_contract_path=paths["feature_contract_path"],
    )
    partition = loader.load("test")
    expected_ids = tuple(partition.record_ids.astype(str).tolist())
    expected_labels = partition.y.to_numpy(dtype=np.int8)
    _validate_bootstrap(paths, expected_labels, expected_ids)
    lineage = _lineage(paths, expected_ids)
    if (
        tuple(frame.record_id.astype(str)) != expected_ids
        or not np.array_equal(labels, expected_labels)
        or manifest.get("contract_id") != CONTRACT_ID
        or manifest.get("status") != "PASS"
        or manifest.get("rows") != EXPECTED_TEST_ROWS
        or manifest.get("threshold") != threshold
        or manifest.get("lineage_sha256") != lineage
        or result.get("contract_id") != CONTRACT_ID
        or result.get("status") != "PASS"
        or result.get("threshold") != threshold
        or result.get("lineage_sha256") != lineage
        or result.get("prediction_sha256") != _sha256_file(directory / "test_predictions.csv")
        or result.get("bootstrap_replicates_sha256")
        != _sha256_file(directory / "rq1_bootstrap_metrics.csv")
        or result.get("execution") != {
            "model_fit_calls": 0,
            "preprocessing_fit_calls": 0,
            "preprocessing_transform_calls": 1,
            "predict_proba_calls": 1,
        }
        or result.get("data_boundary") != {
            "source_partition": "test",
            "model_fit_performed": False,
            "preprocessing_fit_performed": False,
            "threshold_changed": False,
            "shared_bootstrap_regenerated": False,
        }
    ):
        raise PlainMLPTestEvaluationError("Test order, boundary, threshold, or lineage is invalid.")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or validate PlainMLP Test evaluation.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "validate"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        value = (
            run_plain_mlp_test_evaluation(args.output_dir)
            if args.command == "run"
            else validate_plain_mlp_test_evaluation(args.output_dir)
        )
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return 0
    except PlainMLPTestEvaluationError as exc:
        print(f"PlainMLP Test evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
