"""XGB-09: frozen-threshold Test inference, metrics, and locked bootstrap CIs."""

from __future__ import annotations

import csv
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from scipy.optimize import brentq, minimize, root
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, roc_auc_score

from src.dataset_b.evaluation.xgb_validation_threshold import validate_xgb_validation_threshold
from src.dataset_b.preprocessing import DatasetBInputQualityGate, DatasetBSplitLoader
from src.dataset_b.training.xgb_final_training import _validated_run_id, validate_xgb_final_training
from src.utils.path import CONFIG_DIR, DATASET_B_DIR, PROJECT_ROOT


CONTRACT_ID = "XGB-09"
EXPECTED_TEST_ROWS = 43_996
BOOTSTRAP_REPLICATES = 1_000
CALIBRATION_SLOPE_DECIMALS = 6
BOOTSTRAP_RECOMPUTATION_ATOL = 1e-14
POINT_METRIC_ATOL = 1e-12
DEFAULT_BOOTSTRAP_INDICES_PATH = PROJECT_ROOT / "artifacts/dataset_b/rq1_test_bootstrap_indices.npz"
DEFAULT_BOOTSTRAP_MANIFEST_PATH = PROJECT_ROOT / "artifacts/dataset_b/rq1_test_bootstrap_manifest.json"
DEFAULT_MODEL_PROTOCOL_PATH = CONFIG_DIR / "dataset_b/model_protocol_v1.yaml"
DEFAULT_SPLIT_MANIFEST_PATH = PROJECT_ROOT / "artifacts/dataset_b/split_v1.json"
DEFAULT_SPLIT_DIR = DATASET_B_DIR / "split_v1"
DEFAULT_FEATURE_CONTRACT_PATH = CONFIG_DIR / "dataset_b/feature_set_v1.yaml"
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


class XGBTestEvaluationError(RuntimeError):
    """Raised when XGB-09 violates its frozen Test boundary."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XGBTestEvaluationError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise XGBTestEvaluationError(f"Cannot read JSON artifact: {path}") from exc
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _paths(run_dir: Path, output_dir: Path, run_id: str) -> dict[str, Path]:
    _validated_run_id(run_id)
    root = run_dir.resolve(strict=True)
    _require(root.name == run_id, "XGB-09 run_dir must end with the explicit run_id.")
    destination = output_dir.resolve(strict=False)
    _require(destination == (root / "test").resolve(strict=False), "XGB-09 output must be run_dir/test.")
    paths = {
        "run_dir": root,
        "output_dir": destination,
        "temporary_dir": destination.with_name("test.tmp"),
        "pipeline": root / "checkpoints/fitted_pipeline.joblib",
        "training_manifest": root / "manifest.json",
        "validation_dir": root / "validation",
        "validation_manifest": root / "validation/manifest.json",
        "threshold": root / "validation/threshold_selection.json",
    }
    for name in ("pipeline", "training_manifest", "validation_manifest", "threshold"):
        _require(paths[name].is_file(), f"Missing frozen XGBoost source: {paths[name]}")
    return paths


def _locked_threshold(paths: Mapping[str, Path], run_id: str) -> tuple[dict[str, Any], float]:
    training = validate_xgb_final_training(paths["run_dir"], run_id=run_id)
    validation = validate_xgb_validation_threshold(paths["validation_dir"], run_dir=paths["run_dir"], run_id=run_id)
    selection = _read_json(paths["threshold"])
    _require(training.get("run_id") == validation.get("run_id") == run_id, "XGB-07/08 run_id lineage differs.")
    _require(training.get("model_key") == validation.get("model_key") == "xgb", "XGB-07/08 model lineage differs.")
    _require(training.get("reporting_key") == validation.get("reporting_key") == "xgboost", "XGB-07/08 reporting lineage differs.")
    _require(validation.get("selected_threshold") == selection.get("selected_threshold"), "XGB-08 threshold lineage differs.")
    _require(selection.get("fitted_pipeline") == str(paths["pipeline"].resolve()), "XGB-08 threshold belongs to another pipeline.")
    _require(selection.get("baseline_threshold_reused") is False, "XGB-08 reused a historical threshold.")
    threshold = float(selection["selected_threshold"])
    _require(0 < threshold < 1, "Frozen XGBoost threshold is invalid.")
    return training, threshold


def _load_pipeline(path: Path, expected_scale_pos_weight: float):
    try:
        pipeline = joblib.load(path)
    except Exception as exc:
        raise XGBTestEvaluationError("Cannot load frozen XGBoost pipeline.") from exc
    estimator = getattr(pipeline, "named_steps", {}).get("estimator")
    _require(estimator is not None, "Frozen XGBoost estimator is missing.")
    _require(np.array_equal(getattr(estimator, "classes_", None), np.asarray([0, 1])), "Frozen XGBoost class order differs.")
    _require(estimator.get_params().get("scale_pos_weight") == expected_scale_pos_weight, "Frozen XGBoost weighting differs from XGB-07.")
    return pipeline


def _predict_test_once(pipeline: Any, frame: Any) -> np.ndarray:
    probabilities = np.asarray(pipeline.predict_proba(frame), dtype=np.float64)
    _require(probabilities.shape == (len(frame), 2), "Test probability shape differs.")
    _require(
        np.isfinite(probabilities).all() and np.all((probabilities >= 0) & (probabilities <= 1)),
        "Test probabilities are invalid.",
    )
    _require(np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-7), "Test probability rows do not sum to one.")
    return probabilities[:, 1]


def _calibration(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float, dict[str, str]]:
    """Calculate calibration-in-the-large and calibration slope."""
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
        loss = float(np.mean(np.logaddexp(0, eta) - targets * eta))
        score = np.asarray([np.mean(error), np.mean(error * logits)], dtype=np.float64)
        return loss, score

    try:
        fit = minimize(
            objective,
            [0.0, 1.0],
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 100, "ftol": 1e-12, "gtol": 1e-9},
        )
        theta = fit.x
        if not fit.success or not np.isfinite(theta).all():
            initial = theta if np.isfinite(theta).all() else [0.0, 1.0]
            theta = root(lambda values: objective(values)[1], initial, method="hybr", tol=1e-10).x
        if not np.isfinite(theta).all() or np.max(np.abs(objective(theta)[1])) > 1e-7:
            raise ArithmeticError("Calibration score equations did not converge.")
        slope = float(theta[1])
    except (ArithmeticError, ValueError, RuntimeError) as exc:
        invalid["calibration_slope"] = str(exc)
    return intercept, slope, invalid


def _metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[dict[str, float], dict[str, str], dict[str, int]]:
    """Calculate the ten locked RQ1 metrics for one fixed population."""
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predicted = probabilities >= threshold
    approved = labels == 1
    denied = ~approved
    tp = int(np.count_nonzero(approved & predicted))
    fn = int(np.count_nonzero(approved & ~predicted))
    tn = int(np.count_nonzero(denied & ~predicted))
    fp = int(np.count_nonzero(denied & predicted))
    n_approved = tp + fn
    n_denied = tn + fp
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
    counts = {
        "tp_approved": tp,
        "fn_approved": fn,
        "tn_denied": tn,
        "fp_denied": fp,
        "n_approved": n_approved,
        "n_denied": n_denied,
    }
    return values, invalid, counts


def _calibration_curve(labels: np.ndarray, probabilities: np.ndarray) -> list[dict[str, Any]]:
    """Build the locked ten-quantile Test calibration summary."""
    edges = np.unique(np.quantile(probabilities, np.linspace(0, 1, 11)))
    if len(edges) < 2:
        raise ArithmeticError("Calibration curve needs at least two unique edges.")
    bin_ids = np.searchsorted(edges[1:-1], probabilities, side="right")
    rows: list[dict[str, Any]] = []
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
    _require(sum(row["count"] for row in rows) == len(labels), "Calibration bins do not cover Test.")
    return rows


def _bootstrap_rows(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    indices: np.ndarray,
) -> list[dict[str, Any]]:
    """Recompute metrics on the supplied locked replicate indices only."""
    rows: list[dict[str, Any]] = []
    for replicate_id, row_indices in enumerate(indices):
        values, invalid, _ = _metrics(labels[row_indices], probabilities[row_indices], threshold)
        rows.append(
            {
                "replicate_id": replicate_id,
                **values,
                "invalid_metrics": json.dumps(invalid, ensure_ascii=False, sort_keys=True),
            }
        )
    return rows


def _intervals(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    """Calculate locked 95% percentile intervals from replicate metrics."""
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
    """Apply the locked numerical-equivalence policy to one replicate value."""
    if not np.isfinite(stored) or not np.isfinite(recomputed):
        return bool(np.isfinite(stored) == np.isfinite(recomputed))
    if name == "calibration_slope":
        return bool(round(stored, CALIBRATION_SLOPE_DECIMALS) == round(recomputed, CALIBRATION_SLOPE_DECIMALS))
    return bool(np.isclose(stored, recomputed, rtol=0.0, atol=BOOTSTRAP_RECOMPUTATION_ATOL))


def _intervals_match(
    stored: Mapping[str, Mapping[str, Any]],
    recomputed: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Compare CI payloads with exact counts and the locked endpoint tolerance."""
    if set(stored) != set(METRIC_NAMES) or set(recomputed) != set(METRIC_NAMES):
        return False
    for name in METRIC_NAMES:
        left = stored[name]
        right = recomputed[name]
        if left.get("valid_replicates") != right.get("valid_replicates"):
            return False
        if left.get("invalid_replicates") != right.get("invalid_replicates"):
            return False
        for endpoint in ("lower_2_5", "upper_97_5"):
            left_value = left.get(endpoint)
            right_value = right.get(endpoint)
            if left_value is None or right_value is None:
                if left_value is not None or right_value is not None:
                    return False
            elif not np.isclose(
                float(left_value),
                float(right_value),
                rtol=0.0,
                atol=BOOTSTRAP_RECOMPUTATION_ATOL,
            ):
                return False
    return True


def _load_locked_bootstrap(
    indices_path: Path,
    manifest_path: Path,
    labels: np.ndarray,
    ids: Sequence[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        with np.load(indices_path, allow_pickle=False) as loaded:
            _require(set(loaded.files) == {"indices"}, "Locked bootstrap payload keys differ.")
            indices = loaded["indices"]
    except XGBTestEvaluationError:
        raise
    except Exception as exc:
        raise XGBTestEvaluationError("Cannot read locked bootstrap indices.") from exc
    manifest = _read_json(manifest_path)
    _require(len(ids) == EXPECTED_TEST_ROWS, "Locked bootstrap Test identifier count differs.")
    _require(manifest.get("dataset") == "dataset_b" and manifest.get("split_version") == "split_v1" and manifest.get("partition") == "test", "Locked bootstrap contract identity differs.")
    _require(manifest.get("test_rows") == EXPECTED_TEST_ROWS and manifest.get("indices_file") == indices_path.name, "Locked bootstrap source metadata differs.")
    _require(manifest.get("shape") == [BOOTSTRAP_REPLICATES, EXPECTED_TEST_ROWS] and manifest.get("dtype") == "int32", "Locked bootstrap manifest shape/dtype differs.")
    _require("xgboost" in manifest.get("reuse", []), "Locked bootstrap is not approved for XGBoost reuse.")
    _require(manifest.get("replicates") == BOOTSTRAP_REPLICATES, "Locked bootstrap replicate count differs.")
    _require(indices.shape == (BOOTSTRAP_REPLICATES, EXPECTED_TEST_ROWS) and indices.dtype == np.int32, "Locked bootstrap shape/dtype differs.")
    _require(np.all((indices >= 0) & (indices < EXPECTED_TEST_ROWS)), "Locked bootstrap indices are out of bounds.")
    n_denied = int(np.count_nonzero(labels == 0))
    _require(n_denied > 0 and n_denied < EXPECTED_TEST_ROWS, "Locked bootstrap labels require both classes.")
    _require(np.all(labels[indices[:, :n_denied]] == 0) and np.all(labels[indices[:, n_denied:]] == 1), "Locked bootstrap stratification differs.")
    return indices, manifest


def _write_predictions(path: Path, ids: Sequence[str], labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(PREDICTION_COLUMNS)
        for record_id, label, probability in zip(ids, labels, probabilities, strict=True):
            predicted = int(probability >= threshold)
            writer.writerow((record_id, int(label), repr(float(probability)), repr(float(1 - probability)), predicted, 1 - predicted))


def _read_predictions(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(tuple(reader.fieldnames or ()) == PREDICTION_COLUMNS, "XGB-09 prediction columns differ.")
        rows = list(reader)
    return (
        tuple(row["record_id"] for row in rows),
        np.asarray([int(row["y_approved"]) for row in rows], dtype=np.int8),
        np.asarray([float(row["p_approved"]) for row in rows], dtype=np.float64),
        np.asarray([float(row["p_denied"]) for row in rows], dtype=np.float64),
        np.asarray([int(row["predicted_approved"]) for row in rows], dtype=np.int8),
    )


def _write_bootstrap(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BOOTSTRAP_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_bootstrap(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(tuple(reader.fieldnames or ()) == BOOTSTRAP_COLUMNS, "XGB-09 bootstrap columns differ.")
        return [
            {
                "replicate_id": int(row["replicate_id"]),
                **{name: float(row[name]) for name in METRIC_NAMES},
                "invalid_metrics": row["invalid_metrics"],
            }
            for row in reader
        ]


def _lineage(
    paths: Mapping[str, Path],
    ids: Sequence[str],
    bootstrap_indices: Path,
    bootstrap_manifest: Path,
    model_protocol: Path,
    split_manifest: Path,
    split_dir: Path,
    feature_contract: Path,
) -> dict[str, Any]:
    return {
        "training_manifest": str(paths["training_manifest"].resolve()),
        "fitted_pipeline": str(paths["pipeline"].resolve()),
        "validation_manifest": str(paths["validation_manifest"].resolve()),
        "validation_threshold": str(paths["threshold"].resolve()),
        "bootstrap_indices": str(bootstrap_indices.resolve()),
        "bootstrap_manifest": str(bootstrap_manifest.resolve()),
        "model_protocol": str(model_protocol.resolve()),
        "split_manifest": str(split_manifest.resolve()),
        "feature_contract": str(feature_contract.resolve()),
        "test_source": str((split_dir / "test.csv").resolve()),
        "test_record_ids_verified": True,
        "test_record_count": len(ids),
    }


def _validate_artifact(directory: Path, *, run_id: str, expected_indices: np.ndarray | None = None) -> dict[str, Any]:
    required = {"test_predictions.csv", "rq1_test_metrics.json", "rq1_bootstrap_metrics.csv", "run_manifest.json", "COMPLETED"}
    _require(directory.is_dir() and {p.name for p in directory.iterdir() if p.is_file()} == required, "XGB-09 artifact set is incomplete or unexpected.")
    manifest = _read_json(directory / "run_manifest.json")
    metrics = _read_json(directory / "rq1_test_metrics.json")
    _require(manifest.get("contract_id") == metrics.get("contract_id") == CONTRACT_ID and manifest.get("status") == metrics.get("status") == "PASS", "XGB-09 is not PASS.")
    _require(manifest.get("run_id") == metrics.get("run_id") == run_id, "XGB-09 run_id lineage differs.")
    _require(manifest.get("model_key") == metrics.get("model_key") == "xgb", "XGB-09 model key differs.")
    _require(manifest.get("reporting_key") == metrics.get("reporting_key") == "xgboost", "XGB-09 reporting key differs.")
    _require(manifest.get("feature_version") == metrics.get("feature_version") == "feature_set_v1" and manifest.get("split_version") == metrics.get("split_version") == "split_v1", "XGB-09 feature/split lineage differs.")
    _require(manifest.get("lineage") == metrics.get("lineage"), "XGB-09 manifest/metric lineage differs.")
    ids, labels, approved, denied, predicted = _read_predictions(directory / "test_predictions.csv")
    threshold = float(metrics["threshold"])
    _require(np.isfinite(approved).all() and np.all((approved >= 0) & (approved <= 1)), "XGB-09 approval probabilities are invalid.")
    _require(np.allclose(denied, 1 - approved, rtol=0, atol=1e-15), "XGB-09 denied probabilities differ.")
    _require(np.array_equal(predicted, (approved >= threshold).astype(np.int8)), "XGB-09 used a changed threshold.")
    point, invalid, counts = _metrics(labels, approved, threshold)
    _require(not invalid and counts == metrics.get("counts"), "XGB-09 point metrics are not reproducible.")
    _require(all(np.isclose(point[name], metrics["point_metrics"][name], rtol=0, atol=POINT_METRIC_ATOL) for name in METRIC_NAMES), "XGB-09 point metric differs.")
    _require(metrics.get("calibration_curve") == _calibration_curve(labels, approved), "XGB-09 calibration curve is not reproducible.")
    stored = _read_bootstrap(directory / "rq1_bootstrap_metrics.csv")
    _require([row["replicate_id"] for row in stored] == list(range(BOOTSTRAP_REPLICATES)), "XGB-09 bootstrap replicate IDs differ.")
    _require(
        _intervals_match(metrics.get("confidence_intervals_95_percentile", {}), _intervals(stored)),
        "XGB-09 confidence intervals differ.",
    )
    if expected_indices is not None:
        recomputed = _bootstrap_rows(labels, approved, threshold, expected_indices)
        _require(len(stored) == len(recomputed), "XGB-09 bootstrap row count differs.")
        for left, right in zip(stored, recomputed, strict=True):
            _require(left["replicate_id"] == right["replicate_id"] and left["invalid_metrics"] == right["invalid_metrics"], "XGB-09 bootstrap payload differs.")
            for name in METRIC_NAMES:
                _require(_bootstrap_metric_matches(name, left[name], right[name]), f"XGB-09 bootstrap recomputation differs: {name}.")
    _require(metrics.get("execution") == {"model_fit_calls": 0, "preprocessing_fit_calls": 0, "predict_proba_calls": 1}, "XGB-09 execution boundary differs.")
    _require(metrics.get("bootstrap_contract", {}).get("regenerated") is False, "XGB-09 regenerated bootstrap evidence.")
    _require(
        metrics.get("numerical_comparison_policy")
        == {
            "calibration_slope_decimals": CALIBRATION_SLOPE_DECIMALS,
            "other_metric_absolute_tolerance": BOOTSTRAP_RECOMPUTATION_ATOL,
            "confidence_interval_method": "percentile_2.5_97.5_linear",
        },
        "XGB-09 numerical-comparison policy differs.",
    )
    _require(len(ids) == len(set(ids)) == EXPECTED_TEST_ROWS == manifest.get("rows") == manifest.get("unique_record_ids"), "XGB-09 Test population differs.")
    _require(np.array_equal(np.unique(labels), np.asarray([0, 1])), "XGB-09 labels differ from the binary contract.")
    return manifest


def run_xgb_test_evaluation(
    run_id: str,
    *,
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    split_manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST_PATH,
    split_dir: str | Path = DEFAULT_SPLIT_DIR,
    feature_contract_path: str | Path = DEFAULT_FEATURE_CONTRACT_PATH,
    model_protocol_path: str | Path = DEFAULT_MODEL_PROTOCOL_PATH,
    bootstrap_indices_path: str | Path = DEFAULT_BOOTSTRAP_INDICES_PATH,
    bootstrap_manifest_path: str | Path = DEFAULT_BOOTSTRAP_MANIFEST_PATH,
) -> dict[str, Any]:
    """Infer Test exactly once and reuse, never regenerate, locked indices."""
    root = Path(run_dir)
    destination = Path(output_dir) if output_dir is not None else root / "test"
    paths = _paths(root, destination, run_id)
    if paths["output_dir"].exists():
        return validate_xgb_test_evaluation(
            paths["output_dir"],
            run_dir=root,
            run_id=run_id,
            bootstrap_indices_path=bootstrap_indices_path,
            bootstrap_manifest_path=bootstrap_manifest_path,
            split_manifest_path=split_manifest_path,
            split_dir=split_dir,
            feature_contract_path=feature_contract_path,
            model_protocol_path=model_protocol_path,
        )
    training, threshold = _locked_threshold(paths, run_id)
    scale_pos_weight = float(training["full_train_effective_parameters"]["scale_pos_weight"])
    gated = DatasetBInputQualityGate().apply(
        DatasetBSplitLoader(
            manifest_path=split_manifest_path,
            split_dir=split_dir,
            feature_contract_path=feature_contract_path,
        ).load("test")
    )
    ids = tuple(gated.record_ids.astype(str).tolist())
    labels = gated.y.to_numpy(dtype=np.int8)
    _require(len(ids) == len(set(ids)) == EXPECTED_TEST_ROWS, "Test population differs.")
    indices_path = Path(bootstrap_indices_path)
    bootstrap_path = Path(bootstrap_manifest_path)
    indices, bootstrap_manifest = _load_locked_bootstrap(indices_path, bootstrap_path, labels, ids)
    pipeline = _load_pipeline(paths["pipeline"], scale_pos_weight)
    started = time.perf_counter()
    probabilities = _predict_test_once(pipeline, gated.X.reset_index(drop=True))
    inference_seconds = time.perf_counter() - started
    point, invalid, counts = _metrics(labels, probabilities, threshold)
    _require(not invalid, f"XGB-09 point metrics invalid: {invalid}")
    bootstrap = _bootstrap_rows(labels, probabilities, threshold, indices)
    lineage = _lineage(
        paths,
        ids,
        indices_path,
        bootstrap_path,
        Path(model_protocol_path),
        Path(split_manifest_path),
        Path(split_dir),
        Path(feature_contract_path),
    )
    tmp = paths["temporary_dir"]
    _require(not tmp.exists(), "XGB-09 temporary directory exists.")
    tmp.mkdir()
    try:
        _write_predictions(tmp / "test_predictions.csv", ids, labels, probabilities, threshold)
        _write_bootstrap(tmp / "rq1_bootstrap_metrics.csv", bootstrap)
        _write_json(
            tmp / "rq1_test_metrics.json",
            {
                "schema_version": 1,
                "contract_id": CONTRACT_ID,
                "status": "PASS",
                "run_id": run_id,
                "model_key": "xgb",
                "reporting_key": "xgboost",
                "feature_version": training["feature_version"],
                "split_version": training["split_version"],
                "rows": len(ids),
                "threshold": threshold,
                "scale_pos_weight": scale_pos_weight,
                "counts": counts,
                "point_metrics": point,
                "confidence_intervals_95_percentile": _intervals(bootstrap),
                "calibration_curve": _calibration_curve(labels, probabilities),
                "lineage": lineage,
                "quality_audit": gated.quality_audit.as_dict(),
                "bootstrap_contract": {
                    "seed": bootstrap_manifest.get("seed"),
                    "replicates": bootstrap_manifest.get("replicates"),
                    "sampling": bootstrap_manifest.get("sampling"),
                    "regenerated": False,
                },
                "numerical_comparison_policy": {
                    "calibration_slope_decimals": CALIBRATION_SLOPE_DECIMALS,
                    "other_metric_absolute_tolerance": BOOTSTRAP_RECOMPUTATION_ATOL,
                    "confidence_interval_method": "percentile_2.5_97.5_linear",
                },
                "execution": {"model_fit_calls": 0, "preprocessing_fit_calls": 0, "predict_proba_calls": 1},
                "data_boundary": {
                    "source_partition": "test",
                    "threshold_changed": False,
                    "shared_bootstrap_regenerated": False,
                },
                "inference_seconds": inference_seconds,
            },
        )
        _write_json(
            tmp / "run_manifest.json",
            {
                "schema_version": 1,
                "contract_id": CONTRACT_ID,
                "status": "PASS",
                "run_id": run_id,
                "model_key": "xgb",
                "reporting_key": "xgboost",
                "feature_version": training["feature_version"],
                "split_version": training["split_version"],
                "preprocessing_version": training["preprocessing_version"],
                "rows": len(ids),
                "unique_record_ids": len(set(ids)),
                "source_order_preserved": True,
                "threshold": threshold,
                "lineage": lineage,
                "prohibited_outputs_written": {
                    "cv_or_full_train_artifacts": False,
                    "validation_artifacts": False,
                    "legacy_xgb_artifacts": False,
                },
            },
        )
        (tmp / "COMPLETED").write_text(f"{CONTRACT_ID} PASS\n", encoding="utf-8")
        os.replace(tmp, paths["output_dir"])
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    result = _validate_artifact(paths["output_dir"], run_id=run_id, expected_indices=indices)
    _require(result.get("lineage") == lineage, "XGB-09 source lineage differs after persistence.")
    return result


def validate_xgb_test_evaluation(
    output_dir: str | Path,
    *,
    run_dir: str | Path,
    run_id: str,
    bootstrap_indices_path: str | Path = DEFAULT_BOOTSTRAP_INDICES_PATH,
    bootstrap_manifest_path: str | Path = DEFAULT_BOOTSTRAP_MANIFEST_PATH,
    split_manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST_PATH,
    split_dir: str | Path = DEFAULT_SPLIT_DIR,
    feature_contract_path: str | Path = DEFAULT_FEATURE_CONTRACT_PATH,
    model_protocol_path: str | Path = DEFAULT_MODEL_PROTOCOL_PATH,
) -> dict[str, Any]:
    """Independently recompute XGB-09 bootstrap metrics without inference."""
    paths = _paths(Path(run_dir), Path(output_dir), run_id)
    training, locked_threshold = _locked_threshold(paths, run_id)
    ids, labels, *_ = _read_predictions(paths["output_dir"] / "test_predictions.csv")
    gated = DatasetBInputQualityGate().apply(
        DatasetBSplitLoader(
            manifest_path=split_manifest_path,
            split_dir=split_dir,
            feature_contract_path=feature_contract_path,
        ).load("test")
    )
    expected_ids = tuple(gated.record_ids.astype(str).tolist())
    expected_labels = gated.y.to_numpy(dtype=np.int8)
    _require(ids == expected_ids and np.array_equal(labels, expected_labels), "XGB-09 predictions do not align to the Test source.")
    indices, _ = _load_locked_bootstrap(Path(bootstrap_indices_path), Path(bootstrap_manifest_path), labels, ids)
    manifest = _validate_artifact(paths["output_dir"], run_id=run_id, expected_indices=indices)
    expected_lineage = _lineage(
        paths,
        ids,
        Path(bootstrap_indices_path),
        Path(bootstrap_manifest_path),
        Path(model_protocol_path),
        Path(split_manifest_path),
        Path(split_dir),
        Path(feature_contract_path),
    )
    _require(manifest.get("lineage") == expected_lineage, "XGB-09 source lineage differs.")
    metrics = _read_json(paths["output_dir"] / "rq1_test_metrics.json")
    _require(float(metrics.get("threshold")) == locked_threshold == float(manifest.get("threshold")), "XGB-09 frozen threshold differs.")
    _require(metrics.get("lineage") == expected_lineage, "XGB-09 metric lineage differs.")
    _require(metrics.get("quality_audit") == gated.quality_audit.as_dict(), "XGB-09 quality audit differs.")
    _require(metrics.get("scale_pos_weight") == training["full_train_effective_parameters"]["scale_pos_weight"], "XGB-09 weighting lineage differs.")
    return manifest


def parse_args(argv: Sequence[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(description="Run or validate XGB-09.")
    parser.add_argument("command", choices=("run", "validate"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-indices", type=Path, default=DEFAULT_BOOTSTRAP_INDICES_PATH)
    parser.add_argument("--bootstrap-manifest", type=Path, default=DEFAULT_BOOTSTRAP_MANIFEST_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output_dir or args.run_dir / "test"
    value = (
        run_xgb_test_evaluation(
            args.run_id,
            run_dir=args.run_dir,
            output_dir=output,
            bootstrap_indices_path=args.bootstrap_indices,
            bootstrap_manifest_path=args.bootstrap_manifest,
        )
        if args.command == "run"
        else validate_xgb_test_evaluation(
            output,
            run_dir=args.run_dir,
            run_id=args.run_id,
            bootstrap_indices_path=args.bootstrap_indices,
            bootstrap_manifest_path=args.bootstrap_manifest,
        )
    )
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
