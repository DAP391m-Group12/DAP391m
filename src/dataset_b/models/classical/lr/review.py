"""LR-10: direct, read-only review of existing LR-07/08/09 evidence.

This closure deliberately validates content and numerical reproduction without
using file digests as acceptance gates.  It never fits or invokes a model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score

from src.dataset_b.models.classical.lr.test_evaluation import (
    BOOTSTRAP_RECOMPUTATION_ATOL,
    CALIBRATION_SLOPE_DECIMALS,
    METRIC_NAMES,
    _bootstrap_rows,
    _calibration_curve,
    _intervals,
    _metrics,
    _read_bootstrap,
    _read_predictions as _read_test_predictions,
)
from src.dataset_b.research.rq1.reporting import validate_model_summary
from src.utils.path import PROJECT_ROOT


CONTRACT_ID = "LR-10"
RUN_ID = "lr_final_15f_20261002_01"
EXPECTED_TRAIN_ROWS = 205_310
EXPECTED_VALIDATION_ROWS = 43_995
EXPECTED_TEST_ROWS = 43_996
EXPECTED_BOOTSTRAP_REPLICATES = 1_000
DEFAULT_LR_ROOT = PROJECT_ROOT / "artifacts/dataset_b/lr/working"
DEFAULT_LR_FINAL_ROOT = PROJECT_ROOT / "artifacts/dataset_b/lr/final"
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "data/dataset_b/split_v1"
DEFAULT_BOOTSTRAP_INDICES_PATH = PROJECT_ROOT / "artifacts/dataset_b/rq1_test_bootstrap_indices.npz"


class LRFinalReviewError(RuntimeError):
    """Raised when existing LR artifacts fail direct final review."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LRFinalReviewError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LRFinalReviewError(f"Cannot read review source: {path}") from exc
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _split_identity(path: Path) -> tuple[tuple[str, ...], np.ndarray]:
    frame = pd.read_csv(
        path,
        usecols=["record_id", "y_approved"],
        dtype={"record_id": str, "y_approved": np.int8},
        keep_default_na=False,
    )
    return tuple(frame["record_id"].tolist()), frame["y_approved"].to_numpy(dtype=np.int8)


def _close(left: float, right: float, *, atol: float = 1e-14) -> bool:
    return bool(np.isclose(float(left), float(right), rtol=0.0, atol=atol))


def _validate_inventory(root: Path, *, final: bool) -> None:
    required_files = {
        "COMPLETED", "cv_oof_predictions.csv", "cv_results.json",
        "full_train_result.json", "manifest.json", "warnings.json",
    }
    allowed_files = required_files | ({"FROZEN"} if final else set())
    observed_files = {path.name for path in root.iterdir() if path.is_file()}
    _require(required_files.issubset(observed_files) and observed_files.issubset(allowed_files), "LR root artifact inventory differs.")
    required_dirs = {"checkpoints", "validation", "test"}
    allowed_dirs = required_dirs | ({"reporting", "shared_bootstrap"} if final else set())
    observed_dirs = {path.name for path in root.iterdir() if path.is_dir()}
    if final:
        _require(required_dirs | {"shared_bootstrap"} <= observed_dirs <= allowed_dirs, "LR root directory inventory differs.")
    else:
        _require(observed_dirs == required_dirs, "LR root directory inventory differs.")
    _require(
        {path.name for path in (root / "checkpoints").iterdir() if path.is_file()}
        == {"fitted_pipeline.joblib", "fitted_preprocessor.joblib", "lr_model.joblib"},
        "LR checkpoint inventory differs.",
    )


def _validate_training(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = _read_json(root / "manifest.json")
    cv = _read_json(root / "cv_results.json")
    full = _read_json(root / "full_train_result.json")
    _require(manifest.get("status") == "PASS" and manifest.get("contract_id") == "LR-07", "LR-07 is not PASS.")
    _require(manifest.get("run_id") == cv.get("run_id") == full.get("run_id") == RUN_ID, "LR-07 run lineage differs.")
    _require(manifest.get("model_key") == "lr", "LR-07 model key differs.")
    _require(manifest.get("feature_version") == "feature_set_v1", "LR-07 feature version differs.")
    _require(manifest.get("split_version") == "split_v1", "LR-07 split version differs.")
    _require(manifest.get("preprocessing_version") == "dataset_b_preprocessing_v2", "LR preprocessing version differs.")
    _require(manifest.get("source_partition") == "train", "LR-07 is not Train-only.")
    _require(manifest.get("validation_accessed") is False and manifest.get("test_accessed") is False, "LR-07 crossed a holdout boundary.")
    _require(manifest.get("hyperparameter_search") is False, "LR-07 performed hyperparameter search.")
    schemas = manifest.get("schemas", {})
    _require((schemas.get("raw_feature_count"), schemas.get("semantic_feature_count"), schemas.get("encoded_feature_count")) == (15, 19, 62), "LR-07 feature counts differ.")
    expected_params = {"penalty": "l2", "C": 1.0, "solver": "lbfgs", "max_iter": 1000, "fit_intercept": True, "class_weight": "balanced"}
    _require(manifest.get("model_parameters") == expected_params, "LR-07 model parameters differ.")
    _require(cv.get("fold_count") == 5 and len(cv.get("folds", [])) == 5, "LR CV does not contain five folds.")
    _require(cv.get("fresh_estimator_per_fold") is True and cv.get("hyperparameter_search") is False, "LR CV estimator/search evidence differs.")
    _require((cv.get("oof_rows"), cv.get("oof_unique_record_ids"), cv.get("oof_coverage_min"), cv.get("oof_coverage_max")) == (EXPECTED_TRAIN_ROWS, EXPECTED_TRAIN_ROWS, 1, 1), "LR OOF coverage differs.")
    _require(all(fold.get("train_rows") == 164_248 and fold.get("fold_validation_rows") == 41_062 for fold in cv["folds"]), "LR fold sizes differ.")
    _require(full.get("status") == "PASS" and full.get("retained_train_rows") == EXPECTED_TRAIN_ROWS, "LR full-Train evidence differs.")
    _require((full.get("raw_feature_count"), full.get("semantic_feature_count"), full.get("encoded_feature_count")) == (15, 19, 62), "LR full-Train feature counts differ.")
    _require(full.get("matrix_format") == "csr" and full.get("matrix_dtype") == "float64" and full.get("matrix_finite") is True, "LR full-Train matrix evidence differs.")
    _require(full.get("fresh_preprocessor") is True and full.get("fresh_estimator") is True, "LR full-Train objects were not fresh.")
    _require(full.get("reload_predictions_exact") is True and full.get("reload_prediction_rows_checked") == 4096, "LR reload reproduction evidence differs.")

    train_ids, train_labels = _split_identity(DEFAULT_SPLIT_DIR / "train.csv")
    oof = pd.read_csv(root / "cv_oof_predictions.csv", dtype={"record_id": str}, keep_default_na=False)
    _require(tuple(oof.columns) == ("record_id", "y_approved", "p_approved", "fold"), "LR OOF columns differ.")
    _require(len(oof) == len(set(oof["record_id"])) == EXPECTED_TRAIN_ROWS, "LR OOF IDs are incomplete or duplicated.")
    _require(tuple(oof["record_id"].tolist()) == train_ids, "LR OOF record order differs from Train.")
    _require(np.array_equal(oof["y_approved"].to_numpy(dtype=np.int8), train_labels), "LR OOF labels differ from Train.")
    _require(sorted(oof["fold"].astype(int).unique().tolist()) == [1, 2, 3, 4, 5], "LR OOF folds differ.")
    _require(_close(roc_auc_score(train_labels, oof["p_approved"].to_numpy(dtype=np.float64)), cv["oof_roc_auc"]), "LR OOF ROC-AUC does not reproduce.")

    pipeline = joblib.load(root / "checkpoints/fitted_pipeline.joblib")
    preprocessor = joblib.load(root / "checkpoints/fitted_preprocessor.joblib")
    estimator = joblib.load(root / "checkpoints/lr_model.joblib")
    _require(tuple(getattr(pipeline, "named_steps", {})) == ("preprocess", "estimator"), "LR pipeline steps differ.")
    fitted_estimator = pipeline.named_steps["estimator"]
    _require(fitted_estimator.get_params(deep=False).get("class_weight") == "balanced", "LR fitted estimator is not balanced.")
    _require(np.array_equal(getattr(fitted_estimator, "classes_", None), np.asarray([0, 1])), "LR fitted class order differs.")
    _require(hasattr(preprocessor, "transform") and hasattr(estimator, "coef_"), "LR checkpoint fitted state is missing.")
    _require(estimator.get_params(deep=False).get("class_weight") == "balanced", "LR standalone estimator is not balanced.")
    return manifest, cv, full


def _validate_validation(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    directory = root / "validation"
    expected = {"COMPLETED", "manifest.json", "prediction_manifest.json", "threshold_candidates.csv", "threshold_selection.json", "validation_predictions.csv"}
    _require({path.name for path in directory.iterdir() if path.is_file()} == expected, "LR-08 artifact inventory differs.")
    manifest = _read_json(directory / "manifest.json")
    selection = _read_json(directory / "threshold_selection.json")
    _require(manifest.get("status") == selection.get("status") == "PASS", "LR-08 is not PASS.")
    _require(manifest.get("contract_id") == selection.get("contract_id") == "LR-08", "LR-08 contract differs.")
    _require(manifest.get("run_id") == selection.get("run_id") == RUN_ID, "LR-08 run lineage differs.")
    validation = pd.read_csv(directory / "validation_predictions.csv", dtype={"record_id": str}, keep_default_na=False)
    _require(tuple(validation.columns) == ("record_id", "y_approved", "p_approved"), "LR Validation prediction columns differ.")
    source_ids, source_labels = _split_identity(DEFAULT_SPLIT_DIR / "validation.csv")
    _require(len(validation) == len(set(validation["record_id"])) == EXPECTED_VALIDATION_ROWS, "LR Validation IDs are incomplete or duplicated.")
    _require(tuple(validation["record_id"].tolist()) == source_ids, "LR Validation record order differs.")
    labels = validation["y_approved"].to_numpy(dtype=np.int8)
    probabilities = validation["p_approved"].to_numpy(dtype=np.float64)
    _require(np.array_equal(labels, source_labels), "LR Validation labels differ.")
    rows = []
    for threshold in (value / 1000 for value in range(1, 1000)):
        predicted = probabilities >= threshold
        rows.append({
            "threshold": threshold,
            "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
            "recall_approved": float(recall_score(labels, predicted, pos_label=1, zero_division=0)),
            "recall_denied": float(recall_score(labels, predicted, pos_label=0, zero_division=0)),
            "tp_approved": int(np.count_nonzero((labels == 1) & predicted)),
            "fn_approved": int(np.count_nonzero((labels == 1) & ~predicted)),
            "tn_denied": int(np.count_nonzero((labels == 0) & ~predicted)),
            "fp_denied": int(np.count_nonzero((labels == 0) & predicted)),
        })
    candidates = pd.read_csv(directory / "threshold_candidates.csv")
    expected_candidates = pd.DataFrame(rows)
    _require(tuple(candidates.columns) == tuple(expected_candidates.columns), "LR threshold candidate columns differ.")
    _require(np.allclose(candidates.iloc[:, :4], expected_candidates.iloc[:, :4], rtol=0.0, atol=1e-15) and np.array_equal(candidates.iloc[:, 4:].to_numpy(), expected_candidates.iloc[:, 4:].to_numpy()), "LR threshold candidates do not reproduce.")
    best = max(row["balanced_accuracy"] for row in rows)
    maximizing = [row for row in rows if row["balanced_accuracy"] == best]
    chosen = min(maximizing, key=lambda row: (abs(row["threshold"] - 0.5), row["threshold"]))
    _require(_close(selection.get("selected_threshold"), chosen["threshold"], atol=0.0), "LR selected threshold does not reproduce.")
    _require(selection.get("maximizer_count") == len(maximizing), "LR threshold tie count differs.")
    for name in ("balanced_accuracy", "recall_approved", "recall_denied"):
        _require(_close(selection["selected_metrics"][name], chosen[name]), f"LR selected {name} does not reproduce.")
    return manifest, selection


def _validate_test(root: Path, bootstrap_indices_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    directory = root / "test"
    expected = {"COMPLETED", "run_manifest.json", "rq1_bootstrap_metrics.csv", "rq1_test_metrics.json", "test_predictions.csv"}
    _require({path.name for path in directory.iterdir() if path.is_file()} == expected, "LR-09 artifact inventory differs.")
    manifest = _read_json(directory / "run_manifest.json")
    metrics = _read_json(directory / "rq1_test_metrics.json")
    _require(manifest.get("status") == metrics.get("status") == "PASS", "LR-09 is not PASS.")
    _require(manifest.get("contract_id") == metrics.get("contract_id") == "LR-09", "LR-09 contract differs.")
    _require(manifest.get("run_id") == metrics.get("run_id") == RUN_ID, "LR-09 run lineage differs.")
    _require(metrics.get("threshold") == 0.493, "LR Test threshold differs from frozen Validation threshold.")
    _require(metrics.get("data_boundary", {}).get("shared_bootstrap_regenerated") is False, "LR bootstrap indices were regenerated.")
    _require(metrics.get("data_boundary", {}).get("threshold_changed") is False, "LR Test threshold was changed.")
    _require(metrics.get("execution") == {"model_fit_calls": 0, "preprocessing_fit_calls": 0, "predict_proba_calls": 1}, "LR Test execution evidence differs.")
    ids, labels, probabilities, denied, predicted = _read_test_predictions(directory / "test_predictions.csv")
    source_ids, source_labels = _split_identity(DEFAULT_SPLIT_DIR / "test.csv")
    _require(len(ids) == len(set(ids)) == EXPECTED_TEST_ROWS and ids == source_ids, "LR Test IDs/order differ.")
    _require(np.array_equal(labels, source_labels), "LR Test labels differ.")
    _require(np.allclose(denied, 1.0 - probabilities, rtol=0.0, atol=1e-15), "LR denied probabilities differ.")
    _require(np.array_equal(predicted, (probabilities >= 0.493).astype(np.int8)), "LR stored Test classes differ.")
    point, invalid, counts = _metrics(labels, probabilities, 0.493)
    _require(not invalid, "LR Test point metrics contain invalid values.")
    for name in METRIC_NAMES:
        _require(_close(point[name], metrics["point_metrics"][name]), f"LR point metric does not reproduce: {name}")
    _require(counts == metrics["counts"], "LR Test confusion counts do not reproduce.")
    curve = _calibration_curve(labels, probabilities)
    _require(curve == metrics["calibration_curve"], "LR calibration curve does not reproduce.")
    try:
        with np.load(bootstrap_indices_path, allow_pickle=False) as loaded:
            indices = np.asarray(loaded["indices"])
    except (OSError, ValueError, KeyError) as exc:
        raise LRFinalReviewError("Cannot read locked LR bootstrap indices.") from exc
    _require(indices.shape == (EXPECTED_BOOTSTRAP_REPLICATES, EXPECTED_TEST_ROWS) and indices.dtype == np.int32, "LR bootstrap shape/dtype differs.")
    _require(np.all((indices >= 0) & (indices < EXPECTED_TEST_ROWS)), "LR bootstrap indices are out of bounds.")
    n_denied = int(np.count_nonzero(labels == 0))
    _require(np.all(labels[indices[:, :n_denied]] == 0) and np.all(labels[indices[:, n_denied:]] == 1), "LR bootstrap stratification differs.")
    stored_rows = _read_bootstrap(directory / "rq1_bootstrap_metrics.csv")
    recomputed_rows = _bootstrap_rows(labels, probabilities, 0.493, indices)
    _require(len(stored_rows) == len(recomputed_rows) == EXPECTED_BOOTSTRAP_REPLICATES, "LR bootstrap replicate count differs.")
    for stored, recomputed in zip(stored_rows, recomputed_rows, strict=True):
        _require(stored["replicate_id"] == recomputed["replicate_id"] and stored["invalid_metrics"] == recomputed["invalid_metrics"], "LR bootstrap replicate identity/payload differs.")
        for name in METRIC_NAMES:
            if name == "calibration_slope":
                matches = round(stored[name], CALIBRATION_SLOPE_DECIMALS) == round(recomputed[name], CALIBRATION_SLOPE_DECIMALS)
            else:
                matches = _close(stored[name], recomputed[name], atol=BOOTSTRAP_RECOMPUTATION_ATOL)
            _require(matches, f"LR bootstrap metric does not reproduce: replicate={stored['replicate_id']} metric={name}")
    intervals = _intervals(recomputed_rows)
    for name in METRIC_NAMES:
        _require(intervals[name]["valid_replicates"] == metrics["confidence_intervals_95_percentile"][name]["valid_replicates"], f"LR interval valid count differs: {name}")
        _require(intervals[name]["invalid_replicates"] == metrics["confidence_intervals_95_percentile"][name]["invalid_replicates"], f"LR interval invalid count differs: {name}")
        for bound in ("lower_2_5", "upper_97_5"):
            _require(_close(intervals[name][bound], metrics["confidence_intervals_95_percentile"][name][bound]), f"LR interval does not reproduce: {name}/{bound}")
    return manifest, metrics


def review_lr_final(
    run_id: str,
    *,
    run_dir: str | Path,
    bootstrap_indices_path: str | Path = DEFAULT_BOOTSTRAP_INDICES_PATH,
) -> dict[str, Any]:
    """Review stored LR evidence without fitting, inference, or artifact writes."""

    _require(run_id == RUN_ID, "LR-10 requires the frozen explicit run ID.")
    root = Path(run_dir).resolve(strict=True)
    _require(root.name == run_id, "LR-10 run_dir does not match the explicit run_id.")
    allowed_roots = {DEFAULT_LR_ROOT.resolve(strict=False), DEFAULT_LR_FINAL_ROOT.resolve(strict=False)}
    _require(root.parent in allowed_roots, "LR-10 refuses artifacts outside the LR working/final roots.")
    final = root.parent == DEFAULT_LR_FINAL_ROOT.resolve(strict=False)
    if final and Path(bootstrap_indices_path) == DEFAULT_BOOTSTRAP_INDICES_PATH:
        local = root / "shared_bootstrap/rq1_test_bootstrap_indices.npz"
        if local.is_file():
            bootstrap_indices_path = local
    _validate_inventory(root, final=final)
    training, cv, full = _validate_training(root)
    validation, selection = _validate_validation(root)
    test, metrics = _validate_test(root, Path(bootstrap_indices_path).resolve(strict=True))
    _require(training["run_id"] == validation["run_id"] == test["run_id"] == run_id, "LR stage run lineage differs.")
    _require(float(selection["selected_threshold"]) == float(metrics["threshold"]), "LR Validation/Test threshold lineage differs.")
    point_metrics = dict(metrics["point_metrics"])
    intervals = dict(metrics["confidence_intervals_95_percentile"])
    point_metrics["brier_score_approved"] = point_metrics.pop("brier_approved")
    intervals["brier_score_approved"] = intervals.pop("brier_approved")
    bootstrap_source = str(Path(bootstrap_indices_path).resolve())
    summary = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "status": "PASS",
        "reporting_status": "COMPLETE",
        "run_id": run_id,
        "model_key": "lr",
        "feature_version": "feature_set_v1",
        "split_version": "split_v1",
        "preprocessing_version": training["preprocessing_version"],
        "stages": {"lr_07": "PASS", "lr_08": "PASS", "lr_09": "PASS", "lr_10": "PASS"},
        "data_boundaries": {"train_rows": EXPECTED_TRAIN_ROWS, "validation_rows": EXPECTED_VALIDATION_ROWS, "test_rows": EXPECTED_TEST_ROWS, "validation_accessed": True, "test_accessed": True},
        "feature_counts": {"raw": 15, "semantic": 19, "model_input": 62},
        "cv": {"fold_count": 5, "mean_roc_auc": cv["mean_fold_roc_auc"], "std_roc_auc": cv["std_fold_roc_auc"], "oof_roc_auc": cv["oof_roc_auc"], "oof_rows": cv["oof_rows"], "convergence_warning_count": sum(int(fold.get("convergence_warning_count", 0)) for fold in cv["folds"])},
        "full_train": {"fit_rows": full["retained_train_rows"], "fit_seconds": full["fit_seconds"], "n_iter": full["n_iter"], "convergence_warning_count": full["convergence_warning_count"]},
        "validation": {"rows": validation["rows"], "unique_record_ids": validation["unique_record_ids"], "source_order_preserved": True, "threshold": selection["selected_threshold"], **selection["selected_metrics"], "maximizer_count": selection["maximizer_count"], "probability_min": _read_json(root / "validation/prediction_manifest.json")["probability_min"], "probability_max": _read_json(root / "validation/prediction_manifest.json")["probability_max"], "ordered_record_ids_verified": True},
        "test": {"rows": test["rows"], "unique_record_ids": test["unique_record_ids"], "source_order_preserved": True, "threshold": metrics["threshold"], "point_metrics": point_metrics, "confidence_intervals_95_percentile": intervals, "counts": metrics["counts"], "calibration_curve": metrics["calibration_curve"], "ordered_record_ids_verified": True, "bootstrap_indices": bootstrap_source},
        "selected_threshold": metrics["threshold"],
        "point_metrics": point_metrics,
        "confidence_intervals_95_percentile": intervals,
        "artifact_sources": {"training_manifest": str(root / "manifest.json"), "cv_results": str(root / "cv_results.json"), "full_train_result": str(root / "full_train_result.json"), "validation_manifest": str(root / "validation/manifest.json"), "test_manifest": str(root / "test/run_manifest.json"), "test_predictions": str(root / "test/test_predictions.csv")},
        "execution": {"fit_calls": 0, "predict_proba_calls": 0, "bootstrap_regenerated": False, "threshold_changed": False},
        "review_policy": "direct_content_and_numerical_reproduction",
    }
    validate_model_summary(summary, require_complete=True)
    return summary


def parse_args(argv: Sequence[str] | None = None):
    import argparse
    parser = argparse.ArgumentParser(description="Read-only LR-10 final review.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(review_lr_final(args.run_id, run_dir=args.run_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
