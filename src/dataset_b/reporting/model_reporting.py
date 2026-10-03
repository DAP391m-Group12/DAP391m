"""Read-only reporting contract shared by final Dataset B model notebooks.

This module deliberately does not train, transform, predict, or select a
threshold. It validates persisted evidence, converts it into a stable
presentation schema, and exposes an explicit writer for the derived XGBoost
reporting package so notebooks and the six-model review do not need
model-specific parsing branches.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import pandas as pd
import numpy as np


FINAL_MODEL_KEYS = ("lr", "rf", "xgboost", "plain_mlp", "realmlp", "tabmmini")
FINAL_FEATURE_VERSION = "feature_set_v1"
FINAL_SPLIT_VERSION = "split_v1"
EXPECTED_TRAIN_ROWS = 205_310
EXPECTED_VALIDATION_ROWS = 43_995
EXPECTED_TEST_ROWS = 43_996
EXPECTED_BOOTSTRAP_REPLICATES = 1_000
REPORTING_SCHEMA_VERSION = 1

EXPECTED_REPORTING_FILES = (
    "model_summary.json",
    "cv_fold_metrics.csv",
    "rq1_metrics_long.csv",
    "confusion_matrix.csv",
    "calibration_curve.csv",
    "artifact_index.json",
)

REPORTING_CSV_COLUMNS = {
    "cv_fold_metrics.csv": (
        "model_key",
        "run_id",
        "feature_version",
        "split_version",
        "fold",
        "roc_auc",
        "fit_seconds",
        "predict_seconds",
        "warning_count",
    ),
    "rq1_metrics_long.csv": (
        "model_key",
        "run_id",
        "feature_version",
        "split_version",
        "partition",
        "metric",
        "estimate",
        "ci_low",
        "ci_high",
        "unit",
        "higher_is_better",
        "valid_replicates",
    ),
    "confusion_matrix.csv": (
        "model_key",
        "run_id",
        "actual_class",
        "predicted_class",
        "count",
    ),
    "calibration_curve.csv": (
        "model_key",
        "run_id",
        "bin",
        "lower_edge",
        "upper_edge",
        "count",
        "mean_p_approved",
        "observed_approval_rate",
    ),
}

REQUIRED_RQ1_METRICS = {
    "roc_auc_approved",
    "average_precision_denied",
    "precision_denied",
    "recall_denied",
    "f1_denied",
    "recall_approved",
    "balanced_accuracy",
    "brier_score_approved",
    "calibration_intercept",
    "calibration_slope",
}

MODEL_COLORS = {
    "lr": "#2F6B9A",
    "rf": "#D97A2B",
    "xgboost": "#C9A227",
    "plain_mlp": "#C05A8A",
    "realmlp": "#6B7D3C",
    "tabmmini": "#5B4B8A",
}


class ReportingContractError(ValueError):
    """Raised when persisted evidence cannot satisfy the reporting contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportingContractError(message)


def _load_json_object(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"Required reporting source is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"Expected a JSON object: {path}")
    return payload


def _find_project_root(path: Path) -> Path:
    for candidate in (path.resolve(), *path.resolve().parents):
        if (candidate / "config" / "dataset_b" / "model_protocol_v1.yaml").is_file():
            return candidate
    raise ReportingContractError(f"Cannot locate the project root from {path}.")


def build_lr07_reporting_summary(run_dir: str | Path) -> dict[str, Any]:
    """Validate persisted LR evidence through the latest completed stage."""

    root = Path(run_dir)
    manifest = _load_json_object(root / "manifest.json")
    cv = _load_json_object(root / "cv_results.json")
    full_train = _load_json_object(root / "full_train_result.json")
    project_root = _find_project_root(root)
    preprocessing_path = project_root / str(manifest.get("preprocessing_manifest", ""))
    preprocessing = _load_json_object(preprocessing_path)

    _require(manifest.get("status") == "PASS", "LR-07 manifest is not PASS.")
    _require(manifest.get("contract_id") == "LR-07", "Expected an LR-07 manifest.")
    _require(manifest.get("model_key") == "lr", "Expected model_key='lr'.")
    _require(manifest.get("feature_version") == FINAL_FEATURE_VERSION, "Not a final feature run.")
    _require(manifest.get("split_version") == FINAL_SPLIT_VERSION, "Unexpected split version.")
    _require(manifest.get("validation_accessed") is False, "LR-07 accessed Validation.")
    _require(manifest.get("test_accessed") is False, "LR-07 accessed Test.")
    _require(preprocessing.get("run_id") == manifest.get("run_id"), "LR-06/LR-07 run_id lineage differs.")
    _require(preprocessing.get("feature_version") == FINAL_FEATURE_VERSION, "LR-06 is not final-feature preprocessing.")
    _require(preprocessing.get("split_version") == FINAL_SPLIT_VERSION, "LR-06 uses an unexpected split.")
    _require(preprocessing.get("source_partition") == "train", "LR-06 is not Train-only.")
    _require(preprocessing.get("raw_feature_count") == 15, "LR-06 must contain 15 raw features.")
    _require(preprocessing.get("semantic_feature_count") == 19, "LR-06 must contain 19 semantic features.")
    _require(preprocessing.get("encoded_feature_count") == 62, "LR-06 must contain 62 LR features.")
    coverage = preprocessing.get("fold_validation_coverage", {})
    _require(coverage.get("rows") == EXPECTED_TRAIN_ROWS, "LR-06 coverage has an unexpected row count.")
    _require(coverage.get("exactly_once") is True, "LR-06 fold-validation coverage is not exactly once.")
    normalization = preprocessing.get("normalization_audit", {})
    _require(normalization.get("raw_equals_normalized") is True, "LR-06 exemption normalization failed.")
    _require(normalization.get("semantic_1111_count") == 0, "LR-06 retained semantic 1111 values.")
    _require(cv.get("run_id") == manifest.get("run_id") == full_train.get("run_id"), "Broken run_id lineage.")
    _require(cv.get("fold_count") == 5, "LR-07 must contain five CV folds.")
    _require(len(cv.get("folds", [])) == 5, "LR-07 fold payload is incomplete.")
    _require(cv.get("oof_rows") == EXPECTED_TRAIN_ROWS, "Unexpected LR OOF row count.")
    _require(cv.get("oof_unique_record_ids") == EXPECTED_TRAIN_ROWS, "LR OOF IDs are not complete and unique.")
    _require(cv.get("oof_coverage_min") == cv.get("oof_coverage_max") == 1, "LR OOF coverage is not exactly 1/1.")
    _require(cv.get("fresh_estimator_per_fold") is True, "LR CV does not record fresh estimators.")
    _require(cv.get("validation_accessed") is False and cv.get("test_accessed") is False, "LR CV crossed a holdout boundary.")
    _require(full_train.get("retained_train_rows") == EXPECTED_TRAIN_ROWS, "Unexpected full-Train row count.")
    _require(full_train.get("raw_feature_count") == 15, "Expected 15 raw features.")
    _require(full_train.get("semantic_feature_count") == 19, "Expected 19 semantic features.")
    _require(full_train.get("encoded_feature_count") == 62, "Expected 62 encoded LR features.")
    _require(full_train.get("matrix_format") == "csr", "Expected CSR full-Train matrix.")
    _require(full_train.get("matrix_dtype") == "float64", "Expected float64 LR matrix.")
    _require(full_train.get("matrix_finite") is True, "Full-Train matrix is not finite.")
    _require(full_train.get("fresh_preprocessor") is True, "Full-Train preprocessor is not fresh.")
    _require(full_train.get("fresh_estimator") is True, "Full-Train estimator is not fresh.")
    _require(full_train.get("validation_accessed") is False and full_train.get("test_accessed") is False, "Full-Train fit crossed a holdout boundary.")
    validation_dir = root / "validation"
    validation_payload = None
    validation_sources: dict[str, str] = {}
    validation_complete = validation_dir.is_dir()
    if validation_complete:
        from src.dataset_b.evaluation.lr_validation_threshold import (
            validate_lr_validation_threshold,
        )

        validation_manifest = validate_lr_validation_threshold(validation_dir)
        threshold = _load_json_object(validation_dir / "threshold_selection.json")
        prediction_manifest = _load_json_object(validation_dir / "prediction_manifest.json")
        _require(validation_manifest.get("run_id") == manifest.get("run_id"), "LR-07/LR-08 run_id lineage differs.")
        _require(threshold.get("run_id") == manifest.get("run_id"), "LR threshold run_id lineage differs.")
        _require(prediction_manifest.get("run_id") == manifest.get("run_id"), "LR predictions run_id lineage differs.")
        validation_payload = {
            "rows": validation_manifest["rows"],
            "unique_record_ids": validation_manifest["unique_record_ids"],
            "source_order_preserved": validation_manifest["source_order_preserved"],
            "threshold": threshold["selected_threshold"],
            "balanced_accuracy": threshold["selected_metrics"]["balanced_accuracy"],
            "recall_approved": threshold["selected_metrics"]["recall_approved"],
            "recall_denied": threshold["selected_metrics"]["recall_denied"],
            "maximizer_count": threshold["maximizer_count"],
            "probability_min": prediction_manifest["probability_min"],
            "probability_max": prediction_manifest["probability_max"],
            "ordered_record_ids_verified": True,
        }
        validation_sources = {
            "validation_manifest": str(validation_dir / "manifest.json"),
            "validation_predictions": str(validation_dir / "validation_predictions.csv"),
            "threshold_candidates": str(validation_dir / "threshold_candidates.csv"),
            "threshold_selection": str(validation_dir / "threshold_selection.json"),
        }

    test_dir = root / "test"
    test_payload = None
    test_sources: dict[str, str] = {}
    test_complete = test_dir.is_dir()
    if test_complete:
        from src.dataset_b.evaluation.lr_test_evaluation import validate_lr_test_evaluation

        test_manifest = validate_lr_test_evaluation(test_dir, run_dir=root)
        test_metrics = _load_json_object(test_dir / "rq1_test_metrics.json")
        _require(test_manifest.get("run_id") == manifest.get("run_id"), "LR-07/LR-09 run_id lineage differs.")
        _require(test_metrics.get("run_id") == manifest.get("run_id"), "LR Test metric run_id lineage differs.")
        test_payload = {
            "rows": test_manifest["rows"],
            "unique_record_ids": test_manifest["unique_record_ids"],
            "source_order_preserved": test_manifest["source_order_preserved"],
            "threshold": test_metrics["threshold"],
            "point_metrics": test_metrics["point_metrics"],
            "confidence_intervals_95_percentile": test_metrics["confidence_intervals_95_percentile"],
            "counts": test_metrics["counts"],
            "calibration_curve": test_metrics["calibration_curve"],
            "ordered_record_ids_verified": True,
            "bootstrap_indices": str(project_root / "artifacts/dataset_b/rq1_test_bootstrap_indices.npz"),
        }
        test_sources = {
            "test_manifest": str(test_dir / "run_manifest.json"),
            "test_predictions": str(test_dir / "test_predictions.csv"),
            "test_metrics": str(test_dir / "rq1_test_metrics.json"),
            "test_bootstrap": str(test_dir / "rq1_bootstrap_metrics.csv"),
        }

    summary = {
        "schema_version": REPORTING_SCHEMA_VERSION,
        "reporting_status": "PARTIAL",
        "model_key": "lr",
        "run_id": manifest["run_id"],
        "feature_version": manifest["feature_version"],
        "split_version": manifest["split_version"],
        "preprocessing_version": manifest["preprocessing_version"],
        "stages": {
            "train_only_cv": "COMPLETE",
            "full_train_fit": "COMPLETE",
            "validation_prediction": "COMPLETE" if validation_complete else "PENDING",
            "threshold_selection": "COMPLETE" if validation_complete else "PENDING",
            "test_evaluation": "COMPLETE" if test_complete else "PENDING",
            "final_review": "PENDING",
        },
        "data_boundaries": {
            "train_rows": EXPECTED_TRAIN_ROWS,
            "validation_rows": EXPECTED_VALIDATION_ROWS if validation_complete else None,
            "test_rows": EXPECTED_TEST_ROWS if test_complete else None,
            "validation_accessed": validation_complete,
            "test_accessed": test_complete,
        },
        "feature_counts": {"raw": 15, "semantic": 19, "model_input": 62},
        "cv": {
            "fold_count": cv["fold_count"],
            "mean_roc_auc": cv["mean_fold_roc_auc"],
            "std_roc_auc": cv["std_fold_roc_auc"],
            "oof_roc_auc": cv["oof_roc_auc"],
            "oof_rows": cv["oof_rows"],
            "convergence_warning_count": sum(
                int(fold.get("convergence_warning_count", 0)) for fold in cv["folds"]
            ),
        },
        "full_train": {
            "fit_rows": full_train["retained_train_rows"],
            "fit_seconds": full_train["fit_seconds"],
            "n_iter": full_train["n_iter"],
            "convergence_warning_count": full_train["convergence_warning_count"],
        },
        "validation": validation_payload,
        "test": test_payload,
        "artifact_sources": {
            "preprocessing_manifest": str(preprocessing_path),
            "manifest": str(root / "manifest.json"),
            "cv_results": str(root / "cv_results.json"),
            "full_train_result": str(root / "full_train_result.json"),
            **validation_sources,
            **test_sources,
        },
    }
    validate_model_summary(summary, require_complete=False)
    return summary


def _canonicalize_xgboost_metrics(metrics: Any, *, field_name: str) -> dict[str, Any]:
    _require(isinstance(metrics, Mapping), f"XGBoost {field_name} is missing.")
    canonical = dict(metrics)
    if "brier_approved" in canonical:
        _require(
            "brier_score_approved" not in canonical,
            f"XGBoost {field_name} contains both stored and canonical Brier names.",
        )
        canonical["brier_score_approved"] = canonical.pop("brier_approved")
    _require(set(canonical) == REQUIRED_RQ1_METRICS, f"XGBoost {field_name} does not contain the required ten metrics.")
    return canonical


def adapt_xgboost_reporting_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one validated XGB-10 summary to the canonical reporting schema."""

    _require(summary.get("contract_id") == "XGB-10", "Expected frozen XGB-10 evidence.")
    _require(summary.get("status") == "PASS", "XGB-10 evidence is not PASS.")
    _require(summary.get("model_key") in {"xgb", "xgboost"}, "Expected computational model key 'xgb'.")
    _require(summary.get("protocol_model_key", "xgb") == "xgb", "Unexpected XGBoost protocol model key.")
    adapted = deepcopy(dict(summary))
    adapted["model_key"] = "xgboost"
    adapted["protocol_model_key"] = "xgb"
    test = adapted.get("test")
    _require(isinstance(test, dict), "XGB-10 summary is missing Test evidence.")
    test["point_metrics"] = _canonicalize_xgboost_metrics(test.get("point_metrics"), field_name="point metrics")
    test["confidence_intervals_95_percentile"] = _canonicalize_xgboost_metrics(
        test.get("confidence_intervals_95_percentile"),
        field_name="confidence intervals",
    )
    if "point_metrics" in adapted:
        adapted["point_metrics"] = _canonicalize_xgboost_metrics(adapted["point_metrics"], field_name="top-level point metrics")
    if "confidence_intervals_95_percentile" in adapted:
        adapted["confidence_intervals_95_percentile"] = _canonicalize_xgboost_metrics(
            adapted["confidence_intervals_95_percentile"],
            field_name="top-level confidence intervals",
        )
    validate_model_summary(adapted, require_complete=True)
    return adapted


def _xgboost_cv_reporting_frame(summary: Mapping[str, Any], cv_results: Mapping[str, Any]) -> pd.DataFrame:
    folds = cv_results.get("folds")
    _require(isinstance(folds, list) and len(folds) == 5, "XGBoost reporting requires exactly five CV folds.")
    rows = [
        {
            "model_key": "xgboost",
            "run_id": summary["run_id"],
            "feature_version": summary["feature_version"],
            "split_version": summary["split_version"],
            "fold": int(fold["fold"]),
            "roc_auc": float(fold["roc_auc"]),
            "fit_seconds": float(fold["fit_seconds"]),
            "predict_seconds": float(fold["predict_seconds"]),
            "warning_count": int(fold.get("warning_count", 0)),
        }
        for fold in folds
    ]
    return pd.DataFrame(rows, columns=REPORTING_CSV_COLUMNS["cv_fold_metrics.csv"])


def _xgboost_metric_reporting_frame(summary: Mapping[str, Any]) -> pd.DataFrame:
    test = summary["test"]
    point = test["point_metrics"]
    intervals = test["confidence_intervals_95_percentile"]
    rows = []
    for name in sorted(REQUIRED_RQ1_METRICS):
        interval = intervals[name]
        rows.append(
            {
                "model_key": "xgboost",
                "run_id": summary["run_id"],
                "feature_version": summary["feature_version"],
                "split_version": summary["split_version"],
                "partition": "test",
                "metric": name,
                "estimate": float(point[name]),
                "ci_low": interval["lower_2_5"],
                "ci_high": interval["upper_97_5"],
                "unit": "coefficient" if name.startswith("calibration_") else "rate",
                "higher_is_better": name not in {"brier_score_approved", "calibration_intercept", "calibration_slope"},
                "valid_replicates": int(interval["valid_replicates"]),
            }
        )
    return pd.DataFrame(rows, columns=REPORTING_CSV_COLUMNS["rq1_metrics_long.csv"])


def _xgboost_confusion_reporting_frame(summary: Mapping[str, Any]) -> pd.DataFrame:
    counts = summary["test"]["counts"]
    cells = (
        ("approved", "approved", "tp_approved"),
        ("approved", "denied", "fn_approved"),
        ("denied", "approved", "fp_denied"),
        ("denied", "denied", "tn_denied"),
    )
    rows = [
        {
            "model_key": "xgboost",
            "run_id": summary["run_id"],
            "actual_class": actual,
            "predicted_class": predicted,
            "count": int(counts[key]),
        }
        for actual, predicted, key in cells
    ]
    return pd.DataFrame(rows, columns=REPORTING_CSV_COLUMNS["confusion_matrix.csv"])


def _xgboost_calibration_reporting_frame(summary: Mapping[str, Any]) -> pd.DataFrame:
    rows = [
        {"model_key": "xgboost", "run_id": summary["run_id"], **dict(row)}
        for row in summary["test"]["calibration_curve"]
    ]
    return pd.DataFrame(rows, columns=REPORTING_CSV_COLUMNS["calibration_curve.csv"])


def write_xgboost_reporting_package(summary: Mapping[str, Any], *, run_dir: str | Path) -> dict[str, Any]:
    """Atomically write the six reporting files from frozen XGB-10 evidence."""

    adapted = adapt_xgboost_reporting_summary(summary)
    root = Path(run_dir).resolve(strict=True)
    _require(root.name == adapted["run_id"], "XGBoost reporting run_dir does not match run_id.")
    cv_source = Path(str(adapted.get("artifact_sources", {}).get("cv_results", ""))).resolve(strict=True)
    _require(cv_source.parent == root, "XGBoost CV reporting source is outside the reviewed run.")
    cv_results = _load_json_object(cv_source)
    _require(cv_results.get("run_id") == adapted["run_id"], "XGBoost CV/reporting run_id lineage differs.")

    reporting_dir = root / "reporting"
    temporary_dir = root / ".reporting.tmp"
    _require(not reporting_dir.exists(), "XGBoost reporting package already exists.")
    _require(not temporary_dir.exists(), "XGBoost temporary reporting package already exists.")
    frames = {
        "cv_fold_metrics.csv": _xgboost_cv_reporting_frame(adapted, cv_results),
        "rq1_metrics_long.csv": _xgboost_metric_reporting_frame(adapted),
        "confusion_matrix.csv": _xgboost_confusion_reporting_frame(adapted),
        "calibration_curve.csv": _xgboost_calibration_reporting_frame(adapted),
    }
    artifact_index = {
        "schema_version": REPORTING_SCHEMA_VERSION,
        "model_key": "xgboost",
        "protocol_model_key": "xgb",
        "run_id": adapted["run_id"],
        "artifacts": [
            {"name": name, "path": str(path)}
            for name, path in sorted(adapted["artifact_sources"].items())
        ],
    }
    temporary_dir.mkdir()
    try:
        (temporary_dir / "model_summary.json").write_text(
            json.dumps(adapted, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for filename, frame in frames.items():
            frame.to_csv(temporary_dir / filename, index=False)
        (temporary_dir / "artifact_index.json").write_text(
            json.dumps(artifact_index, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validate_reporting_directory(temporary_dir)
        temporary_dir.replace(reporting_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return validate_reporting_directory(reporting_dir)


def write_lr_reporting_package(summary: Mapping[str, Any], *, run_dir: str | Path) -> dict[str, Any]:
    """Write the common reporting package from a completed direct LR review."""

    validate_model_summary(summary, require_complete=True)
    _require(summary.get("model_key") == "lr", "Expected completed LR evidence.")
    root = Path(run_dir).resolve(strict=True)
    _require(root.name == summary["run_id"], "LR reporting run_dir does not match run_id.")
    cv_source = Path(str(summary.get("artifact_sources", {}).get("cv_results", ""))).resolve(strict=True)
    _require(cv_source.parent == root, "LR CV reporting source is outside the reviewed run.")
    cv_results = _load_json_object(cv_source)
    folds = cv_results.get("folds")
    _require(isinstance(folds, list) and len(folds) == 5, "LR reporting requires exactly five CV folds.")

    cv_frame = pd.DataFrame(
        [
            {
                "model_key": "lr",
                "run_id": summary["run_id"],
                "feature_version": summary["feature_version"],
                "split_version": summary["split_version"],
                "fold": int(fold["fold"]),
                "roc_auc": float(fold["roc_auc"]),
                "fit_seconds": float(fold["fit_seconds"]),
                "predict_seconds": 0.0,
                "warning_count": int(fold.get("convergence_warning_count", 0)),
            }
            for fold in folds
        ],
        columns=REPORTING_CSV_COLUMNS["cv_fold_metrics.csv"],
    )
    test = summary["test"]
    metric_rows = []
    for name in sorted(REQUIRED_RQ1_METRICS):
        interval = test["confidence_intervals_95_percentile"][name]
        metric_rows.append(
            {
                "model_key": "lr",
                "run_id": summary["run_id"],
                "feature_version": summary["feature_version"],
                "split_version": summary["split_version"],
                "partition": "test",
                "metric": name,
                "estimate": float(test["point_metrics"][name]),
                "ci_low": interval["lower_2_5"],
                "ci_high": interval["upper_97_5"],
                "unit": "coefficient" if name.startswith("calibration_") else "rate",
                "higher_is_better": name not in {"brier_score_approved", "calibration_intercept", "calibration_slope"},
                "valid_replicates": int(interval["valid_replicates"]),
            }
        )
    metric_frame = pd.DataFrame(metric_rows, columns=REPORTING_CSV_COLUMNS["rq1_metrics_long.csv"])
    cells = (
        ("approved", "approved", "tp_approved"),
        ("approved", "denied", "fn_approved"),
        ("denied", "approved", "fp_denied"),
        ("denied", "denied", "tn_denied"),
    )
    confusion_frame = pd.DataFrame(
        [
            {"model_key": "lr", "run_id": summary["run_id"], "actual_class": actual, "predicted_class": predicted, "count": int(test["counts"][key])}
            for actual, predicted, key in cells
        ],
        columns=REPORTING_CSV_COLUMNS["confusion_matrix.csv"],
    )
    calibration_frame = pd.DataFrame(
        [{"model_key": "lr", "run_id": summary["run_id"], **dict(row)} for row in test["calibration_curve"]],
        columns=REPORTING_CSV_COLUMNS["calibration_curve.csv"],
    )
    reporting_dir = root / "reporting"
    temporary_dir = root / ".reporting.tmp"
    _require(not reporting_dir.exists(), "LR reporting package already exists.")
    _require(not temporary_dir.exists(), "LR temporary reporting package already exists.")
    temporary_dir.mkdir()
    try:
        (temporary_dir / "model_summary.json").write_text(json.dumps(dict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for filename, frame in {
            "cv_fold_metrics.csv": cv_frame,
            "rq1_metrics_long.csv": metric_frame,
            "confusion_matrix.csv": confusion_frame,
            "calibration_curve.csv": calibration_frame,
        }.items():
            frame.to_csv(temporary_dir / filename, index=False)
        artifacts = []
        for name, value in sorted(summary["artifact_sources"].items()):
            source = Path(str(value)).resolve(strict=True)
            try:
                relative = source.relative_to(root)
                indexed_path = str(Path("..") / relative)
            except ValueError:
                indexed_path = str(source)
            artifacts.append({"name": name, "path": indexed_path})
        (temporary_dir / "artifact_index.json").write_text(
            json.dumps({"schema_version": REPORTING_SCHEMA_VERSION, "model_key": "lr", "run_id": summary["run_id"], "artifacts": artifacts}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validate_reporting_directory(temporary_dir)
        temporary_dir.replace(reporting_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return validate_reporting_directory(reporting_dir)


def validate_model_summary(summary: Mapping[str, Any], *, require_complete: bool) -> None:
    """Validate the stable model-level summary consumed by notebook 07."""

    _require(summary.get("schema_version") == REPORTING_SCHEMA_VERSION, "Unsupported reporting schema version.")
    _require(summary.get("model_key") in FINAL_MODEL_KEYS, "Unknown final model key.")
    _require(isinstance(summary.get("run_id"), str) and bool(summary["run_id"]), "Missing run_id.")
    _require(summary.get("feature_version") == FINAL_FEATURE_VERSION, "Summary is not final-feature evidence.")
    _require(summary.get("split_version") == FINAL_SPLIT_VERSION, "Summary does not use split_v1.")
    _require(summary.get("reporting_status") in {"PARTIAL", "COMPLETE"}, "Invalid reporting status.")
    boundaries = summary.get("data_boundaries")
    _require(isinstance(boundaries, Mapping), "Missing data_boundaries.")
    _require(boundaries.get("train_rows") == EXPECTED_TRAIN_ROWS, "Unexpected Train row count.")
    if require_complete:
        _require(summary.get("reporting_status") == "COMPLETE", "A final comparison requires COMPLETE summaries.")
        _require(boundaries.get("validation_rows") == EXPECTED_VALIDATION_ROWS, "Unexpected Validation row count.")
        _require(boundaries.get("test_rows") == EXPECTED_TEST_ROWS, "Unexpected Test row count.")
        _require(summary.get("validation") is not None, "Missing Validation reporting payload.")
        _require(summary.get("test") is not None, "Missing Test reporting payload.")
        test = summary["test"]
        sources = summary.get("artifact_sources")
        _require(test.get("ordered_record_ids_verified") is True, "Test record order has not been verified.")
        _require(isinstance(test.get("bootstrap_indices"), str) and bool(test["bootstrap_indices"]), "Missing bootstrap indices source.")
        _require(isinstance(sources, Mapping) and isinstance(sources.get("test_predictions"), str), "Missing Test prediction source.")


def _read_test_record_ids(summary: Mapping[str, Any]) -> tuple[str, ...]:
    path = Path(str(summary["artifact_sources"]["test_predictions"]))
    _require(path.is_file(), f"Test prediction source is missing: {path}")
    try:
        frame = pd.read_csv(path, usecols=["record_id"], dtype={"record_id": str}, keep_default_na=False)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ReportingContractError(f"Cannot read Test record IDs: {path}") from exc
    values = tuple(frame["record_id"].astype(str).tolist())
    _require(len(values) == EXPECTED_TEST_ROWS and len(set(values)) == EXPECTED_TEST_ROWS, "Test record IDs are incomplete or duplicated.")
    return values


def _load_bootstrap_indices(path_value: Any) -> np.ndarray:
    path = Path(str(path_value))
    _require(path.is_file(), f"Bootstrap indices source is missing: {path}")
    try:
        with np.load(path, allow_pickle=False) as loaded:
            indices = np.asarray(loaded["indices"])
    except (OSError, ValueError, KeyError) as exc:
        raise ReportingContractError(f"Cannot read bootstrap indices: {path}") from exc
    _require(indices.shape == (EXPECTED_BOOTSTRAP_REPLICATES, EXPECTED_TEST_ROWS) and indices.dtype == np.int32, "Bootstrap indices shape or dtype differs.")
    _require(np.all((indices >= 0) & (indices < EXPECTED_TEST_ROWS)), "Bootstrap indices are out of bounds.")
    return indices


def validate_cross_model_summaries(summaries: Sequence[Mapping[str, Any]]) -> None:
    """Gate the six frozen summaries before any cross-model visualization."""

    _require(len(summaries) == len(FINAL_MODEL_KEYS), "Exactly six model summaries are required.")
    for summary in summaries:
        validate_model_summary(summary, require_complete=True)
    keys = [str(summary["model_key"]) for summary in summaries]
    _require(len(set(keys)) == len(keys), "Model summaries contain duplicate model keys.")
    _require(set(keys) == set(FINAL_MODEL_KEYS), "The required six model families are incomplete.")
    expected_ids = _read_test_record_ids(summaries[0])
    for summary in summaries[1:]:
        _require(_read_test_record_ids(summary) == expected_ids, "Test membership/order lineage differs across models.")
    bootstrap_paths = [Path(str(summary["test"]["bootstrap_indices"])).resolve() for summary in summaries]
    expected_indices = _load_bootstrap_indices(bootstrap_paths[0])
    checked_paths = {bootstrap_paths[0]}
    for path in bootstrap_paths[1:]:
        if path in checked_paths:
            continue
        candidate = _load_bootstrap_indices(path)
        _require(np.array_equal(candidate, expected_indices), "Bootstrap index lineage differs across models.")
        checked_paths.add(path)


def validate_reporting_directory(reporting_dir: str | Path) -> dict[str, Any]:
    """Validate one complete final-run reporting package from disk."""

    root = Path(reporting_dir)
    for filename in EXPECTED_REPORTING_FILES:
        _require((root / filename).is_file(), f"Reporting package is missing {filename}.")

    summary = _load_json_object(root / "model_summary.json")
    validate_model_summary(summary, require_complete=True)
    model_key = str(summary["model_key"])
    run_id = str(summary["run_id"])

    frames: dict[str, pd.DataFrame] = {}
    for filename, expected_columns in REPORTING_CSV_COLUMNS.items():
        frame = pd.read_csv(root / filename)
        _require(tuple(frame.columns) == expected_columns, f"{filename} has an unexpected column schema.")
        _require(not frame.empty, f"{filename} is empty.")
        _require(set(frame["model_key"].astype(str)) == {model_key}, f"{filename} has inconsistent model_key values.")
        _require(set(frame["run_id"].astype(str)) == {run_id}, f"{filename} has inconsistent run_id values.")
        frames[filename] = frame

    cv = frames["cv_fold_metrics.csv"]
    _require(sorted(cv["fold"].astype(int).tolist()) == [1, 2, 3, 4, 5], "CV reporting rows must cover folds 1-5 exactly once.")
    metrics = set(frames["rq1_metrics_long.csv"]["metric"].astype(str))
    _require(metrics == REQUIRED_RQ1_METRICS, "RQ1 reporting metrics do not match the required ten-metric contract.")
    confusion = frames["confusion_matrix.csv"]
    expected_cells = {
        ("approved", "approved"),
        ("approved", "denied"),
        ("denied", "approved"),
        ("denied", "denied"),
    }
    observed_cells = set(zip(confusion["actual_class"].astype(str), confusion["predicted_class"].astype(str)))
    _require(observed_cells == expected_cells and len(confusion) == 4, "Confusion reporting must contain the four class cells exactly once.")

    artifact_index = _load_json_object(root / "artifact_index.json")
    _require(artifact_index.get("schema_version") == REPORTING_SCHEMA_VERSION, "Artifact index schema version differs.")
    _require(artifact_index.get("model_key") == model_key, "Artifact index model_key differs.")
    _require(artifact_index.get("run_id") == run_id, "Artifact index run_id differs.")
    artifacts = artifact_index.get("artifacts")
    _require(isinstance(artifacts, list) and artifacts, "Artifact index is empty.")
    for artifact in artifacts:
        _require(isinstance(artifact, Mapping), "Artifact index entry is not an object.")
        path_value = artifact.get("path")
        _require(isinstance(path_value, str) and bool(path_value), "Artifact index entry has no path.")
        artifact_path = Path(path_value)
        if not artifact_path.is_absolute():
            artifact_path = (root / artifact_path).resolve()
        _require(artifact_path.is_file(), f"Indexed artifact is missing: {artifact_path}")
    return summary


def cv_fold_frame(cv_results: Mapping[str, Any], *, model_key: str = "lr") -> pd.DataFrame:
    """Return the common tidy CV fold table used by all final notebooks."""

    _require(model_key in FINAL_MODEL_KEYS, "Unknown model key for CV table.")
    folds = cv_results.get("folds")
    _require(isinstance(folds, list) and folds, "CV results contain no folds.")
    rows = []
    for fold in folds:
        rows.append(
            {
                "model_key": model_key,
                "fold": int(fold["fold"]),
                "train_rows": int(fold["train_rows"]),
                "fold_validation_rows": int(fold["fold_validation_rows"]),
                "roc_auc": float(fold["roc_auc"]),
                "fit_seconds": float(fold["fit_seconds"]),
                "convergence_warning_count": int(fold.get("convergence_warning_count", 0)),
            }
        )
    frame = pd.DataFrame(rows).sort_values("fold", kind="stable").reset_index(drop=True)
    _require(frame["fold"].tolist() == list(range(1, len(frame) + 1)), "CV fold numbering is not contiguous.")
    return frame


def plot_cv_roc_auc(frame: pd.DataFrame, *, model_key: str, mean_roc_auc: float):
    """Render the standard fold dot plot used by final model notebooks."""

    _require(model_key in FINAL_MODEL_KEYS, "Unknown model key for plot styling.")
    required = {"fold", "roc_auc"}
    _require(required.issubset(frame.columns), "CV plot input is missing required columns.")
    import matplotlib.pyplot as plt

    color = MODEL_COLORS[model_key]
    figure, axis = plt.subplots(figsize=(8.4, 4.6), constrained_layout=True)
    axis.scatter(frame["fold"], frame["roc_auc"], s=72, color=color, edgecolor="#263238", linewidth=0.8, zorder=3)
    axis.axhline(mean_roc_auc, color="#37474F", linestyle="--", linewidth=1.4, label=f"Mean = {mean_roc_auc:.6f}")
    for row in frame.itertuples(index=False):
        axis.annotate(f"{row.roc_auc:.6f}", (row.fold, row.roc_auc), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
    axis.set_title("Train-only five-fold ROC-AUC", loc="left", fontweight="bold")
    axis.set_xlabel("Fold")
    axis.set_ylabel("ROC-AUC")
    axis.set_xticks(frame["fold"].tolist())
    axis.grid(axis="y", color="#D9DEE7", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="lower right")
    return figure, axis


def plot_threshold_balanced_accuracy(
    frame: pd.DataFrame,
    *,
    model_key: str,
    selected_threshold: float,
):
    """Render the standard Validation threshold-selection curve."""

    _require(model_key in FINAL_MODEL_KEYS, "Unknown model key for plot styling.")
    required = {"threshold", "balanced_accuracy"}
    _require(required.issubset(frame.columns), "Threshold plot input is missing required columns.")
    selected = frame.loc[frame["threshold"].eq(selected_threshold)]
    _require(len(selected) == 1, "Selected threshold is absent or duplicated in candidate data.")
    import matplotlib.pyplot as plt

    color = MODEL_COLORS[model_key]
    selected_ba = float(selected.iloc[0]["balanced_accuracy"])
    figure, axis = plt.subplots(figsize=(8.4, 4.6), constrained_layout=True)
    axis.plot(frame["threshold"], frame["balanced_accuracy"], color=color, linewidth=2.0)
    axis.scatter([selected_threshold], [selected_ba], s=78, color="#C62828", edgecolor="white", linewidth=0.9, zorder=3)
    axis.annotate(
        f"Selected = {selected_threshold:.3f}\nBA = {selected_ba:.6f}",
        (selected_threshold, selected_ba),
        xytext=(16, -6),
        textcoords="offset points",
        fontsize=9,
    )
    axis.set_title("Validation threshold selection by balanced accuracy", loc="left", fontweight="bold")
    axis.set_xlabel("Decision threshold")
    axis.set_ylabel("Balanced accuracy")
    axis.grid(color="#D9DEE7", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    return figure, axis


def plot_calibration_curve(rows: Sequence[Mapping[str, Any]], *, model_key: str):
    """Render the standard Test calibration plot from frozen bin summaries."""

    _require(model_key in FINAL_MODEL_KEYS, "Unknown model key for plot styling.")
    frame = pd.DataFrame(rows)
    required = {"mean_p_approved", "observed_approval_rate", "count"}
    _require(not frame.empty and required.issubset(frame.columns), "Calibration plot input is incomplete.")
    import matplotlib.pyplot as plt

    color = MODEL_COLORS[model_key]
    figure, axis = plt.subplots(figsize=(6.7, 5.2), constrained_layout=True)
    axis.plot([0, 1], [0, 1], color="#607D8B", linestyle="--", linewidth=1.25, label="Perfect calibration")
    axis.plot(
        frame["mean_p_approved"],
        frame["observed_approval_rate"],
        marker="o",
        color=color,
        linewidth=2.0,
        label=f"{model_key.upper()} Test bins",
    )
    axis.set_title("Test calibration curve", loc="left", fontweight="bold")
    axis.set_xlabel("Mean predicted approval probability")
    axis.set_ylabel("Observed approval rate")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(color="#D9DEE7", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="lower right")
    return figure, axis
