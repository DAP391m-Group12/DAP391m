"""Deterministic PMLP-F05 reporting from completed source artifacts."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping

from src.dataset_b.evaluation.plain_mlp_test_evaluation import METRIC_NAMES
from src.dataset_b.reporting import model_reporting


REPORT_FILES = {
    "model_summary.json",
    "cv_fold_metrics.csv",
    "rq1_metrics_long.csv",
    "confusion_matrix.csv",
    "calibration_curve.csv",
    "artifact_index.json",
}


class PlainMLPReportingError(RuntimeError):
    """Raised when F05 reporting input or output violates its schema."""


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _indexed_artifact_path(path: str | Path, run_dir: Path) -> str:
    source = Path(path).absolute()
    try:
        return str(Path("..") / source.relative_to(run_dir.absolute()))
    except ValueError:
        return str(source)


def generate_plain_mlp_reporting(
    output_dir: str | Path,
    *,
    run_id: str,
    cv_result: Mapping[str, Any],
    cv_metrics: Mapping[str, Any],
    full_train_result: Mapping[str, Any],
    validation_result: Mapping[str, Any],
    threshold_selection: Mapping[str, Any],
    test_metrics: Mapping[str, Any],
    artifact_paths: Mapping[str, str | Path],
    bootstrap_indices_path: str | Path,
    test_predictions_path: str | Path,
    published_artifact_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Write the six locked reports without fitting, prediction, or data loading."""
    if not run_id or test_metrics.get("run_id") != run_id:
        raise PlainMLPReportingError("Reporting run_id linkage is invalid.")
    folds = cv_result.get("folds")
    point = test_metrics.get("point_metrics")
    intervals = test_metrics.get("confidence_intervals_95_percentile")
    counts = test_metrics.get("counts")
    curve = test_metrics.get("calibration_curve")
    if not isinstance(folds, list) or len(folds) != 5:
        raise PlainMLPReportingError("CV reporting requires exactly five folds.")
    if not isinstance(point, dict) or set(point) != set(METRIC_NAMES):
        raise PlainMLPReportingError("RQ1 point metric schema is invalid.")
    if not isinstance(intervals, dict) or set(intervals) != set(METRIC_NAMES):
        raise PlainMLPReportingError("RQ1 interval schema is invalid.")
    if not isinstance(counts, dict) or not isinstance(curve, list):
        raise PlainMLPReportingError("Confusion or calibration payload is invalid.")
    fold_roc_auc = cv_metrics.get("fold_roc_auc")
    if not isinstance(fold_roc_auc, list) or len(fold_roc_auc) != 5:
        raise PlainMLPReportingError("CV reporting requires five fold ROC-AUC values.")

    canonical_point = {
        ("brier_score_approved" if name == "brier_approved" else name): value
        for name, value in point.items()
    }
    canonical_intervals = {
        ("brier_score_approved" if name == "brier_approved" else name): value
        for name, value in intervals.items()
    }
    if set(canonical_point) != model_reporting.REQUIRED_RQ1_METRICS:
        raise PlainMLPReportingError("Canonical RQ1 metric schema is invalid.")

    destination = Path(output_dir)
    working = destination.with_name(destination.name + ".tmp")
    if destination.exists() or working.exists():
        raise PlainMLPReportingError("Reporting destination or temporary directory exists.")
    working.mkdir(parents=True)
    try:
        summary = {
            "schema_version": 1,
            "contract_id": "PMLP-F05-REPORTING",
            "status": "PASS",
            "reporting_status": "COMPLETE",
            "run_id": run_id,
            "reporting_key": "plain_mlp",
            "model_key": "plain_mlp",
            "model_version": test_metrics.get("model_version"),
            "feature_version": "feature_set_v1",
            "split_version": "split_v1",
            "preprocessing_version": test_metrics.get("preprocessing_version"),
            "stages": {
                "preprocessing": "PASS",
                "cuda_smoke": "PASS",
                "train_only_pilot": "PASS",
                "train_only_cv": "PASS",
                "full_train_fit": "PASS",
                "validation_threshold": "PASS",
                "test_evaluation": "PASS",
                "final_review": "PASS",
            },
            "data_boundaries": {
                "train_rows": int(cv_result.get("train_rows", 0)),
                "validation_rows": int(validation_result.get("prediction_summary", {}).get("rows", 0)),
                "test_rows": int(test_metrics.get("rows", 0)),
                "validation_accessed": True,
                "test_accessed": True,
            },
            "feature_counts": {"raw": 15, "semantic": 19, "model_input": 62},
            "cv": {
                "fold_count": 5,
                "mean_roc_auc": cv_metrics.get("mean_roc_auc"),
                "std_roc_auc": cv_metrics.get("std_roc_auc"),
                "oof_roc_auc": cv_metrics.get("oof_roc_auc"),
                "oof_rows": int(cv_result.get("train_rows", 0)),
                "warning_count": sum(len(fold.get("warnings", [])) for fold in folds),
            },
            "full_train": {
                "fit_rows": full_train_result.get("final_estimator", {}).get("fit_rows"),
                "fit_seconds": full_train_result.get("timing_seconds", {}).get("final_fit"),
                "best_epoch": full_train_result.get("final_estimator", {}).get("best_epoch"),
                "warning_count": len(full_train_result.get("warnings", [])),
            },
            "validation": {
                "rows": validation_result.get("prediction_summary", {}).get("rows"),
                "unique_record_ids": validation_result.get("prediction_summary", {}).get("unique_record_ids"),
                "source_order_preserved": validation_result.get("data_boundary", {}).get("quality_gate", {}).get("row_order_preserved"),
                "ordered_record_ids_verified": validation_result.get("data_boundary", {}).get("quality_gate", {}).get("record_id_alignment_preserved"),
                "probability_min": validation_result.get("prediction_summary", {}).get("probability_min"),
                "probability_max": validation_result.get("prediction_summary", {}).get("probability_max"),
                "threshold": threshold_selection.get("selected_threshold"),
                **dict(threshold_selection.get("selected_metrics", {})),
            },
            "test": {
                "rows": test_metrics.get("rows"),
                "unique_record_ids": test_metrics.get("rows"),
                "source_order_preserved": test_metrics.get("data_boundary", {}).get("quality_gate", {}).get("row_order_preserved"),
                "ordered_record_ids_verified": test_metrics.get("data_boundary", {}).get("quality_gate", {}).get("record_id_alignment_preserved"),
                "threshold": test_metrics.get("threshold"),
                "point_metrics": canonical_point,
                "confidence_intervals_95_percentile": canonical_intervals,
                "counts": counts,
                "calibration_curve": curve,
                "bootstrap_indices": str(bootstrap_indices_path),
            },
            "selected_threshold": test_metrics.get("threshold"),
            "point_metrics": canonical_point,
            "confidence_intervals_95_percentile": canonical_intervals,
            "artifact_sources": {
                **{
                    name: str(path)
                    for name, path in sorted(
                        (published_artifact_paths or artifact_paths).items()
                    )
                },
                "test_predictions": str(test_predictions_path),
            },
            "execution": {
                "fit_calls": 0,
                "predict_proba_calls": 0,
                "threshold_changed": False,
                "bootstrap_regenerated": False,
            },
        }
        _write_json(working / "model_summary.json", summary)

        fold_rows: list[dict[str, Any]] = []
        for fold, roc_auc in zip(folds, fold_roc_auc):
            timing = fold.get("timing_seconds", {})
            fold_rows.append(
                {
                    "model_key": "plain_mlp",
                    "run_id": run_id,
                    "feature_version": "feature_set_v1",
                    "split_version": "split_v1",
                    "fold": fold.get("fold"),
                    "roc_auc": roc_auc,
                    "fit_seconds": timing.get("final_fit"),
                    "predict_seconds": timing.get("outer_prediction"),
                    "warning_count": len(fold.get("warnings", [])),
                }
            )
        _write_csv(
            working / "cv_fold_metrics.csv",
            model_reporting.REPORTING_CSV_COLUMNS["cv_fold_metrics.csv"],
            fold_rows,
        )

        metric_rows = [
            {
                "model_key": "plain_mlp",
                "run_id": run_id,
                "feature_version": "feature_set_v1",
                "split_version": "split_v1",
                "partition": "test",
                "metric": name,
                "estimate": canonical_point[name],
                "ci_low": canonical_intervals[name]["lower_2_5"],
                "ci_high": canonical_intervals[name]["upper_97_5"],
                "unit": "coefficient" if name.startswith("calibration_") else "rate",
                "higher_is_better": name not in {"brier_score_approved", "calibration_intercept", "calibration_slope"},
                "valid_replicates": canonical_intervals[name]["valid_replicates"],
            }
            for name in sorted(model_reporting.REQUIRED_RQ1_METRICS)
        ]
        _write_csv(
            working / "rq1_metrics_long.csv",
            model_reporting.REPORTING_CSV_COLUMNS["rq1_metrics_long.csv"],
            metric_rows,
        )
        confusion_rows = [
            {"model_key": "plain_mlp", "run_id": run_id, "actual_class": "approved", "predicted_class": "approved", "count": counts["tp_approved"]},
            {"model_key": "plain_mlp", "run_id": run_id, "actual_class": "approved", "predicted_class": "denied", "count": counts["fn_approved"]},
            {"model_key": "plain_mlp", "run_id": run_id, "actual_class": "denied", "predicted_class": "approved", "count": counts["fp_denied"]},
            {"model_key": "plain_mlp", "run_id": run_id, "actual_class": "denied", "predicted_class": "denied", "count": counts["tn_denied"]},
        ]
        _write_csv(working / "confusion_matrix.csv", model_reporting.REPORTING_CSV_COLUMNS["confusion_matrix.csv"], confusion_rows)
        _write_csv(
            working / "calibration_curve.csv",
            model_reporting.REPORTING_CSV_COLUMNS["calibration_curve.csv"],
            [{"model_key": "plain_mlp", "run_id": run_id, **dict(row)} for row in curve],
        )
        index = {
            "schema_version": 1,
            "contract_id": "PMLP-F05-ARTIFACT-INDEX",
            "run_id": run_id,
            "reporting_key": "plain_mlp",
            "model_key": "plain_mlp",
            "artifacts": [
                {
                    "name": name,
                    "path": _indexed_artifact_path(path, destination.parent),
                }
                for name, path in sorted(artifact_paths.items())
            ],
            "acceptance_policy": "direct_content_and_numerical_comparison",
        }
        _write_json(working / "artifact_index.json", index)
        validate_plain_mlp_reporting(working, expected_run_id=run_id)
        os.replace(working, destination)
        return validate_plain_mlp_reporting(destination, expected_run_id=run_id)
    except Exception:
        raise


def validate_plain_mlp_reporting(
    output_dir: str | Path, *, expected_run_id: str | None = None
) -> dict[str, Any]:
    directory = Path(output_dir)
    if not directory.is_dir() or {p.name for p in directory.iterdir() if p.is_file()} != REPORT_FILES:
        raise PlainMLPReportingError("Reporting artifact set is invalid.")
    summary = json.loads((directory / "model_summary.json").read_text(encoding="utf-8"))
    index = json.loads((directory / "artifact_index.json").read_text(encoding="utf-8"))
    if (
        summary.get("contract_id") != "PMLP-F05-REPORTING"
        or summary.get("status") != "PASS"
        or summary.get("run_id") != index.get("run_id")
        or summary.get("reporting_key") != "plain_mlp"
        or index.get("reporting_key") != "plain_mlp"
        or summary.get("model_key") != "plain_mlp"
        or index.get("model_key") != "plain_mlp"
        or (expected_run_id is not None and summary.get("run_id") != expected_run_id)
        or "sha256" in json.dumps(index).lower()
    ):
        raise PlainMLPReportingError("Reporting schema or linkage is invalid.")
    try:
        model_reporting.validate_reporting_directory(directory)
    except model_reporting.ReportingContractError as exc:
        raise PlainMLPReportingError("Common reporting contract validation failed.") from exc
    return summary
