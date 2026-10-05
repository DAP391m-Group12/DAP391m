"""Common-schema reporting package for a completed TabMmini final run."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.dataset_b.evaluation.plain_mlp_test_evaluation import METRIC_NAMES
from src.dataset_b.evaluation.tabmmini_final_evaluation import (
    DEFAULT_ARTIFACT_ROOT,
    TabMMiniFinalPaths,
    validate_tabmmini_prediction_stage,
    validate_tabmmini_test_evaluation,
    validate_tabmmini_threshold,
)
from src.dataset_b.reporting.model_reporting import (
    REPORTING_CSV_COLUMNS,
    REPORTING_SCHEMA_VERSION,
    validate_reporting_directory,
)


class TabMMiniReportingError(RuntimeError):
    """Raised when persisted TabMmini evidence cannot form a report."""


def write_tabmmini_reporting(
    run_id: str,
    *,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    cv_metrics_path: str | Path | None = None,
) -> dict[str, Any]:
    """Read completed artifacts and atomically write the shared reporting schema."""
    paths = TabMMiniFinalPaths.for_run(run_id, artifact_root=artifact_root)
    full_train = _read_json(paths.stage("full_train") / "full_train_result.json")
    validation = validate_tabmmini_prediction_stage(paths, "validation")
    threshold = validate_tabmmini_threshold(paths)
    test = validate_tabmmini_test_evaluation(paths)
    cv_source = Path(cv_metrics_path) if cv_metrics_path else paths.stage("cv") / "cv_result.json"
    cv = _read_json(cv_source)
    _require_run_id(run_id, full_train, cv, validation, threshold, test)
    folds = cv.get("folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise TabMMiniReportingError("TabMmini reporting requires five CV fold rows.")

    destination = paths.stage("reporting")
    temporary = destination.with_name(destination.name + ".working")
    if destination.exists() or temporary.exists():
        raise FileExistsError("TabMmini reporting output or preserved attempt already exists.")
    temporary.mkdir(parents=True)
    try:
        summary = {
            "schema_version": REPORTING_SCHEMA_VERSION,
            "reporting_status": "COMPLETE",
            "model_key": "tabmmini",
            "run_id": run_id,
            "feature_version": "feature_set_v1",
            "split_version": "split_v1",
            "preprocessing_version": "dataset_b_tabmmini_native_v1",
            "data_boundaries": {
                "train_rows": full_train["train_rows"],
                "validation_rows": validation["rows"],
                "test_rows": test["rows"],
            },
            "validation": {"threshold": threshold["selected_threshold"]},
            "test": {
                "ordered_record_ids_verified": True,
                "bootstrap_indices": test["bootstrap_indices"],
                "threshold": test["threshold"],
                "point_metrics": test["point_metrics"],
                "confidence_intervals_95_percentile": test[
                    "confidence_intervals_95_percentile"
                ],
                "counts": test["counts"],
                "calibration_curve": test["calibration_curve"],
            },
            "artifact_sources": {
                "cv_results": str(cv_source.resolve()),
                "test_predictions": str(
                    (paths.stage("test_evaluation") / "test_predictions.csv").resolve()
                ),
            },
            "timing": {
                "training_seconds": full_train.get("training_seconds"),
                "test_inference_seconds": test.get("test_inference_seconds"),
            },
            "execution": {
                "fit_calls": 0,
                "prediction_calls": 0,
                "bootstrap_regenerated": False,
                "final_review": "PENDING",
            },
        }
        _write_json(temporary / "model_summary.json", summary)
        _write_cv(temporary / "cv_fold_metrics.csv", run_id, folds)
        _write_metrics(temporary / "rq1_metrics_long.csv", run_id, test)
        _write_confusion(temporary / "confusion_matrix.csv", run_id, test["counts"])
        _write_frame(
            temporary / "calibration_curve.csv",
            [
                {"model_key": "tabmmini", "run_id": run_id, **row}
                for row in test["calibration_curve"]
            ],
        )
        _write_json(
            temporary / "artifact_index.json",
            {
                "schema_version": REPORTING_SCHEMA_VERSION,
                "model_key": "tabmmini",
                "run_id": run_id,
                "artifacts": [
                    {"path": "../full_train/model/model_state.pt"},
                    {"path": "../test_evaluation/test_predictions.npz"},
                    {"path": "../test_evaluation/rq1_bootstrap_metrics.csv"},
                ],
            },
        )
        validate_reporting_directory(temporary)
        os.replace(temporary, destination)
    except Exception:
        # Preserve the working directory for diagnosis; never publish partial output.
        raise
    return validate_reporting_directory(destination)


def _write_cv(path: Path, run_id: str, folds: list[Mapping[str, Any]]) -> None:
    required = {"fold", "roc_auc", "fit_seconds"}
    if any(not required.issubset(row) for row in folds):
        raise TabMMiniReportingError(
            "Each TabMmini CV reporting row requires fold, roc_auc, and fit_seconds."
        )
    rows = [
        {
            "model_key": "tabmmini",
            "run_id": run_id,
            "feature_version": "feature_set_v1",
            "split_version": "split_v1",
            "fold": row["fold"],
            "roc_auc": row["roc_auc"],
            "fit_seconds": row["fit_seconds"],
            "predict_seconds": row.get("predict_seconds", 0.0),
            "warning_count": row.get("warning_count", 0),
        }
        for row in folds
    ]
    _write_frame(path, rows)


def _write_metrics(path: Path, run_id: str, test: Mapping[str, Any]) -> None:
    rows = []
    for name in METRIC_NAMES:
        canonical = "brier_score_approved" if name == "brier_approved" else name
        interval = test["confidence_intervals_95_percentile"][name]
        rows.append(
            {
                "model_key": "tabmmini",
                "run_id": run_id,
                "feature_version": "feature_set_v1",
                "split_version": "split_v1",
                "partition": "test",
                "metric": canonical,
                "estimate": test["point_metrics"][name],
                "ci_low": interval["lower_2_5"],
                "ci_high": interval["upper_97_5"],
                "unit": "coefficient" if canonical.startswith("calibration_") else "rate",
                "higher_is_better": canonical
                not in {
                    "brier_score_approved",
                    "calibration_intercept",
                    "calibration_slope",
                },
                "valid_replicates": interval["valid_replicates"],
            }
        )
    _write_frame(path, rows)


def _write_confusion(path: Path, run_id: str, counts: Mapping[str, Any]) -> None:
    _write_frame(
        path,
        [
            {"model_key": "tabmmini", "run_id": run_id, "actual_class": "approved", "predicted_class": "approved", "count": counts["tp_approved"]},
            {"model_key": "tabmmini", "run_id": run_id, "actual_class": "approved", "predicted_class": "denied", "count": counts["fn_approved"]},
            {"model_key": "tabmmini", "run_id": run_id, "actual_class": "denied", "predicted_class": "approved", "count": counts["fp_denied"]},
            {"model_key": "tabmmini", "run_id": run_id, "actual_class": "denied", "predicted_class": "denied", "count": counts["tn_denied"]},
        ],
    )


def _write_frame(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows, columns=REPORTING_CSV_COLUMNS[path.name]).to_csv(path, index=False)


def _require_run_id(run_id: str, *payloads: Mapping[str, Any]) -> None:
    if any(payload.get("run_id") != run_id for payload in payloads):
        raise TabMMiniReportingError("TabMmini reporting run_id lineage differs.")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TabMMiniReportingError(f"Cannot read TabMmini artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise TabMMiniReportingError(f"TabMmini artifact is not a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
