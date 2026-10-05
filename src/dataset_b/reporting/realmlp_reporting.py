"""Common-schema reporting package for a completed final RealMLP run."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.dataset_b.evaluation.plain_mlp_test_evaluation import METRIC_NAMES
from src.dataset_b.evaluation.realmlp_final_evaluation import (
    validate_realmlp_test_evaluation,
    validate_realmlp_threshold,
    validate_realmlp_validation,
)
from src.dataset_b.reporting.model_reporting import (
    REPORTING_CSV_COLUMNS,
    REPORTING_SCHEMA_VERSION,
    validate_reporting_directory,
)
from src.dataset_b.runtime.realmlp_final_contract import RealMLPFinalPaths, read_json
from src.dataset_b.runtime.realmlp_final_training import (
    validate_realmlp_cv,
    validate_realmlp_full_train,
)
from src.utils.path import PROJECT_ROOT


DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "dataset_b" / "realmlp"


def write_realmlp_reporting(
    run_id: str, *, artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT
) -> dict[str, Any]:
    paths = RealMLPFinalPaths.for_run(run_id, artifact_root=artifact_root)
    cv = validate_realmlp_cv(paths)
    full_train = validate_realmlp_full_train(paths)
    validation = validate_realmlp_validation(paths)
    threshold = validate_realmlp_threshold(paths)
    test = validate_realmlp_test_evaluation(paths)
    destination = paths.stage("reporting")
    temporary = destination.with_name("reporting.tmp")
    if destination.exists() or temporary.exists():
        raise FileExistsError("RealMLP reporting output already exists.")
    temporary.mkdir(parents=True)
    test_metrics = read_json(paths.stage("test_evaluation") / "rq1_test_metrics.json")
    summary = {
        "schema_version": REPORTING_SCHEMA_VERSION,
        "reporting_status": "COMPLETE",
        "model_key": "realmlp",
        "run_id": run_id,
        "feature_version": "feature_set_v1",
        "split_version": "split_v1",
        "preprocessing_version": "dataset_b_neural_preprocessing_v1",
        "data_boundaries": {
            "train_rows": full_train["train_rows"],
            "validation_rows": validation["rows"],
            "test_rows": test["rows"],
        },
        "validation": {"threshold": threshold["selected_threshold"]},
        "test": {
            "ordered_record_ids_verified": True,
            "bootstrap_indices": test_metrics["bootstrap_indices"],
            "threshold": test_metrics["threshold"],
            "point_metrics": test_metrics["point_metrics"],
            "confidence_intervals_95_percentile": test_metrics[
                "confidence_intervals_95_percentile"
            ],
            "counts": test_metrics["counts"],
            "calibration_curve": test_metrics["calibration_curve"],
        },
        "artifact_sources": {
            "cv_results": str((paths.stage("cv", final=True) / "cv_results.json").resolve()),
            "test_predictions": str(
                (paths.stage("test_evaluation", final=True) / "test_predictions.csv").resolve()
            ),
        },
        "execution": {
            "fit_calls": 0,
            "predict_proba_calls": 0,
            "bootstrap_regenerated": False,
            "final_review": "PENDING",
        },
    }
    _write_json(temporary / "model_summary.json", summary)
    cv_rows = [
        {
            "model_key": "realmlp",
            "run_id": run_id,
            "feature_version": "feature_set_v1",
            "split_version": "split_v1",
            "fold": row["fold"],
            "roc_auc": row["roc_auc"],
            "fit_seconds": row["fit_seconds"],
            "predict_seconds": 0.0,
            "warning_count": 0,
        }
        for row in cv["folds"]
    ]
    _write_frame(temporary / "cv_fold_metrics.csv", cv_rows)
    metric_rows = []
    for name in METRIC_NAMES:
        canonical = "brier_score_approved" if name == "brier_approved" else name
        interval = test_metrics["confidence_intervals_95_percentile"][name]
        metric_rows.append(
            {
                "model_key": "realmlp",
                "run_id": run_id,
                "feature_version": "feature_set_v1",
                "split_version": "split_v1",
                "partition": "test",
                "metric": canonical,
                "estimate": test_metrics["point_metrics"][name],
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
    _write_frame(temporary / "rq1_metrics_long.csv", metric_rows)
    counts = test_metrics["counts"]
    _write_frame(
        temporary / "confusion_matrix.csv",
        [
            {"model_key": "realmlp", "run_id": run_id, "actual_class": "approved", "predicted_class": "approved", "count": counts["tp_approved"]},
            {"model_key": "realmlp", "run_id": run_id, "actual_class": "approved", "predicted_class": "denied", "count": counts["fn_approved"]},
            {"model_key": "realmlp", "run_id": run_id, "actual_class": "denied", "predicted_class": "approved", "count": counts["fp_denied"]},
            {"model_key": "realmlp", "run_id": run_id, "actual_class": "denied", "predicted_class": "denied", "count": counts["tn_denied"]},
        ],
    )
    _write_frame(
        temporary / "calibration_curve.csv",
        [
            {"model_key": "realmlp", "run_id": run_id, **row}
            for row in test_metrics["calibration_curve"]
        ],
    )
    _write_json(
        temporary / "artifact_index.json",
        {
            "schema_version": REPORTING_SCHEMA_VERSION,
            "model_key": "realmlp",
            "run_id": run_id,
            "artifacts": [
                {"path": "../full_train/fitted_model.joblib"},
                {"path": "../test_evaluation/test_predictions.csv"},
                {"path": "../test_evaluation/rq1_bootstrap_metrics.csv"},
            ],
        },
    )
    validate_reporting_directory(temporary)
    os.replace(temporary, destination)
    return validate_reporting_directory(destination)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_frame(path: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows, columns=REPORTING_CSV_COLUMNS[path.name])
    frame.to_csv(path, index=False)
