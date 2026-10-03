"""Deterministic PMLP-F05 reporting from completed source artifacts."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping

from src.dataset_b.evaluation.plain_mlp_test_evaluation import METRIC_NAMES


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


def generate_plain_mlp_reporting(
    output_dir: str | Path,
    *,
    run_id: str,
    cv_result: Mapping[str, Any],
    test_metrics: Mapping[str, Any],
    artifact_paths: Mapping[str, str | Path],
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
            "run_id": run_id,
            "model_version": test_metrics.get("model_version"),
            "threshold": test_metrics.get("threshold"),
            "test_rows": test_metrics.get("rows"),
            "metric_count": len(METRIC_NAMES),
            "bootstrap_replicates": 1000,
            "execution": {
                "model_fit_calls": 0,
                "predict_proba_calls": 0,
            },
        }
        _write_json(working / "model_summary.json", summary)

        fold_rows: list[dict[str, Any]] = []
        for fold in folds:
            timing = fold.get("timing_seconds", {})
            final = fold.get("final_estimator", {})
            fold_rows.append(
                {
                    "fold": fold.get("fold"),
                    "fit_rows": fold.get("fit_rows"),
                    "fold_validation_rows": fold.get("fold_validation_rows"),
                    "best_epoch": final.get("best_epoch"),
                    "pos_weight": final.get("pos_weight"),
                    "selector_seconds": timing.get("selector"),
                    "final_fit_seconds": timing.get("final_fit"),
                    "outer_prediction_seconds": timing.get("outer_prediction"),
                    "warnings": json.dumps(fold.get("warnings", []), ensure_ascii=False),
                }
            )
        _write_csv(
            working / "cv_fold_metrics.csv",
            ("fold", "fit_rows", "fold_validation_rows", "best_epoch", "pos_weight",
             "selector_seconds", "final_fit_seconds", "outer_prediction_seconds", "warnings"),
            fold_rows,
        )

        metric_rows = [
            {
                "metric": name,
                "point_estimate": point[name],
                "ci_lower_2_5": intervals[name]["lower_2_5"],
                "ci_upper_97_5": intervals[name]["upper_97_5"],
                "valid_replicates": intervals[name]["valid_replicates"],
                "invalid_replicates": intervals[name]["invalid_replicates"],
            }
            for name in METRIC_NAMES
        ]
        _write_csv(
            working / "rq1_metrics_long.csv",
            ("metric", "point_estimate", "ci_lower_2_5", "ci_upper_97_5",
             "valid_replicates", "invalid_replicates"),
            metric_rows,
        )
        confusion_rows = [
            {"actual": "approved", "predicted": "approved", "count": counts["tp_approved"]},
            {"actual": "approved", "predicted": "denied", "count": counts["fn_approved"]},
            {"actual": "denied", "predicted": "denied", "count": counts["tn_denied"]},
            {"actual": "denied", "predicted": "approved", "count": counts["fp_denied"]},
        ]
        _write_csv(working / "confusion_matrix.csv", ("actual", "predicted", "count"), confusion_rows)
        _write_csv(
            working / "calibration_curve.csv",
            ("bin", "lower_edge", "upper_edge", "count", "mean_p_approved", "observed_approval_rate"),
            [dict(row) for row in curve],
        )
        index = {
            "schema_version": 1,
            "contract_id": "PMLP-F05-ARTIFACT-INDEX",
            "run_id": run_id,
            "artifacts": [
                {"name": name, "path": str(path)}
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
    with (directory / "cv_fold_metrics.csv").open(encoding="utf-8", newline="") as handle:
        folds = list(csv.DictReader(handle))
    with (directory / "rq1_metrics_long.csv").open(encoding="utf-8", newline="") as handle:
        metrics = list(csv.DictReader(handle))
    with (directory / "confusion_matrix.csv").open(encoding="utf-8", newline="") as handle:
        confusion = list(csv.DictReader(handle))
    if (
        summary.get("contract_id") != "PMLP-F05-REPORTING"
        or summary.get("status") != "PASS"
        or summary.get("run_id") != index.get("run_id")
        or (expected_run_id is not None and summary.get("run_id") != expected_run_id)
        or len(folds) != 5
        or [row.get("metric") for row in metrics] != list(METRIC_NAMES)
        or len(confusion) != 4
        or "sha256" in json.dumps(index).lower()
    ):
        raise PlainMLPReportingError("Reporting schema or linkage is invalid.")
    return summary
