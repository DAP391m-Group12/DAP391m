"""Read-only PMLP-F05 final review using direct numerical recomputation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.dataset_b.evaluation.plain_mlp_test_evaluation import (
    BOOTSTRAP_COLUMNS,
    METRIC_NAMES,
    _bootstrap_metric_matches,
    _metrics,
)
from src.dataset_b.reporting.plain_mlp_reporting import validate_plain_mlp_reporting


class PlainMLPFinalReviewError(RuntimeError):
    """Raised when direct F05 review finds an inconsistency."""


def review_plain_mlp_f05(
    *,
    run_id: str,
    validation_result_path: str | Path,
    threshold_selection_path: str | Path,
    test_dir: str | Path,
    bootstrap_indices_path: str | Path,
    reporting_dir: str | Path,
) -> dict[str, Any]:
    """Review stored content without loading a model, fitting, or inference."""
    validation = json.loads(Path(validation_result_path).read_text(encoding="utf-8"))
    threshold = json.loads(Path(threshold_selection_path).read_text(encoding="utf-8"))
    test_path = Path(test_dir)
    result = json.loads((test_path / "rq1_test_metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(test_path / "test_predictions.csv", dtype={"record_id": str})
    stored_bootstrap = pd.read_csv(test_path / "rq1_bootstrap_metrics.csv")
    with np.load(bootstrap_indices_path, allow_pickle=False) as payload:
        indices = payload["indices"]
    if {validation.get("run_id"), threshold.get("run_id"), result.get("run_id")} != {run_id}:
        raise PlainMLPFinalReviewError("Exact run_id linkage failed.")
    if result.get("execution") != {
        "model_fit_calls": 0,
        "preprocessing_fit_calls": 0,
        "preprocessing_transform_calls": 1,
        "predict_proba_calls": 1,
    }:
        raise PlainMLPFinalReviewError("Test execution counters are invalid.")
    if tuple(stored_bootstrap.columns) != BOOTSTRAP_COLUMNS or len(indices) != len(stored_bootstrap):
        raise PlainMLPFinalReviewError("Bootstrap content schema is invalid.")
    labels = predictions.y_approved.to_numpy(dtype=np.int8)
    probabilities = predictions.p_approved.to_numpy(dtype=np.float64)
    point, invalid, counts = _metrics(labels, probabilities, float(result["threshold"]))
    if invalid or counts != result.get("counts"):
        raise PlainMLPFinalReviewError("Point metric inputs or confusion counts differ.")
    for name in METRIC_NAMES:
        if not bool(_bootstrap_metric_matches(
            name, np.asarray([result["point_metrics"][name]]), np.asarray([point[name]])
        ).all()):
            raise PlainMLPFinalReviewError(f"Point metric differs: {name}")
    recomputed_rows: list[dict[str, Any]] = []
    for replicate_id, positions in enumerate(indices):
        values, problems, _ = _metrics(labels[positions], probabilities[positions], float(result["threshold"]))
        recomputed_rows.append({
            "replicate_id": replicate_id,
            **values,
            "invalid_metrics": json.dumps(problems, ensure_ascii=False, sort_keys=True),
        })
    recomputed = pd.DataFrame(recomputed_rows, columns=BOOTSTRAP_COLUMNS)
    if not np.array_equal(stored_bootstrap.replicate_id, recomputed.replicate_id):
        raise PlainMLPFinalReviewError("Bootstrap replicate identifiers differ.")
    if not stored_bootstrap.invalid_metrics.fillna("").astype(str).equals(
        recomputed.invalid_metrics.fillna("").astype(str)
    ):
        raise PlainMLPFinalReviewError("Bootstrap invalid payloads differ.")
    for name in METRIC_NAMES:
        if not bool(_bootstrap_metric_matches(name, stored_bootstrap[name], recomputed[name]).all()):
            raise PlainMLPFinalReviewError(f"Bootstrap metric differs: {name}")
        finite = recomputed[name].to_numpy(dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        expected = result["confidence_intervals_95_percentile"][name]
        low = float(np.quantile(finite, 0.025, method="linear")) if len(finite) else None
        high = float(np.quantile(finite, 0.975, method="linear")) if len(finite) else None
        if (
            expected["valid_replicates"] != len(finite)
            or expected["invalid_replicates"] != len(recomputed) - len(finite)
            or (low is not None and not np.isclose(low, expected["lower_2_5"], rtol=0, atol=1e-14))
            or (high is not None and not np.isclose(high, expected["upper_97_5"], rtol=0, atol=1e-14))
        ):
            raise PlainMLPFinalReviewError(f"Bootstrap interval differs: {name}")
    validate_plain_mlp_reporting(reporting_dir, expected_run_id=run_id)
    report_path = Path(reporting_dir)
    metric_report = pd.read_csv(report_path / "rq1_metrics_long.csv")
    confusion_report = pd.read_csv(report_path / "confusion_matrix.csv")
    calibration_report = pd.read_csv(report_path / "calibration_curve.csv")
    canonical_names = sorted(
        "brier_score_approved" if name == "brier_approved" else name
        for name in METRIC_NAMES
    )
    if metric_report.metric.tolist() != canonical_names:
        raise PlainMLPFinalReviewError("Reported metric order differs.")
    for row in metric_report.itertuples(index=False):
        source_name = "brier_approved" if row.metric == "brier_score_approved" else row.metric
        expected = result["confidence_intervals_95_percentile"][source_name]
        if (
            not np.isclose(row.estimate, point[source_name], rtol=0, atol=1e-14)
            or not np.isclose(row.ci_low, expected["lower_2_5"], rtol=0, atol=1e-14)
            or not np.isclose(row.ci_high, expected["upper_97_5"], rtol=0, atol=1e-14)
            or row.valid_replicates != expected["valid_replicates"]
        ):
            raise PlainMLPFinalReviewError(f"Reported metric differs: {row.metric}")
    expected_confusion = [
        ("approved", "approved", counts["tp_approved"]),
        ("approved", "denied", counts["fn_approved"]),
        ("denied", "approved", counts["fp_denied"]),
        ("denied", "denied", counts["tn_denied"]),
    ]
    observed_confusion = list(
        confusion_report[["actual_class", "predicted_class", "count"]].itertuples(index=False, name=None)
    )
    if observed_confusion != expected_confusion:
        raise PlainMLPFinalReviewError("Reported confusion matrix differs.")
    curve_columns = [
        "bin", "lower_edge", "upper_edge", "count",
        "mean_p_approved", "observed_approval_rate",
    ]
    expected_curve = pd.DataFrame(result["calibration_curve"])[curve_columns]
    observed_curve = calibration_report[curve_columns]
    if list(expected_curve.columns) != list(observed_curve.columns):
        raise PlainMLPFinalReviewError("Reported calibration schema differs.")
    for column in expected_curve.columns:
        left, right = observed_curve[column], expected_curve[column]
        if left.dtype.kind in "if" and right.dtype.kind in "if":
            if not np.allclose(left, right, rtol=0, atol=1e-14):
                raise PlainMLPFinalReviewError(f"Reported calibration differs: {column}")
        elif left.astype(str).tolist() != right.astype(str).tolist():
            raise PlainMLPFinalReviewError(f"Reported calibration differs: {column}")
    return {
        "schema_version": 1,
        "contract_id": "PMLP-F05-FINAL-REVIEW",
        "status": "PASS",
        "run_id": run_id,
        "acceptance_policy": "direct_content_and_numerical_recomputation",
        "execution": {"model_fit_calls": 0, "predict_proba_calls": 0},
        "bootstrap_indices_regenerated": False,
        "metric_count": len(METRIC_NAMES),
        "bootstrap_replicates": len(recomputed),
    }
