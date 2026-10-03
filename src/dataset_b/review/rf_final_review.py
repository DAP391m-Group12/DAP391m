"""RF-10: read-only final review of existing RF-07/08/09 artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from src.dataset_b.evaluation.rf_test_evaluation import (
    DEFAULT_BOOTSTRAP_INDICES_PATH,
    DEFAULT_BOOTSTRAP_MANIFEST_PATH,
    validate_rf_test_evaluation,
)
from src.dataset_b.evaluation.rf_validation_threshold import validate_rf_validation_threshold
from src.dataset_b.training.rf_final_training import _validated_run_id, validate_rf_final_training
from src.dataset_b.reporting.model_reporting import validate_model_summary
from src.utils.path import PROJECT_ROOT


CONTRACT_ID = "RF-10"
DEFAULT_RF_ROOT = PROJECT_ROOT / "artifacts/dataset_b/rf/working"
DEFAULT_RF_FINAL_ROOT = PROJECT_ROOT / "artifacts/dataset_b/rf/final"


class RFFinalReviewError(RuntimeError):
    """Raised when existing RF artifacts cannot pass read-only review."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RFFinalReviewError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RFFinalReviewError(f"Cannot read review source: {path}") from exc
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def review_rf_final(
    run_id: str,
    *,
    run_dir: str | Path,
    bootstrap_indices_path: str | Path = DEFAULT_BOOTSTRAP_INDICES_PATH,
    bootstrap_manifest_path: str | Path = DEFAULT_BOOTSTRAP_MANIFEST_PATH,
) -> dict[str, Any]:
    """Validate existing artifacts only; never fit, infer, regenerate, or write."""
    run_id = _validated_run_id(run_id)
    root = Path(run_dir).resolve(strict=True)
    _require(root.name == run_id, "RF-10 run_dir does not match the explicit run_id.")
    allowed_roots = {
        DEFAULT_RF_ROOT.resolve(strict=False),
        DEFAULT_RF_FINAL_ROOT.resolve(strict=False),
    }
    _require(root.parent in allowed_roots, "RF-10 refuses artifacts outside the RF working/final roots.")
    _require("legacy" not in str(root).lower(), "RF-10 refuses legacy artifact paths.")
    validation_dir = root / "validation"; test_dir = root / "test"
    _require(validation_dir.is_dir() and test_dir.is_dir(), "RF-10 refuses incomplete stage artifacts.")
    if root.parent == DEFAULT_RF_FINAL_ROOT.resolve(strict=False):
        local_indices = root / "shared_bootstrap/rq1_test_bootstrap_indices.npz"
        local_manifest = root / "shared_bootstrap/rq1_test_bootstrap_manifest.json"
        if Path(bootstrap_indices_path) == DEFAULT_BOOTSTRAP_INDICES_PATH and local_indices.is_file():
            bootstrap_indices_path = local_indices
        if Path(bootstrap_manifest_path) == DEFAULT_BOOTSTRAP_MANIFEST_PATH and local_manifest.is_file():
            bootstrap_manifest_path = local_manifest
    training = validate_rf_final_training(root, run_id=run_id)
    validation = validate_rf_validation_threshold(validation_dir, run_dir=root, run_id=run_id)
    test = validate_rf_test_evaluation(
        test_dir, run_dir=root, run_id=run_id,
        bootstrap_indices_path=bootstrap_indices_path,
        bootstrap_manifest_path=bootstrap_manifest_path,
    )
    _require(training.get("run_id") == validation.get("run_id") == test.get("run_id") == run_id, "RF-10 found misaligned run IDs.")
    _require(training.get("feature_version") == validation.get("feature_version") == test.get("feature_version") == "feature_set_v1", "RF-10 refuses legacy/misaligned feature artifacts.")
    _require(training.get("split_version") == validation.get("split_version") == test.get("split_version") == "split_v1", "RF-10 found misaligned split lineage.")
    threshold = _read_json(validation_dir / "threshold_selection.json")
    metrics = _read_json(test_dir / "rq1_test_metrics.json")
    _require(float(threshold["selected_threshold"]) == float(metrics["threshold"]) == float(test["threshold"]), "RF-10 found a changed threshold.")
    _require(metrics.get("bootstrap_contract", {}).get("regenerated") is False, "RF-10 refuses regenerated bootstrap evidence.")
    _require(metrics.get("execution") == {"model_fit_calls": 0, "preprocessing_fit_calls": 0, "predict_proba_calls": 1}, "RF-10 found an invalid Test execution boundary.")
    cv = _read_json(root / "cv_results.json"); full = _read_json(root / "full_train_result.json")
    point_metrics = dict(metrics["point_metrics"]); intervals = dict(metrics["confidence_intervals_95_percentile"])
    point_metrics["brier_score_approved"] = point_metrics.pop("brier_approved")
    intervals["brier_score_approved"] = intervals.pop("brier_approved")
    summary = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "status": "PASS",
        "reporting_status": "COMPLETE",
        "run_id": run_id,
        "model_key": "rf",
        "feature_version": "feature_set_v1",
        "split_version": "split_v1",
        "preprocessing_version": training["preprocessing_version"],
        "stages": {"rf_07": "PASS", "rf_08": "PASS", "rf_09": "PASS", "rf_10": "PASS"},
        "data_boundaries": {"train_rows": 205_310, "validation_rows": 43_995, "test_rows": 43_996, "validation_accessed": True, "test_accessed": True},
        "feature_counts": {"raw": 15, "semantic": 19, "model_input": 62},
        "cv": {"fold_count": cv["fold_count"], "mean_roc_auc": cv["mean_fold_roc_auc"], "std_roc_auc": cv["std_fold_roc_auc"], "oof_roc_auc": cv["oof_roc_auc"], "oof_rows": cv["oof_rows"], "warning_count": sum(int(fold.get("warning_count", 0)) for fold in cv["folds"])},
        "full_train": {"fit_rows": full["retained_train_rows"], "fit_seconds": full["fit_seconds"], "warning_count": full.get("warning_count", 0)},
        "validation": {"rows": validation["rows"], "unique_record_ids": validation["unique_record_ids"], "source_order_preserved": validation["source_order_preserved"], "threshold": threshold["selected_threshold"], **threshold["selected_metrics"]},
        "test": {"rows": test["rows"], "unique_record_ids": test["unique_record_ids"], "source_order_preserved": test["source_order_preserved"], "threshold": metrics["threshold"], "point_metrics": point_metrics, "confidence_intervals_95_percentile": intervals, "counts": metrics["counts"], "calibration_curve": metrics["calibration_curve"], "ordered_record_ids_verified": metrics["lineage"]["test_record_ids_verified"], "bootstrap_indices": metrics["lineage"]["bootstrap_indices"]},
        "selected_threshold": metrics["threshold"],
        "point_metrics": point_metrics,
        "confidence_intervals_95_percentile": intervals,
        "artifact_sources": {"training_manifest": str(root / "manifest.json"), "cv_results": str(root / "cv_results.json"), "full_train_result": str(root / "full_train_result.json"), "validation_manifest": str(validation_dir / "manifest.json"), "test_manifest": str(test_dir / "run_manifest.json"), "test_predictions": str(test_dir / "test_predictions.csv")},
        "execution": {"fit_calls": 0, "predict_proba_calls": 0, "bootstrap_regenerated": False, "threshold_changed": False},
    }
    validate_model_summary(summary, require_complete=True)
    return summary


def parse_args(argv: Sequence[str] | None = None):
    import argparse
    parser = argparse.ArgumentParser(description="Read-only RF-10 final review.")
    parser.add_argument("--run-id", required=True); parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(review_rf_final(args.run_id, run_dir=args.run_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
