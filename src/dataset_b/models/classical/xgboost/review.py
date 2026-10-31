"""XGB-10: read-only final review of existing XGB-07/08/09 artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.dataset_b.models.classical.xgboost.test_evaluation import (
    BOOTSTRAP_RECOMPUTATION_ATOL,
    CALIBRATION_SLOPE_DECIMALS,
    DEFAULT_BOOTSTRAP_INDICES_PATH,
    DEFAULT_BOOTSTRAP_MANIFEST_PATH,
    validate_xgb_test_evaluation,
)
from src.dataset_b.models.classical.xgboost.training import validate_xgb_final_training
from src.dataset_b.models.classical.xgboost.validation import validate_xgb_validation_threshold
from src.dataset_b.research.rq1.reporting import validate_model_summary
from src.utils.path import PROJECT_ROOT


CONTRACT_ID = "XGB-10"
DEFAULT_XGB_ROOT = PROJECT_ROOT / "artifacts/dataset_b/xgboost/working"
HISTORICAL_XGB_BASELINE = PROJECT_ROOT / "artifacts/dataset_b/xgb_baseline_v1"
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class XGBFinalReviewError(RuntimeError):
    """Raised when existing XGBoost artifacts cannot pass read-only review."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XGBFinalReviewError(message)


def _validated_run_id(run_id: str) -> str:
    _require(isinstance(run_id, str) and bool(_RUN_ID.fullmatch(run_id)), "Invalid explicit XGBoost review run_id.")
    _require("legacy" not in run_id.lower(), "Legacy XGBoost run IDs are prohibited.")
    return run_id


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise XGBFinalReviewError(f"Cannot read review source: {path}") from exc
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _require_stage_identity(stage: Mapping[str, Any], run_id: str, stage_name: str) -> None:
    _require(stage.get("run_id") == run_id, f"XGB-10 found misaligned run IDs at {stage_name}.")
    _require(stage.get("feature_version") == "feature_set_v1", f"XGB-10 refuses legacy/misaligned feature artifacts at {stage_name}.")
    _require(stage.get("split_version") == "split_v1", f"XGB-10 found misaligned split lineage at {stage_name}.")
    _require(stage.get("model_key") == "xgb", f"XGB-10 found a mismatched model key at {stage_name}.")
    _require(stage.get("reporting_key") == "xgboost", f"XGB-10 found a mismatched reporting key at {stage_name}.")


def _require_bootstrap_review(metrics: Mapping[str, Any]) -> None:
    _require(metrics.get("bootstrap_contract", {}).get("regenerated") is False, "XGB-10 refuses regenerated bootstrap evidence.")
    _require(
        metrics.get("numerical_comparison_policy")
        == {
            "calibration_slope_decimals": CALIBRATION_SLOPE_DECIMALS,
            "other_metric_absolute_tolerance": BOOTSTRAP_RECOMPUTATION_ATOL,
            "confidence_interval_method": "percentile_2.5_97.5_linear",
        },
        "XGB-10 found missing bootstrap recomputation policy evidence.",
    )
    _require(
        metrics.get("execution")
        == {"model_fit_calls": 0, "preprocessing_fit_calls": 0, "predict_proba_calls": 1},
        "XGB-10 found an invalid Test execution boundary.",
    )


def review_xgb_final(
    run_id: str,
    *,
    run_dir: str | Path,
    bootstrap_indices_path: str | Path = DEFAULT_BOOTSTRAP_INDICES_PATH,
    bootstrap_manifest_path: str | Path = DEFAULT_BOOTSTRAP_MANIFEST_PATH,
) -> dict[str, Any]:
    """Validate existing artifacts only; never fit, infer, regenerate, or write."""
    run_id = _validated_run_id(run_id)
    root = Path(run_dir).resolve(strict=True)
    _require(root.name == run_id, "XGB-10 run_dir does not match the explicit run_id.")
    _require(root.parent == DEFAULT_XGB_ROOT.resolve(strict=False), "XGB-10 refuses artifacts outside the XGBoost working root.")
    _require("legacy" not in str(root).lower(), "XGB-10 refuses legacy artifact paths.")
    validation_dir = root / "validation"
    test_dir = root / "test"
    _require(validation_dir.is_dir() and test_dir.is_dir(), "XGB-10 refuses incomplete stage artifacts.")

    training = validate_xgb_final_training(root, run_id=run_id)
    validation = validate_xgb_validation_threshold(validation_dir, run_dir=root, run_id=run_id)
    test = validate_xgb_test_evaluation(
        test_dir,
        run_dir=root,
        run_id=run_id,
        bootstrap_indices_path=bootstrap_indices_path,
        bootstrap_manifest_path=bootstrap_manifest_path,
    )
    _require_stage_identity(training, run_id, "XGB-07")
    _require_stage_identity(validation, run_id, "XGB-08")
    _require_stage_identity(test, run_id, "XGB-09")
    _require(
        training.get("preprocessing_version")
        == validation.get("preprocessing_version")
        == test.get("preprocessing_version")
        == "dataset_b_preprocessing_v2",
        "XGB-10 found misaligned preprocessing lineage.",
    )
    library_versions = training.get("library_versions", {})
    required_libraries = {"python", "numpy", "pandas", "scipy", "scikit-learn", "xgboost", "joblib"}
    _require(
        isinstance(library_versions, dict)
        and required_libraries.issubset(library_versions)
        and all(isinstance(library_versions[name], str) and library_versions[name] for name in required_libraries),
        "XGB-10 runtime-version evidence is incomplete.",
    )

    schemas = training.get("schemas", {})
    _require(
        {key: schemas.get(key) for key in ("raw_feature_count", "semantic_feature_count", "encoded_feature_count")}
        == {"raw_feature_count": 15, "semantic_feature_count": 19, "encoded_feature_count": 62},
        "XGB-10 schema evidence differs from 15/19/62.",
    )
    _require(training.get("fresh_estimator_count") == 6, "XGB-10 fresh-estimator evidence differs.")
    _require(training.get("validation_accessed") is False and training.get("test_accessed") is False, "XGB-07 crossed a holdout boundary.")
    _require(validation.get("rows") == validation.get("unique_record_ids") == 43_995, "XGB-10 Validation population differs.")
    _require(validation.get("source_order_preserved") is True, "XGB-10 Validation order is not verified.")
    _require(test.get("rows") == test.get("unique_record_ids") == 43_996, "XGB-10 Test population differs.")
    _require(test.get("source_order_preserved") is True, "XGB-10 Test order is not verified.")

    threshold = _read_json(validation_dir / "threshold_selection.json")
    metrics = _read_json(test_dir / "rq1_test_metrics.json")
    _require(float(threshold["selected_threshold"]) == float(metrics["threshold"]) == float(test["threshold"]), "XGB-10 found a changed threshold.")
    _require_bootstrap_review(metrics)
    _require(metrics.get("lineage", {}).get("test_record_ids_verified") is True, "XGB-10 Test record order evidence is missing.")
    _require(isinstance(metrics.get("lineage", {}).get("bootstrap_indices"), str), "XGB-10 bootstrap lineage is missing.")

    cv = _read_json(root / "cv_results.json")
    full = _read_json(root / "full_train_result.json")
    _require(cv.get("fold_count") == cv.get("fresh_estimator_count") == 5, "XGB-10 CV estimator/fold evidence differs.")
    _require(cv.get("oof_rows") == 205_310 and cv.get("oof_coverage_min") == cv.get("oof_coverage_max") == 1, "XGB-10 OOF coverage evidence differs.")
    _require(full.get("retained_train_rows") == 205_310, "XGB-10 full-Train population differs.")
    _require(full.get("fresh_preprocessor") is True and full.get("fresh_estimator") is True, "XGB-10 full-Train state is not fresh.")
    _require(full.get("reload_prediction_rows_checked") == 4096 and full.get("reload_predictions_exact") is True, "XGB-10 reload evidence differs.")
    _require(full.get("scale_pos_weight") == full.get("n_denied") / full.get("n_approved"), "XGB-10 full-Train weighting evidence differs.")
    _require(metrics.get("scale_pos_weight") == full.get("scale_pos_weight"), "XGB-10 Test weighting lineage differs.")
    _require(HISTORICAL_XGB_BASELINE.is_dir(), "XGB-10 requires the historical XGBoost baseline directory to remain present.")

    point_metrics = dict(metrics["point_metrics"])
    intervals = dict(metrics["confidence_intervals_95_percentile"])
    point_metrics["brier_score_approved"] = point_metrics.pop("brier_approved")
    intervals["brier_score_approved"] = intervals.pop("brier_approved")
    summary = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "status": "PASS",
        "reporting_status": "COMPLETE",
        "run_id": run_id,
        "model_key": "xgboost",
        "protocol_model_key": "xgb",
        "feature_version": "feature_set_v1",
        "split_version": "split_v1",
        "preprocessing_version": training["preprocessing_version"],
        "library_versions": dict(library_versions),
        "stages": {"xgb_07": "PASS", "xgb_08": "PASS", "xgb_09": "PASS", "xgb_10": "PASS"},
        "data_boundaries": {
            "train_rows": 205_310,
            "validation_rows": 43_995,
            "test_rows": 43_996,
            "validation_accessed": True,
            "test_accessed": True,
        },
        "feature_counts": {"raw": 15, "semantic": 19, "model_input": 62},
        "cv": {
            "fold_count": cv["fold_count"],
            "mean_roc_auc": cv["mean_fold_roc_auc"],
            "std_roc_auc": cv["std_fold_roc_auc"],
            "oof_roc_auc": cv["oof_roc_auc"],
            "oof_rows": cv["oof_rows"],
            "warning_count": sum(int(fold.get("warning_count", 0)) for fold in cv["folds"]),
        },
        "full_train": {
            "fit_rows": full["retained_train_rows"],
            "fit_seconds": full["fit_seconds"],
            "warning_count": full.get("warning_count", 0),
            "scale_pos_weight": full["scale_pos_weight"],
            "reload_predictions_exact": full["reload_predictions_exact"],
        },
        "validation": {
            "rows": validation["rows"],
            "unique_record_ids": validation["unique_record_ids"],
            "source_order_preserved": validation["source_order_preserved"],
            "threshold": threshold["selected_threshold"],
            **threshold["selected_metrics"],
        },
        "test": {
            "rows": test["rows"],
            "unique_record_ids": test["unique_record_ids"],
            "source_order_preserved": test["source_order_preserved"],
            "threshold": metrics["threshold"],
            "point_metrics": point_metrics,
            "confidence_intervals_95_percentile": intervals,
            "counts": metrics["counts"],
            "calibration_curve": metrics["calibration_curve"],
            "ordered_record_ids_verified": metrics["lineage"]["test_record_ids_verified"],
            "bootstrap_indices": metrics["lineage"]["bootstrap_indices"],
            "bootstrap_recomputation_verified": True,
        },
        "selected_threshold": metrics["threshold"],
        "point_metrics": point_metrics,
        "confidence_intervals_95_percentile": intervals,
        "artifact_sources": {
            "training_manifest": str(root / "manifest.json"),
            "cv_results": str(root / "cv_results.json"),
            "full_train_result": str(root / "full_train_result.json"),
            "validation_manifest": str(validation_dir / "manifest.json"),
            "test_manifest": str(test_dir / "run_manifest.json"),
            "test_predictions": str(test_dir / "test_predictions.csv"),
        },
        "execution": {
            "fit_calls": 0,
            "predict_proba_calls": 0,
            "bootstrap_regenerated": False,
            "bootstrap_recomputation_verified": True,
            "threshold_changed": False,
        },
    }
    validate_model_summary(summary, require_complete=True)
    return summary


def parse_args(argv: Sequence[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(description="Read-only XGB-10 final review.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(review_xgb_final(args.run_id, run_dir=args.run_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
