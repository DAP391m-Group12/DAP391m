"""XGB-10 refusal and summary tests for read-only final review."""

from __future__ import annotations

import inspect
import json

import pytest

from src.dataset_b.models.classical.xgboost import review
from src.dataset_b.review import xgb_final_review as compatibility


def test_legacy_review_module_delegates_to_canonical():
    assert compatibility.XGBFinalReviewError is review.XGBFinalReviewError
    assert compatibility.review_xgb_final is review.review_xgb_final


def test_review_refuses_incomplete_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "DEFAULT_XGB_ROOT", tmp_path)
    run_id = "xgb_run"
    root = tmp_path / run_id
    root.mkdir()
    with pytest.raises(review.XGBFinalReviewError, match="incomplete"):
        review.review_xgb_final(run_id, run_dir=root)


def test_review_refuses_legacy_run_id(tmp_path):
    root = tmp_path / "legacy_xgb"
    root.mkdir()
    with pytest.raises(review.XGBFinalReviewError, match="Legacy"):
        review.review_xgb_final("legacy_xgb", run_dir=root)


def test_review_refuses_misaligned_stage_lineage(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "DEFAULT_XGB_ROOT", tmp_path)
    run_id = "xgb_run"
    root = tmp_path / run_id
    (root / "validation").mkdir(parents=True)
    (root / "test").mkdir()
    base = {
        "run_id": run_id,
        "model_key": "xgb",
        "reporting_key": "xgboost",
        "feature_version": "feature_set_v1",
        "split_version": "split_v1",
        "preprocessing_version": "dataset_b_preprocessing_v2",
    }
    monkeypatch.setattr(review, "validate_xgb_final_training", lambda *_a, **_k: base)
    monkeypatch.setattr(review, "validate_xgb_validation_threshold", lambda *_a, **_k: {**base, "run_id": "other"})
    monkeypatch.setattr(review, "validate_xgb_test_evaluation", lambda *_a, **_k: base)
    with pytest.raises(review.XGBFinalReviewError, match="misaligned run IDs"):
        review.review_xgb_final(run_id, run_dir=root)


def _complete_review_sources(tmp_path, monkeypatch, *, include_bootstrap_policy=True):
    run_id = "xgb_run"
    root = tmp_path / run_id
    (root / "validation").mkdir(parents=True)
    (root / "test").mkdir()
    monkeypatch.setattr(review, "DEFAULT_XGB_ROOT", tmp_path)
    historical = tmp_path / "historical"
    historical.mkdir()
    monkeypatch.setattr(review, "HISTORICAL_XGB_BASELINE", historical)
    training = {
        "run_id": run_id,
        "model_key": "xgb",
        "reporting_key": "xgboost",
        "feature_version": "feature_set_v1",
        "split_version": "split_v1",
        "preprocessing_version": "dataset_b_preprocessing_v2",
        "library_versions": {
            "python": "3.12",
            "numpy": "2.0",
            "pandas": "2.2",
            "scipy": "1.14",
            "scikit-learn": "1.5",
            "xgboost": "2.1",
            "joblib": "1.4",
        },
        "schemas": {"raw_feature_count": 15, "semantic_feature_count": 19, "encoded_feature_count": 62},
        "fresh_estimator_count": 6,
        "validation_accessed": False,
        "test_accessed": False,
    }
    validation = {
        "run_id": run_id,
        "model_key": "xgb",
        "reporting_key": "xgboost",
        "feature_version": "feature_set_v1",
        "split_version": "split_v1",
        "preprocessing_version": "dataset_b_preprocessing_v2",
        "rows": 43_995,
        "unique_record_ids": 43_995,
        "source_order_preserved": True,
    }
    test = {
        "run_id": run_id,
        "model_key": "xgb",
        "reporting_key": "xgboost",
        "feature_version": "feature_set_v1",
        "split_version": "split_v1",
        "preprocessing_version": "dataset_b_preprocessing_v2",
        "rows": 43_996,
        "unique_record_ids": 43_996,
        "source_order_preserved": True,
        "threshold": 0.5,
    }
    monkeypatch.setattr(review, "validate_xgb_final_training", lambda *_a, **_k: training)
    monkeypatch.setattr(review, "validate_xgb_validation_threshold", lambda *_a, **_k: validation)
    monkeypatch.setattr(review, "validate_xgb_test_evaluation", lambda *_a, **_k: test)
    for relative in ("manifest.json", "validation/manifest.json", "test/run_manifest.json", "test/test_predictions.csv"):
        (root / relative).write_text("{}\n", encoding="utf-8")
    (root / "cv_results.json").write_text(
        json.dumps(
            {
                "fold_count": 5,
                "fresh_estimator_count": 5,
                "folds": [{"warning_count": 0} for _ in range(5)],
                "mean_fold_roc_auc": 0.8,
                "std_fold_roc_auc": 0.01,
                "oof_roc_auc": 0.8,
                "oof_rows": 205_310,
                "oof_coverage_min": 1,
                "oof_coverage_max": 1,
            }
        ),
        encoding="utf-8",
    )
    (root / "full_train_result.json").write_text(
        json.dumps(
            {
                "retained_train_rows": 205_310,
                "fit_seconds": 1.0,
                "warning_count": 0,
                "fresh_preprocessor": True,
                "fresh_estimator": True,
                "reload_prediction_rows_checked": 4096,
                "reload_predictions_exact": True,
                "n_denied": 1,
                "n_approved": 2,
                "scale_pos_weight": 0.5,
            }
        ),
        encoding="utf-8",
    )
    (root / "validation/threshold_selection.json").write_text(
        json.dumps(
            {
                "selected_threshold": 0.5,
                "selected_metrics": {"balanced_accuracy": 0.7, "recall_approved": 0.8, "recall_denied": 0.6},
            }
        ),
        encoding="utf-8",
    )
    names = (
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
    metrics = {
        "threshold": 0.5,
        "scale_pos_weight": 0.5,
        "point_metrics": {name: 0.5 for name in names},
        "confidence_intervals_95_percentile": {
            name: {"lower_2_5": 0.4, "upper_97_5": 0.6, "valid_replicates": 1000, "invalid_replicates": 0}
            for name in names
        },
        "counts": {},
        "calibration_curve": [],
        "lineage": {"test_record_ids_verified": True, "bootstrap_indices": "bootstrap-indices.npz"},
        "bootstrap_contract": {"regenerated": False},
        "execution": {"model_fit_calls": 0, "preprocessing_fit_calls": 0, "predict_proba_calls": 1},
    }
    if include_bootstrap_policy:
        metrics["numerical_comparison_policy"] = {
            "calibration_slope_decimals": review.CALIBRATION_SLOPE_DECIMALS,
            "other_metric_absolute_tolerance": review.BOOTSTRAP_RECOMPUTATION_ATOL,
            "confidence_interval_method": "percentile_2.5_97.5_linear",
        }
    (root / "test/rq1_test_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return run_id, root


def test_review_refuses_missing_bootstrap_recomputation_policy(tmp_path, monkeypatch):
    run_id, root = _complete_review_sources(tmp_path, monkeypatch, include_bootstrap_policy=False)
    with pytest.raises(review.XGBFinalReviewError, match="missing bootstrap recomputation"):
        review.review_xgb_final(run_id, run_dir=root)


def test_review_refuses_incomplete_runtime_versions(tmp_path, monkeypatch):
    run_id, root = _complete_review_sources(tmp_path, monkeypatch)
    monkeypatch.setattr(
        review,
        "validate_xgb_final_training",
        lambda *_a, **_k: {
            "run_id": run_id,
            "model_key": "xgb",
            "reporting_key": "xgboost",
            "feature_version": "feature_set_v1",
            "split_version": "split_v1",
            "preprocessing_version": "dataset_b_preprocessing_v2",
            "library_versions": {"xgboost": "2.1"},
        },
    )
    with pytest.raises(review.XGBFinalReviewError, match="runtime-version"):
        review.review_xgb_final(run_id, run_dir=root)


def test_review_complete_pass_returns_common_xgboost_summary(tmp_path, monkeypatch):
    run_id, root = _complete_review_sources(tmp_path, monkeypatch)
    result = review.review_xgb_final(run_id, run_dir=root)
    assert result["reporting_status"] == "COMPLETE"
    assert result["model_key"] == "xgboost"
    assert result["protocol_model_key"] == "xgb"
    assert result["library_versions"]["xgboost"] == "2.1"
    assert result["execution"]["bootstrap_recomputation_verified"] is True
    assert result["full_train"]["reload_predictions_exact"] is True
    assert "brier_score_approved" in result["test"]["point_metrics"]
    assert "brier_approved" not in result["test"]["point_metrics"]


def test_review_source_calls_only_public_xgb_stage_validators():
    source = inspect.getsource(review)
    assert "validate_xgb_final_training" in source
    assert "validate_xgb_validation_threshold" in source
    assert "validate_xgb_test_evaluation" in source
    assert "_validate_artifact" not in source
    assert "_locked_threshold" not in source
    assert ".fit(" not in source
    assert ".predict_proba(" not in source
