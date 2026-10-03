"""RF-10 refusal tests for incomplete, legacy, and misaligned evidence."""

import json

import pytest

from src.dataset_b.review import rf_final_review as review
from src.dataset_b.training.rf_final_training import RFTrainingError


def test_review_refuses_incomplete_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "DEFAULT_RF_ROOT", tmp_path)
    monkeypatch.setattr(review, "DEFAULT_RF_FINAL_ROOT", tmp_path / "final")
    run_id = "rf_run"; root = tmp_path / run_id; root.mkdir()
    with pytest.raises(review.RFFinalReviewError, match="incomplete"):
        review.review_rf_final(run_id, run_dir=root)


def test_review_refuses_legacy_run_id(tmp_path):
    root = tmp_path / "legacy_rf"; root.mkdir()
    with pytest.raises(RFTrainingError, match="Legacy"):
        review.review_rf_final("legacy_rf", run_dir=root)


def test_review_refuses_misaligned_stage_lineage(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "DEFAULT_RF_ROOT", tmp_path)
    monkeypatch.setattr(review, "DEFAULT_RF_FINAL_ROOT", tmp_path / "final")
    run_id = "rf_run"; root = tmp_path / run_id; (root / "validation").mkdir(parents=True); (root / "test").mkdir()
    monkeypatch.setattr(review, "validate_rf_final_training", lambda *_a, **_k: {"run_id": run_id, "feature_version": "feature_set_v1", "split_version": "split_v1"})
    monkeypatch.setattr(review, "validate_rf_validation_threshold", lambda *_a, **_k: {"run_id": "other", "feature_version": "feature_set_v1", "split_version": "split_v1"})
    monkeypatch.setattr(review, "validate_rf_test_evaluation", lambda *_a, **_k: {"run_id": run_id, "feature_version": "feature_set_v1", "split_version": "split_v1"})
    with pytest.raises(review.RFFinalReviewError, match="misaligned run IDs"):
        review.review_rf_final(run_id, run_dir=root)


def test_review_complete_pass_returns_common_reporting_summary(tmp_path, monkeypatch):
    run_id = "rf_run"; root = tmp_path / run_id; (root / "validation").mkdir(parents=True); (root / "test").mkdir()
    monkeypatch.setattr(review, "DEFAULT_RF_ROOT", tmp_path)
    monkeypatch.setattr(review, "DEFAULT_RF_FINAL_ROOT", tmp_path / "final")
    training = {"run_id": run_id, "feature_version": "feature_set_v1", "split_version": "split_v1", "preprocessing_version": "dataset_b_preprocessing_v2"}
    validation = {"run_id": run_id, "feature_version": "feature_set_v1", "split_version": "split_v1", "rows": 43_995, "unique_record_ids": 43_995, "source_order_preserved": True}
    test = {"run_id": run_id, "feature_version": "feature_set_v1", "split_version": "split_v1", "rows": 43_996, "unique_record_ids": 43_996, "source_order_preserved": True, "threshold": 0.5}
    monkeypatch.setattr(review, "validate_rf_final_training", lambda *_a, **_k: training)
    monkeypatch.setattr(review, "validate_rf_validation_threshold", lambda *_a, **_k: validation)
    monkeypatch.setattr(review, "validate_rf_test_evaluation", lambda *_a, **_k: test)
    for relative in ("manifest.json", "validation/manifest.json", "test/run_manifest.json"):
        (root / relative).write_text("{}\n", encoding="utf-8")
    (root / "cv_results.json").write_text(json.dumps({"fold_count": 5, "folds": [{"warning_count": 0} for _ in range(5)], "mean_fold_roc_auc": 0.8, "std_fold_roc_auc": 0.01, "oof_roc_auc": 0.8, "oof_rows": 205_310}), encoding="utf-8")
    (root / "full_train_result.json").write_text(json.dumps({"retained_train_rows": 205_310, "fit_seconds": 1.0, "warning_count": 0}), encoding="utf-8")
    (root / "validation/threshold_selection.json").write_text(json.dumps({"selected_threshold": 0.5, "selected_metrics": {"balanced_accuracy": 0.7, "recall_approved": 0.8, "recall_denied": 0.6}}), encoding="utf-8")
    names = ("roc_auc_approved", "average_precision_denied", "precision_denied", "recall_denied", "f1_denied", "recall_approved", "balanced_accuracy", "brier_approved", "calibration_intercept", "calibration_slope")
    metrics = {
        "threshold": 0.5, "point_metrics": {name: 0.5 for name in names},
        "confidence_intervals_95_percentile": {name: {"lower_2_5": 0.4, "upper_97_5": 0.6, "valid_replicates": 1000, "invalid_replicates": 0} for name in names},
        "counts": {}, "calibration_curve": [],
        "lineage": {"test_record_ids_verified": True, "bootstrap_indices": "bootstrap-indices.npz"},
        "bootstrap_contract": {"regenerated": False},
        "execution": {"model_fit_calls": 0, "preprocessing_fit_calls": 0, "predict_proba_calls": 1},
    }
    (root / "test/rq1_test_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    result = review.review_rf_final(run_id, run_dir=root)
    assert result["reporting_status"] == "COMPLETE"
    assert "brier_score_approved" in result["test"]["point_metrics"]
    assert "brier_approved" not in result["test"]["point_metrics"]


def test_review_accepts_canonical_final_root(tmp_path, monkeypatch):
    run_id = "rf_run"
    final_root = tmp_path / "final"
    root = final_root / run_id
    (root / "validation").mkdir(parents=True)
    (root / "test").mkdir()
    monkeypatch.setattr(review, "DEFAULT_RF_ROOT", tmp_path / "working")
    monkeypatch.setattr(review, "DEFAULT_RF_FINAL_ROOT", final_root)
    monkeypatch.setattr(
        review,
        "validate_rf_final_training",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("final root accepted")),
    )
    with pytest.raises(RuntimeError, match="final root accepted"):
        review.review_rf_final(run_id, run_dir=root)
