"""Synthetic Test/RQ1 evaluation tests for PlainMLP."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from src.dataset_b.evaluation import plain_mlp_test_evaluation as evaluation
from src.dataset_b.reporting.plain_mlp_reporting import (
    REPORT_FILES,
    generate_plain_mlp_reporting,
)
from src.dataset_b.reporting import model_reporting
from src.dataset_b.review.plain_mlp_final_review import review_plain_mlp_f05


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_FEATURE_CONTRACT = PROJECT_ROOT / "config/dataset_b/feature_set_v1.yaml"


def _setup(tmp_path: Path, monkeypatch):
    synthetic_replicates = 25
    full = tmp_path / "full_train"
    (full / "checkpoints").mkdir(parents=True)
    (full / "manifest.json").write_text("{}\n", encoding="utf-8")
    (full / "checkpoints/fitted_bundle.joblib").write_bytes(b"bundle")
    (full / "full_train_result.json").write_text(
        '{"timing_seconds":{"selector":0.5,"final_fit":0.75}}\n', encoding="utf-8"
    )
    validation = tmp_path / "validation"
    (validation / "threshold_selection").mkdir(parents=True)
    (validation / "threshold_selection/threshold_selection.json").write_text(
        '{"selected_threshold":0.5,"run_id":"plain_mlp_test_run"}\n', encoding="utf-8"
    )
    split_manifest = tmp_path / "split_v1.json"
    split_manifest.write_text("{}\n", encoding="utf-8")
    split_dir = tmp_path / "split_v1"
    split_dir.mkdir()
    (split_dir / "test.csv").write_text("record_id\nr1\nr2\nr3\nr4\n", encoding="utf-8")
    feature = tmp_path / "feature.yaml"
    feature.write_bytes(PRODUCTION_FEATURE_CONTRACT.read_bytes())
    config = tmp_path / "plain.yaml"
    config.write_text("model_version: plain_mlp_v1\n", encoding="utf-8")
    ids = ("r1", "r2", "r3", "r4")
    labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
    bootstrap_indices = tmp_path / "bootstrap.npz"
    np.savez_compressed(
        bootstrap_indices,
        indices=np.tile(
            np.asarray([0, 1, 2, 3], dtype=np.int32),
            (synthetic_replicates, 1),
        ),
    )
    bootstrap_manifest = tmp_path / "bootstrap.json"
    bootstrap_manifest.write_text(
        json.dumps(
            {
                "indices_sha256": evaluation._sha256_file(bootstrap_indices),
                "test_record_id_sha256": evaluation._sha256_ids(ids),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    partition = SimpleNamespace(
        X=pd.DataFrame({"x": [1, 2, 3, 4]}),
        y=pd.Series(labels),
        record_ids=pd.Series(ids),
    )
    loaded = []
    quality_gate_calls = []

    class Loader:
        def __init__(self, **kwargs): pass
        def load(self, name):
            loaded.append(name)
            assert name == "test"
            return partition

    class QualityGate:
        def apply(self, raw_partition):
            quality_gate_calls.append("test")
            return SimpleNamespace(
                X=raw_partition.X,
                y=raw_partition.y,
                record_ids=raw_partition.record_ids,
                quality_audit=SimpleNamespace(
                    as_dict=lambda: {
                        "status": "PASS",
                        "input_rows": 4,
                        "retained_rows": 4,
                        "rejected_rows": 0,
                        "rejected_record_ids": [],
                        "blank_counts": {},
                        "raw_1111_counts": {},
                        "normalized_exempt_counts": {},
                        "semantic_1111_count": 0,
                        "row_count_preserved_after_normalization": True,
                        "row_order_preserved": True,
                        "target_alignment_preserved": True,
                        "record_id_alignment_preserved": True,
                    }
                ),
            )

    class Preprocessor:
        def __init__(self): self.transform_calls = 0
        def fit(self, *args, **kwargs): raise AssertionError("Test must not fit")
        def fit_transform(self, *args, **kwargs): raise AssertionError("Test must not fit")
        def transform(self, frame):
            self.transform_calls += 1
            return frame

    class Estimator:
        classes_ = np.asarray([0, 1])
        def __init__(self): self.predict_calls = 0
        def fit(self, *args, **kwargs): raise AssertionError("Test must not fit")
        def predict_proba(self, frame):
            self.predict_calls += 1
            approved = np.asarray([0.2, 0.8, 0.3, 0.7])
            return np.column_stack((1 - approved, approved))

    preprocessor, estimator = Preprocessor(), Estimator()
    estimator.quality_gate_calls = quality_gate_calls
    monkeypatch.setattr(evaluation, "EXPECTED_TEST_ROWS", 4)
    monkeypatch.setattr(evaluation, "BOOTSTRAP_REPLICATES", synthetic_replicates)
    monkeypatch.setattr(evaluation, "DatasetBSplitLoader", Loader)
    monkeypatch.setattr(evaluation, "DatasetBInputQualityGate", QualityGate)
    monkeypatch.setattr(
        evaluation, "validate_plain_mlp_full_train",
        lambda path, **kwargs: {
            "contract_id": "PMLP-F04-FULL-TRAIN",
            "run_id": "plain_mlp_test_run",
            "status": "PASS",
        },
    )
    monkeypatch.setattr(
        evaluation, "validate_plain_mlp_validation_threshold", lambda path: {"status": "PASS"}
    )
    monkeypatch.setattr(
        evaluation, "load_plain_mlp_bundle",
        lambda path: (preprocessor, estimator, {"description": "synthetic"}),
    )
    paths = {
        "full_train_dir": full,
        "validation_dir": validation,
        "split_manifest_path": split_manifest,
        "split_dir": split_dir,
        "feature_contract_path": feature,
        "config_path": config,
        "bootstrap_indices_path": bootstrap_indices,
        "bootstrap_manifest_path": bootstrap_manifest,
        "run_id": "plain_mlp_test_run",
    }
    return paths, loaded, preprocessor, estimator


def test_test_runner_uses_frozen_state_and_shared_bootstrap(tmp_path, monkeypatch):
    assert evaluation.BOOTSTRAP_REPLICATES == 1000
    paths, loaded, preprocessor, estimator = _setup(tmp_path, monkeypatch)
    output = tmp_path / "test_evaluation"

    manifest = evaluation.run_plain_mlp_test_evaluation(output, **paths)

    assert manifest["contract_id"] == "PMLP-05"
    assert manifest["status"] == "PASS"
    assert estimator.predict_calls == 1
    assert preprocessor.transform_calls == 1
    assert estimator.quality_gate_calls == ["test"]
    assert loaded == ["test"]
    metrics = json.loads((output / "rq1_test_metrics.json").read_text(encoding="utf-8"))
    assert metrics["evaluation_status"] == evaluation.EVALUATION_STATUS
    assert metrics["feature_version"] == evaluation.FEATURE_VERSION
    assert "temporary_training_lock" not in json.dumps(metrics)
    assert metrics["threshold"] == 0.5
    assert metrics["execution"]["model_fit_calls"] == 0
    assert metrics["execution"]["preprocessing_fit_calls"] == 0
    assert metrics["data_boundary"]["threshold_changed"] is False
    assert metrics["data_boundary"]["shared_bootstrap_regenerated"] is False
    assert metrics["data_boundary"]["quality_gate"]["status"] == "PASS"
    assert metrics["data_boundary"]["quality_gate"]["rejected_rows"] == 0
    assert metrics["data_boundary"]["quality_gate"]["row_order_preserved"] is True
    assert metrics["data_boundary"]["quality_gate"]["target_alignment_preserved"] is True
    bootstrap = pd.read_csv(output / "rq1_bootstrap_metrics.csv")
    assert len(bootstrap) == 25
    predictions = pd.read_csv(output / "test_predictions.csv")
    assert predictions.loc[0, "p_denied"] == pytest.approx(0.8)
    assert predictions.loc[0, "p_approved"] == pytest.approx(0.2)


def test_test_runner_stops_before_transform_when_quality_gate_rejects(tmp_path, monkeypatch):
    paths, _, preprocessor, estimator = _setup(tmp_path, monkeypatch)

    class RejectingQualityGate:
        def apply(self, partition):
            raise evaluation.InputQualityError("synthetic rejection")

    monkeypatch.setattr(evaluation, "DatasetBInputQualityGate", RejectingQualityGate)
    output = tmp_path / "test_evaluation"

    with pytest.raises(evaluation.PlainMLPTestEvaluationError, match="quality gate failed"):
        evaluation.run_plain_mlp_test_evaluation(output, **paths)

    assert preprocessor.transform_calls == 0
    assert estimator.predict_calls == 0
    assert not output.exists()


def test_test_runner_rejects_full_train_schema_before_test_load(tmp_path, monkeypatch):
    paths, loaded, preprocessor, estimator = _setup(tmp_path, monkeypatch)
    paths["full_train_dir"].joinpath("full_train_result.json").write_text(
        '{"execution":{"elapsed_seconds":1.25}}\n', encoding="utf-8"
    )

    with pytest.raises(evaluation.PlainMLPTestEvaluationError, match="timing schema"):
        evaluation.run_plain_mlp_test_evaluation(
            tmp_path / "test_evaluation", **paths
        )

    assert loaded == []
    assert preprocessor.transform_calls == 0
    assert estimator.predict_calls == 0


def test_test_runner_persists_exact_boundary_probabilities(tmp_path, monkeypatch):
    paths, _, _, estimator = _setup(tmp_path, monkeypatch)
    estimator.predict_proba = lambda frame: np.asarray(
        [[1.0, 0.0], [0.8, 0.2], [0.2, 0.8], [0.0, 1.0]], dtype=np.float64
    )
    output = tmp_path / "test_evaluation"

    evaluation.run_plain_mlp_test_evaluation(output, **paths)

    predictions = pd.read_csv(output / "test_predictions.csv")
    assert predictions.p_approved.tolist() == [0.0, 0.2, 0.8, 1.0]
    assert predictions.p_denied.tolist() == [1.0, 0.8, 0.2, 0.0]
    metrics = json.loads((output / "rq1_test_metrics.json").read_text(encoding="utf-8"))
    assert "logit-only clipping" in metrics["methods"]["calibration_intercept"]


def test_test_validator_rejects_tampered_predictions(tmp_path, monkeypatch):
    paths, _, _, _ = _setup(tmp_path, monkeypatch)
    output = tmp_path / "test_evaluation"
    evaluation.run_plain_mlp_test_evaluation(output, **paths)
    with (output / "test_predictions.csv").open("a", encoding="utf-8") as handle:
        handle.write("r5,0,0.2,0.8,0,1\n")
    with pytest.raises(evaluation.PlainMLPTestEvaluationError, match="hash mismatch"):
        evaluation.validate_plain_mlp_test_evaluation(output, **paths)


def test_test_validator_accepts_feature_yaml_presentation_drift(tmp_path, monkeypatch):
    paths, _, _, _ = _setup(tmp_path, monkeypatch)
    output = tmp_path / "test_evaluation"
    evaluation.run_plain_mlp_test_evaluation(output, **paths)
    feature = paths["feature_contract_path"]
    contract = yaml.safe_load(feature.read_text(encoding="utf-8"))
    feature.write_text(
        "# presentation-only change\n" + yaml.safe_dump(contract, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )

    assert evaluation.validate_plain_mlp_test_evaluation(output, **paths)["status"] == "PASS"


def test_test_validator_rejects_historical_feature_lock(tmp_path, monkeypatch):
    paths, _, _, _ = _setup(tmp_path, monkeypatch)
    output = tmp_path / "test_evaluation"
    evaluation.run_plain_mlp_test_evaluation(output, **paths)
    result_path = output / "rq1_test_metrics.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["feature_version"] = "_".join(("temporary", "training", "lock", "v0", "2"))
    result_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_sha256"]["rq1_test_metrics.json"] = evaluation._sha256_file(result_path)
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(evaluation.PlainMLPTestEvaluationError, match="boundary"):
        evaluation.validate_plain_mlp_test_evaluation(output, **paths)


def test_metric_orientation_uses_approved_probability_and_denied_complement():
    labels = np.asarray([1, 1, 0, 0], dtype=np.int8)
    probabilities = np.asarray([0.9, 0.4, 0.8, 0.1], dtype=np.float64)

    values, invalid, counts = evaluation._metrics(labels, probabilities, 0.5)

    assert invalid == {}
    assert counts["n_approved"] == 2
    assert counts["n_denied"] == 2
    assert values["balanced_accuracy"] == 0.5
    assert values["recall_approved"] == 0.5
    assert values["recall_denied"] == 0.5


def test_calibration_stabilizes_exact_zero_and_one_without_mutation():
    labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
    probabilities = np.asarray([0.0, 0.2, 0.8, 1.0], dtype=np.float64)
    original = probabilities.copy()

    intercept, slope, invalid = evaluation._calibration(labels, probabilities)

    assert np.isfinite(intercept)
    assert np.isfinite(slope)
    assert invalid == {}
    np.testing.assert_array_equal(probabilities, original)


def test_bootstrap_recomputation_rounds_only_calibration_slope_to_six_decimals():
    stored = np.asarray([0.9485581697164446, np.nan])
    recomputed = np.asarray([0.9485581553894016, np.nan])

    assert evaluation._bootstrap_metric_matches(
        "calibration_slope", stored, recomputed
    ).all()
    assert not evaluation._bootstrap_metric_matches(
        "roc_auc_approved", stored, recomputed
    ).all()
    assert not evaluation._bootstrap_metric_matches(
        "calibration_slope", stored, np.asarray([0.9485574, np.nan])
    ).all()


def test_reporting_and_final_review_are_read_only_and_digest_free(tmp_path, monkeypatch):
    paths, _, _, _ = _setup(tmp_path, monkeypatch)
    output = tmp_path / "test_evaluation"
    evaluation.run_plain_mlp_test_evaluation(output, **paths)
    metrics = json.loads((output / "rq1_test_metrics.json").read_text(encoding="utf-8"))
    cv_result = {
        "train_rows": 4,
        "folds": [
            {
                "fold": fold,
                "fit_rows": 3,
                "fold_validation_rows": 1,
                "final_estimator": {"best_epoch": 2, "pos_weight": 1.0},
                "timing_seconds": {"selector": 0.1, "final_fit": 0.2, "outer_prediction": 0.01},
                "warnings": [],
            }
            for fold in range(1, 6)
        ]
    }
    monkeypatch.setattr(model_reporting, "EXPECTED_TRAIN_ROWS", 4)
    monkeypatch.setattr(model_reporting, "EXPECTED_VALIDATION_ROWS", 4)
    monkeypatch.setattr(model_reporting, "EXPECTED_TEST_ROWS", 4)
    reporting_dir = tmp_path / "reporting"
    generate_plain_mlp_reporting(
        reporting_dir,
        run_id="plain_mlp_test_run",
        cv_result=cv_result,
        cv_metrics={
            "fold_roc_auc": [0.70, 0.71, 0.72, 0.73, 0.74],
            "mean_roc_auc": 0.72,
            "std_roc_auc": 0.01414213562373095,
            "oof_roc_auc": 0.72,
        },
        full_train_result={
            "final_estimator": {"fit_rows": 4, "best_epoch": 2},
            "timing_seconds": {"final_fit": 0.2},
            "warnings": [],
        },
        validation_result={
            "prediction_summary": {
                "rows": 4,
                "unique_record_ids": 4,
                "probability_min": 0.2,
                "probability_max": 0.8,
            },
            "data_boundary": {
                "quality_gate": {
                    "row_order_preserved": True,
                    "record_id_alignment_preserved": True,
                }
            },
        },
        threshold_selection={
            "selected_threshold": 0.5,
            "selected_metrics": {
                "balanced_accuracy": 1.0,
                "recall_approved": 1.0,
                "recall_denied": 1.0,
            },
        },
        test_metrics=metrics,
        artifact_paths={"test_metrics": output / "rq1_test_metrics.json"},
        bootstrap_indices_path=paths["bootstrap_indices_path"],
        test_predictions_path=output / "test_predictions.csv",
    )
    assert {path.name for path in reporting_dir.iterdir()} == REPORT_FILES
    assert json.loads(
        (reporting_dir / "model_summary.json").read_text(encoding="utf-8")
    )["reporting_key"] == "plain_mlp"
    assert json.loads(
        (reporting_dir / "artifact_index.json").read_text(encoding="utf-8")
    )["reporting_key"] == "plain_mlp"
    assert "sha256" not in (reporting_dir / "artifact_index.json").read_text(encoding="utf-8").lower()
    validation_result = paths["validation_dir"] / "validation_result.json"
    validation_result.write_text(
        '{"run_id":"plain_mlp_test_run"}\n', encoding="utf-8"
    )
    review = review_plain_mlp_f05(
        run_id="plain_mlp_test_run",
        validation_result_path=validation_result,
        threshold_selection_path=paths["validation_dir"] / "threshold_selection/threshold_selection.json",
        test_dir=output,
        bootstrap_indices_path=paths["bootstrap_indices_path"],
        reporting_dir=reporting_dir,
    )
    assert review["status"] == "PASS"
    assert review["execution"] == {"model_fit_calls": 0, "predict_proba_calls": 0}
    assert "sha256" not in json.dumps(review).lower()
