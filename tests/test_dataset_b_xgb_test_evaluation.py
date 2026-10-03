"""Synthetic XGB-09 tests; no real XGBoost model is fitted or inferred."""

from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.dataset_b.evaluation import xgb_test_evaluation as evaluation


def test_source_has_no_private_rf_or_lr_dependency_or_bootstrap_generation():
    source = inspect.getsource(evaluation)
    assert "rf_test_evaluation" not in source
    assert "lr_test_evaluation" not in source
    assert "rf_validation_threshold" not in source
    assert "lr_validation_threshold" not in source
    assert "default_rng" not in source
    assert ".fit(" not in source
    assert ".fit_transform(" not in source


def test_bootstrap_comparison_uses_locked_metric_specific_policy():
    left = 0.9485581697164446
    right = 0.9485581553894016
    assert evaluation._bootstrap_metric_matches("calibration_slope", left, right)
    assert not evaluation._bootstrap_metric_matches("roc_auc_approved", left, right)


def test_confidence_interval_comparison_uses_locked_endpoint_tolerance():
    baseline = {
        name: {
            "lower_2_5": 0.4,
            "upper_97_5": 0.6,
            "valid_replicates": 1000,
            "invalid_replicates": 0,
        }
        for name in evaluation.METRIC_NAMES
    }
    within = json.loads(json.dumps(baseline))
    within["roc_auc_approved"]["lower_2_5"] += evaluation.BOOTSTRAP_RECOMPUTATION_ATOL / 2
    outside = json.loads(json.dumps(baseline))
    outside["roc_auc_approved"]["lower_2_5"] += evaluation.BOOTSTRAP_RECOMPUTATION_ATOL * 2
    wrong_counts = json.loads(json.dumps(baseline))
    wrong_counts["roc_auc_approved"]["valid_replicates"] = 999
    assert evaluation._intervals_match(within, baseline)
    assert not evaluation._intervals_match(outside, baseline)
    assert not evaluation._intervals_match(wrong_counts, baseline)


def _synthetic_state(tmp_path, monkeypatch):
    run_id = "xgb_test_run"
    run_dir = tmp_path / run_id
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "validation").mkdir()
    for relative in (
        "manifest.json",
        "checkpoints/fitted_pipeline.joblib",
        "validation/manifest.json",
        "validation/threshold_selection.json",
    ):
        (run_dir / relative).write_text("{}\n", encoding="utf-8")
    ids = ("r1", "r2", "r3", "r4")
    labels = np.asarray([0, 0, 1, 1], dtype=np.int8)

    class Audit:
        def as_dict(self):
            return {"status": "PASS", "retained_rows": 4}

    gated = SimpleNamespace(
        X=pd.DataFrame({"x": [1, 2, 3, 4]}),
        y=pd.Series(labels),
        record_ids=pd.Series(ids),
        quality_audit=Audit(),
    )
    loaded = []

    class Loader:
        def __init__(self, **_kwargs):
            pass

        def load(self, partition):
            loaded.append(partition)
            assert partition == "test"
            return gated

    class Gate:
        def apply(self, value):
            return value

    class Pipeline:
        def __init__(self):
            self.predict_calls = 0
            self.fit_calls = 0

        def fit(self, *_args):
            self.fit_calls += 1
            raise AssertionError("XGB-09 must not fit")

        def predict_proba(self, _frame):
            self.predict_calls += 1
            approved = np.asarray([0.2, 0.3, 0.7, 0.8])
            return np.column_stack((1 - approved, approved))

    pipeline = Pipeline()
    indices = np.tile(np.arange(4, dtype=np.int32), (5, 1))
    indices_path = tmp_path / "indices.npz"
    np.savez_compressed(indices_path, indices=indices)
    original_indices_bytes = indices_path.read_bytes()
    bootstrap_manifest = tmp_path / "bootstrap.json"
    bootstrap_manifest.write_text(
        json.dumps(
            {
                "dataset": "dataset_b",
                "split_version": "split_v1",
                "partition": "test",
                "test_rows": 4,
                "indices_file": indices_path.name,
                "shape": [5, 4],
                "dtype": "int32",
                "seed": 42,
                "replicates": 5,
                "sampling": "synthetic",
                "reuse": ["lr", "rf", "xgboost"],
            }
        ),
        encoding="utf-8",
    )
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text("{}\n", encoding="utf-8")
    split_manifest = tmp_path / "split.json"
    split_manifest.write_text("{}\n", encoding="utf-8")
    feature_contract = tmp_path / "feature.yaml"
    feature_contract.write_text("{}\n", encoding="utf-8")
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    (split_dir / "test.csv").write_text("synthetic\n", encoding="utf-8")
    training = {
        "run_id": run_id,
        "model_key": "xgb",
        "reporting_key": "xgboost",
        "feature_version": "feature_set_v1",
        "split_version": "split_v1",
        "preprocessing_version": "dataset_b_preprocessing_v2",
        "full_train_effective_parameters": {"scale_pos_weight": 0.5},
    }
    monkeypatch.setattr(evaluation, "EXPECTED_TEST_ROWS", 4)
    monkeypatch.setattr(evaluation, "BOOTSTRAP_REPLICATES", 5)
    monkeypatch.setattr(evaluation, "DatasetBSplitLoader", Loader)
    monkeypatch.setattr(evaluation, "DatasetBInputQualityGate", Gate)
    monkeypatch.setattr(evaluation, "_locked_threshold", lambda _paths, _run_id: (training, 0.5))
    monkeypatch.setattr(evaluation, "_load_pipeline", lambda _path, _weight: pipeline)
    original_bootstrap = evaluation._bootstrap_rows
    calls = {"count": 0}

    def counted(*args, **kwargs):
        calls["count"] += 1
        return original_bootstrap(*args, **kwargs)

    monkeypatch.setattr(evaluation, "_bootstrap_rows", counted)
    result = evaluation.run_xgb_test_evaluation(
        run_id,
        run_dir=run_dir,
        model_protocol_path=protocol,
        bootstrap_indices_path=indices_path,
        bootstrap_manifest_path=bootstrap_manifest,
        split_manifest_path=split_manifest,
        split_dir=split_dir,
        feature_contract_path=feature_contract,
    )
    return {
        "result": result,
        "run_id": run_id,
        "run_dir": run_dir,
        "ids": ids,
        "labels": labels,
        "pipeline": pipeline,
        "loaded": loaded,
        "calls": calls,
        "indices": indices,
        "indices_path": indices_path,
        "original_indices_bytes": original_indices_bytes,
        "bootstrap_manifest": bootstrap_manifest,
        "protocol": protocol,
        "split_manifest": split_manifest,
        "split_dir": split_dir,
        "feature_contract": feature_contract,
    }


def test_one_test_inference_locked_indices_and_independent_recomputation(tmp_path, monkeypatch):
    state = _synthetic_state(tmp_path, monkeypatch)
    assert state["result"]["status"] == "PASS"
    assert state["pipeline"].predict_calls == 1
    assert state["pipeline"].fit_calls == 0
    assert set(state["loaded"]) == {"test"}
    assert state["calls"]["count"] == 2
    assert state["indices_path"].read_bytes() == state["original_indices_bytes"]
    metrics = json.loads((state["run_dir"] / "test/rq1_test_metrics.json").read_text(encoding="utf-8"))
    assert len(metrics["point_metrics"]) == 10
    assert set(metrics["point_metrics"]) == set(evaluation.METRIC_NAMES)
    assert metrics["threshold"] == 0.5
    assert metrics["bootstrap_contract"]["regenerated"] is False
    assert metrics["bootstrap_contract"]["replicates"] == 5
    assert metrics["execution"] == {"model_fit_calls": 0, "preprocessing_fit_calls": 0, "predict_proba_calls": 1}


@pytest.mark.parametrize("failure", ["shape", "dtype", "bounds", "stratification", "reuse"])
def test_locked_bootstrap_rejects_invalid_payloads(tmp_path, monkeypatch, failure):
    labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
    ids = ("r1", "r2", "r3", "r4")
    indices = np.tile(np.arange(4, dtype=np.int32), (5, 1))
    if failure == "shape":
        indices = indices[:4]
    elif failure == "dtype":
        indices = indices.astype(np.int64)
    elif failure == "bounds":
        indices[0, 0] = 4
    elif failure == "stratification":
        indices[0] = np.asarray([2, 0, 1, 3], dtype=np.int32)
    indices_path = tmp_path / "indices.npz"
    np.savez_compressed(indices_path, indices=indices)
    manifest = {
        "dataset": "dataset_b",
        "split_version": "split_v1",
        "partition": "test",
        "test_rows": 4,
        "indices_file": indices_path.name,
        "shape": [5, 4],
        "dtype": "int32",
        "replicates": 5,
        "reuse": ["xgboost"],
    }
    if failure == "reuse":
        manifest["reuse"] = ["rf"]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(evaluation, "EXPECTED_TEST_ROWS", 4)
    monkeypatch.setattr(evaluation, "BOOTSTRAP_REPLICATES", 5)
    with pytest.raises(evaluation.XGBTestEvaluationError):
        evaluation._load_locked_bootstrap(indices_path, manifest_path, labels, ids)


@pytest.mark.parametrize("tamper", ["threshold", "order"])
def test_changed_threshold_or_test_order_is_rejected(tmp_path, monkeypatch, tamper):
    state = _synthetic_state(tmp_path, monkeypatch)
    if tamper == "threshold":
        metrics_path = state["run_dir"] / "test/rq1_test_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["threshold"] = 0.6
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    else:
        predictions_path = state["run_dir"] / "test/test_predictions.csv"
        with predictions_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0], rows[1] = rows[1], rows[0]
        with predictions_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=evaluation.PREDICTION_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    with pytest.raises(evaluation.XGBTestEvaluationError, match="threshold|align|metric|lineage"):
        evaluation.validate_xgb_test_evaluation(
            state["run_dir"] / "test",
            run_dir=state["run_dir"],
            run_id=state["run_id"],
            bootstrap_indices_path=state["indices_path"],
            bootstrap_manifest_path=state["bootstrap_manifest"],
            split_manifest_path=state["split_manifest"],
            split_dir=state["split_dir"],
            feature_contract_path=state["feature_contract"],
            model_protocol_path=state["protocol"],
        )


def test_bootstrap_replicate_ids_are_exact_and_sequential(tmp_path, monkeypatch):
    state = _synthetic_state(tmp_path, monkeypatch)
    bootstrap_path = state["run_dir"] / "test/rq1_bootstrap_metrics.csv"
    with bootstrap_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[1]["replicate_id"] = "9"
    with bootstrap_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=evaluation.BOOTSTRAP_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(evaluation.XGBTestEvaluationError, match="replicate IDs"):
        evaluation._validate_artifact(state["run_dir"] / "test", run_id=state["run_id"], expected_indices=state["indices"])
