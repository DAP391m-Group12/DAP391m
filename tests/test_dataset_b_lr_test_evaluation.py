"""Synthetic boundary tests for LR-09 Test/RQ1 evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.dataset_b.evaluation import lr_test_evaluation as evaluation


def _setup(tmp_path: Path, monkeypatch):
    replicates = 7
    run_dir = tmp_path / "lr_final_15f_20261002_01"
    (run_dir / "checkpoints").mkdir(parents=True)
    for relative in ("manifest.json", "full_train_result.json", "checkpoints/fitted_pipeline.joblib"):
        target = run_dir / relative
        target.write_text('{"fit_seconds": 1.0}\n', encoding="utf-8")
    validation = run_dir / "validation"
    validation.mkdir()
    (validation / "manifest.json").write_text("{}\n", encoding="utf-8")
    (validation / "threshold_selection.json").write_text("{}\n", encoding="utf-8")
    split = tmp_path / "split"
    split.mkdir()
    (split / "test.csv").write_text("record_id\nr1\nr2\nr3\nr4\n", encoding="utf-8")
    split_manifest = tmp_path / "split.json"
    feature = tmp_path / "feature.yaml"
    protocol = tmp_path / "protocol.yaml"
    for path in (split_manifest, feature, protocol):
        path.write_text("{}\n", encoding="utf-8")
    ids = ("r1", "r2", "r3", "r4")
    labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
    indices = np.tile(np.asarray([0, 1, 2, 3], dtype=np.int32), (replicates, 1))
    indices_path = tmp_path / "indices.npz"
    np.savez_compressed(indices_path, indices=indices)
    bootstrap_manifest = tmp_path / "bootstrap.json"
    bootstrap_manifest.write_text(json.dumps({"indices_sha256": evaluation._sha256_file(indices_path), "test_record_id_sha256": evaluation._sha256_ids(ids), "seed": 42, "replicates": replicates, "sampling": "synthetic"}) + "\n", encoding="utf-8")
    gated = SimpleNamespace(X=pd.DataFrame({"x": [1, 2, 3, 4]}), y=pd.Series(labels), record_ids=pd.Series(ids), quality_audit=SimpleNamespace(as_dict=lambda: {"status": "PASS"}))

    class Loader:
        def __init__(self, **_kwargs):
            self.loaded: list[str] = []
        def load(self, name):
            assert name == "test"
            return gated

    class Gate:
        def apply(self, partition):
            return partition

    class Pipeline:
        def __init__(self): self.predict_calls = 0
        def fit(self, *_args, **_kwargs): raise AssertionError("LR-09 must not fit")
        def predict_proba(self, frame):
            self.predict_calls += 1
            approved = np.asarray([0.3, 0.6, 0.4, 0.7])
            return np.column_stack((1 - approved, approved))

    pipeline = Pipeline()
    monkeypatch.setattr(evaluation, "EXPECTED_TEST_ROWS", 4)
    monkeypatch.setattr(evaluation, "BOOTSTRAP_REPLICATES", replicates)
    monkeypatch.setattr(evaluation, "DatasetBSplitLoader", Loader)
    monkeypatch.setattr(evaluation, "DatasetBInputQualityGate", Gate)
    monkeypatch.setattr(evaluation, "_validate_frozen_lineage", lambda _paths: ({"preprocessing_version": "dataset_b_preprocessing_v2"}, {}, 0.5))
    monkeypatch.setattr(evaluation, "_load_frozen_pipeline", lambda _path: pipeline)
    return {
        "run_dir": run_dir,
        "output_dir": run_dir / "test",
        "split_manifest_path": split_manifest,
        "split_dir": split,
        "feature_contract_path": feature,
        "model_protocol_path": protocol,
        "bootstrap_indices_path": indices_path,
        "bootstrap_manifest_path": bootstrap_manifest,
    }, pipeline


def test_runner_predicts_once_with_frozen_threshold_and_bootstrap(tmp_path, monkeypatch):
    paths, pipeline = _setup(tmp_path, monkeypatch)
    manifest = evaluation.run_lr_test_evaluation(**paths)

    assert manifest["contract_id"] == "LR-09"
    assert manifest["status"] == "PASS"
    assert pipeline.predict_calls == 1
    metrics = json.loads((paths["output_dir"] / "rq1_test_metrics.json").read_text(encoding="utf-8"))
    assert metrics["threshold"] == 0.5
    assert metrics["execution"] == {"model_fit_calls": 0, "preprocessing_fit_calls": 0, "predict_proba_calls": 1}
    assert len(pd.read_csv(paths["output_dir"] / "rq1_bootstrap_metrics.csv")) == 7


def test_artifact_validator_rejects_tampered_predictions(tmp_path, monkeypatch):
    paths, _pipeline = _setup(tmp_path, monkeypatch)
    evaluation.run_lr_test_evaluation(**paths)
    with (paths["output_dir"] / "test_predictions.csv").open("a", encoding="utf-8") as handle:
        handle.write("r5,0,0.2,0.8,0,1\n")
    with pytest.raises(evaluation.LRTestEvaluationError, match="hash mismatch"):
        evaluation._validate_artifact(paths["output_dir"])


def test_bootstrap_comparison_rounds_only_calibration_slope():
    assert evaluation._bootstrap_metric_matches("calibration_slope", 0.9485581697164446, 0.9485581553894016)
    assert not evaluation._bootstrap_metric_matches("roc_auc_approved", 0.9485581697164446, 0.9485581553894016)


def test_output_path_must_be_current_run_test(tmp_path):
    run_dir = tmp_path / "current"
    (run_dir / "checkpoints").mkdir(parents=True)
    for relative in ("manifest.json", "checkpoints/fitted_pipeline.joblib", "validation/manifest.json", "validation/threshold_selection.json"):
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder")
    assert evaluation._resolved_paths(run_dir, run_dir / "test")["output_dir"] == (run_dir / "test").resolve()
    with pytest.raises(evaluation.LRTestEvaluationError, match="current run_dir/test"):
        evaluation._resolved_paths(run_dir, tmp_path / "wrong")
