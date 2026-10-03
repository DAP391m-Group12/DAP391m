"""Synthetic RF-09 tests; no real model is fitted or inferred."""

from __future__ import annotations

import json
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.dataset_b.evaluation import rf_test_evaluation as evaluation


def test_rf09_has_no_lr_implementation_dependency():
    source = inspect.getsource(evaluation)
    assert "lr_test_evaluation" not in source
    assert "lr_validation_threshold" not in source


def test_bootstrap_comparison_uses_locked_metric_specific_policy():
    left = 0.9485581697164446
    right = 0.9485581553894016
    assert evaluation._bootstrap_metric_matches("calibration_slope", left, right)
    assert not evaluation._bootstrap_metric_matches("roc_auc_approved", left, right)


def test_one_test_inference_frozen_threshold_and_independent_bootstrap_recheck(tmp_path, monkeypatch):
    run_id = "rf_test_run"; run_dir = tmp_path / run_id; (run_dir / "checkpoints").mkdir(parents=True); (run_dir / "validation").mkdir()
    for relative in ("manifest.json", "checkpoints/fitted_pipeline.joblib", "validation/manifest.json", "validation/threshold_selection.json"):
        (run_dir / relative).write_text("{}\n", encoding="utf-8")
    ids = ("r1", "r2", "r3", "r4"); labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
    class Audit:
        def as_dict(self): return {"status": "PASS", "retained_rows": 4}
    gated = SimpleNamespace(X=pd.DataFrame({"x": [1, 2, 3, 4]}), y=pd.Series(labels), record_ids=pd.Series(ids), quality_audit=Audit())
    class Loader:
        def __init__(self, **_kwargs): pass
        def load(self, partition): assert partition == "test"; return gated
    class Gate:
        def apply(self, value): return value
    class Pipeline:
        def __init__(self): self.predict_calls = 0; self.fit_calls = 0
        def fit(self, *_args): self.fit_calls += 1; raise AssertionError("RF-09 must not fit")
        def predict_proba(self, frame): self.predict_calls += 1; approved = np.asarray([0.2, 0.3, 0.7, 0.8]); return np.column_stack((1 - approved, approved))
    pipeline = Pipeline(); indices = np.tile(np.arange(4, dtype=np.int32), (5, 1)); indices_path = tmp_path / "indices.npz"; np.savez_compressed(indices_path, indices=indices)
    bootstrap_manifest = tmp_path / "bootstrap.json"
    bootstrap_manifest.write_text(json.dumps({
        "dataset": "dataset_b", "split_version": "split_v1", "partition": "test",
        "test_rows": 4, "indices_file": indices_path.name, "shape": [5, 4], "dtype": "int32",
        "seed": 42, "replicates": 5, "sampling": "synthetic", "reuse": ["lr", "rf", "xgboost"],
    }), encoding="utf-8")
    protocol = tmp_path / "protocol.yaml"; protocol.write_text("{}\n", encoding="utf-8")
    split_manifest = tmp_path / "split.json"; split_manifest.write_text("{}\n", encoding="utf-8")
    feature_contract = tmp_path / "feature.yaml"; feature_contract.write_text("{}\n", encoding="utf-8")
    split_dir = tmp_path / "splits"; split_dir.mkdir(); (split_dir / "test.csv").write_text("synthetic\n", encoding="utf-8")
    monkeypatch.setattr(evaluation, "EXPECTED_TEST_ROWS", 4); monkeypatch.setattr(evaluation, "BOOTSTRAP_REPLICATES", 5)
    monkeypatch.setattr(evaluation, "DatasetBSplitLoader", Loader); monkeypatch.setattr(evaluation, "DatasetBInputQualityGate", Gate)
    monkeypatch.setattr(evaluation, "_locked_threshold", lambda _paths, _run_id: ({"feature_version": "feature_set_v1", "split_version": "split_v1", "preprocessing_version": "dataset_b_preprocessing_v2"}, 0.5))
    monkeypatch.setattr(evaluation, "_load_pipeline", lambda _path: pipeline)
    original = evaluation._bootstrap_rows; calls = {"count": 0}
    def counted(*args, **kwargs): calls["count"] += 1; return original(*args, **kwargs)
    monkeypatch.setattr(evaluation, "_bootstrap_rows", counted)
    result = evaluation.run_rf_test_evaluation(
        run_id, run_dir=run_dir, model_protocol_path=protocol,
        bootstrap_indices_path=indices_path, bootstrap_manifest_path=bootstrap_manifest,
        split_manifest_path=split_manifest, split_dir=split_dir, feature_contract_path=feature_contract,
    )
    assert result["status"] == "PASS" and pipeline.predict_calls == 1 and pipeline.fit_calls == 0
    assert calls["count"] == 2
    metrics = json.loads((run_dir / "test/rq1_test_metrics.json").read_text(encoding="utf-8"))
    assert metrics["threshold"] == 0.5 and metrics["bootstrap_contract"]["regenerated"] is False
    assert metrics["execution"] == {"model_fit_calls": 0, "preprocessing_fit_calls": 0, "predict_proba_calls": 1}
    manifest_path = run_dir / "test/run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["lineage"]["test_source"] = str((tmp_path / "wrong.csv").resolve())
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(evaluation.RFTestEvaluationError, match="lineage"):
        evaluation.validate_rf_test_evaluation(
            run_dir / "test", run_dir=run_dir, run_id=run_id,
            bootstrap_indices_path=indices_path, bootstrap_manifest_path=bootstrap_manifest,
            split_manifest_path=split_manifest, split_dir=split_dir,
            feature_contract_path=feature_contract, model_protocol_path=protocol,
        )
