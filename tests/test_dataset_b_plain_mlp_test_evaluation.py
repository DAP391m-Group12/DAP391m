"""Synthetic Test/RQ1 evaluation tests for PlainMLP."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.dataset_b.evaluation import plain_mlp_test_evaluation as evaluation


def _setup(tmp_path: Path, monkeypatch):
    synthetic_replicates = 25
    full = tmp_path / "full_train"
    (full / "checkpoints").mkdir(parents=True)
    (full / "manifest.json").write_text("{}\n", encoding="utf-8")
    (full / "checkpoints/fitted_bundle.joblib").write_bytes(b"bundle")
    (full / "full_train_result.json").write_text(
        '{"execution":{"elapsed_seconds":1.25}}\n', encoding="utf-8"
    )
    validation = tmp_path / "validation"
    (validation / "threshold_selection").mkdir(parents=True)
    (validation / "threshold_selection/threshold_selection.json").write_text(
        '{"selected_threshold":0.5}\n', encoding="utf-8"
    )
    split_manifest = tmp_path / "split_v1.json"
    split_manifest.write_text("{}\n", encoding="utf-8")
    split_dir = tmp_path / "split_v1"
    split_dir.mkdir()
    (split_dir / "test.csv").write_text("record_id\nr1\nr2\nr3\nr4\n", encoding="utf-8")
    feature = tmp_path / "feature.yaml"
    feature.write_text("feature_version: synthetic\n", encoding="utf-8")
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

    class Loader:
        def __init__(self, **kwargs): pass
        def load(self, name):
            loaded.append(name)
            assert name == "test"
            return partition

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
    monkeypatch.setattr(evaluation, "EXPECTED_TEST_ROWS", 4)
    monkeypatch.setattr(evaluation, "BOOTSTRAP_REPLICATES", synthetic_replicates)
    monkeypatch.setattr(evaluation, "DatasetBSplitLoader", Loader)
    monkeypatch.setattr(
        evaluation, "validate_plain_mlp_full_train",
        lambda path: {"contract_id": "PMLP-03", "status": "PASS"},
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
    assert set(loaded) == {"test"}
    metrics = json.loads((output / "rq1_test_metrics.json").read_text(encoding="utf-8"))
    assert metrics["threshold"] == 0.5
    assert metrics["execution"]["model_fit_calls"] == 0
    assert metrics["execution"]["preprocessing_fit_calls"] == 0
    assert metrics["data_boundary"]["threshold_changed"] is False
    assert metrics["data_boundary"]["shared_bootstrap_regenerated"] is False
    bootstrap = pd.read_csv(output / "rq1_bootstrap_metrics.csv")
    assert len(bootstrap) == 25


def test_test_validator_rejects_tampered_predictions(tmp_path, monkeypatch):
    paths, _, _, _ = _setup(tmp_path, monkeypatch)
    output = tmp_path / "test_evaluation"
    evaluation.run_plain_mlp_test_evaluation(output, **paths)
    with (output / "test_predictions.csv").open("a", encoding="utf-8") as handle:
        handle.write("r5,0,0.2,0.8,0,1\n")
    with pytest.raises(evaluation.PlainMLPTestEvaluationError, match="hash mismatch"):
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
