"""Synthetic Validation and threshold tests for PlainMLP."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.dataset_b.evaluation import plain_mlp_validation as validation
from src.dataset_b.evaluation import plain_mlp_validation_threshold as threshold


def _setup(tmp_path: Path, monkeypatch, probabilities=None):
    full = tmp_path / "full_train"
    (full / "checkpoints").mkdir(parents=True)
    (full / "manifest.json").write_text("{}\n", encoding="utf-8")
    (full / "full_train_result.json").write_text("{}\n", encoding="utf-8")
    (full / "checkpoints/fitted_bundle.joblib").write_bytes(b"bundle")
    split_manifest = tmp_path / "split_v1.json"
    split_manifest.write_text("{}\n", encoding="utf-8")
    split_dir = tmp_path / "split_v1"
    split_dir.mkdir()
    (split_dir / "validation.csv").write_text("record_id\nr1\nr2\nr3\nr4\n", encoding="utf-8")
    feature = tmp_path / "feature.yaml"
    feature.write_text("feature_version: synthetic\n", encoding="utf-8")
    config = tmp_path / "plain.yaml"
    config.write_text("model_version: plain_mlp_v1\n", encoding="utf-8")

    partition = SimpleNamespace(
        X=pd.DataFrame({"x": [1, 2, 3, 4]}),
        y=pd.Series([1, 0, 1, 0], dtype="int8"),
        record_ids=pd.Series(["r1", "r2", "r3", "r4"]),
    )
    loaded = []

    class Loader:
        def __init__(self, **kwargs): pass
        def load(self, name):
            loaded.append(name)
            assert name == "validation"
            return partition

    class Preprocessor:
        def __init__(self): self.transform_calls = 0
        def fit(self, *args, **kwargs): raise AssertionError("Validation must not fit")
        def fit_transform(self, *args, **kwargs): raise AssertionError("Validation must not fit")
        def transform(self, frame):
            self.transform_calls += 1
            return frame

    class Estimator:
        classes_ = np.asarray([0, 1])
        def __init__(self): self.predict_calls = 0
        def fit(self, *args, **kwargs): raise AssertionError("Validation must not fit")
        def predict_proba(self, frame):
            self.predict_calls += 1
            return np.asarray(probabilities if probabilities is not None else [
                [0.2, 0.8], [0.8, 0.2], [0.4, 0.6], [0.7, 0.3]
            ])

    preprocessor, estimator = Preprocessor(), Estimator()
    monkeypatch.setattr(validation, "EXPECTED_VALIDATION_ROWS", 4)
    monkeypatch.setattr(validation, "DatasetBSplitLoader", Loader)
    monkeypatch.setattr(
        validation, "validate_plain_mlp_full_train",
        lambda path: {"contract_id": "PMLP-03", "status": "PASS"},
    )
    monkeypatch.setattr(
        validation, "load_plain_mlp_bundle",
        lambda path: (
            preprocessor,
            estimator,
            {
                "description": "synthetic",
                "source_partition": "train",
                "validation_accessed": False,
                "test_accessed": False,
            },
        ),
    )
    paths = {
        "full_train_dir": full,
        "split_manifest_path": split_manifest,
        "split_dir": split_dir,
        "feature_contract_path": feature,
        "config_path": config,
    }
    return paths, loaded, preprocessor, estimator


def test_validation_predicts_once_in_order_without_fit(tmp_path, monkeypatch):
    paths, loaded, preprocessor, estimator = _setup(tmp_path, monkeypatch)
    output = tmp_path / "validation"

    manifest = validation.run_plain_mlp_validation(output, **paths)

    assert manifest["contract_id"] == "PMLP-04A"
    assert estimator.predict_calls == 1
    assert preprocessor.transform_calls == 1
    assert loaded == ["validation"]
    with (output / "validation_predictions.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["record_id"] for row in rows] == ["r1", "r2", "r3", "r4"]
    result = json.loads((output / "validation_result.json").read_text(encoding="utf-8"))
    assert result["execution"]["model_fit_calls"] == 0
    assert result["execution"]["preprocessing_fit_calls"] == 0
    assert result["data_boundary"]["project_test_accessed"] is False


def test_artifact_only_validator_does_not_reopen_sources(tmp_path, monkeypatch):
    paths, loaded, _, _ = _setup(tmp_path, monkeypatch)
    output = tmp_path / "validation"
    manifest = validation.run_plain_mlp_validation(output, **paths)

    def forbidden(*args, **kwargs):
        raise AssertionError("Artifact-only validation must not reopen protected sources")

    monkeypatch.setattr(validation, "DatasetBSplitLoader", forbidden)
    monkeypatch.setattr(validation, "validate_plain_mlp_full_train", forbidden)
    monkeypatch.setattr(validation, "load_plain_mlp_bundle", forbidden)

    assert validation.validate_plain_mlp_validation_artifact(output) == manifest
    assert loaded == ["validation"]


def test_validation_rejects_invalid_probabilities_without_publication(tmp_path, monkeypatch):
    paths, _, _, _ = _setup(
        tmp_path, monkeypatch,
        probabilities=[[0.2, 0.8], [np.nan, 0.5], [0.4, 0.6], [0.7, 0.3]],
    )
    output = tmp_path / "validation"
    with pytest.raises(validation.PlainMLPValidationError, match="probability"):
        validation.run_plain_mlp_validation(output, **paths)
    assert not output.exists()


def test_threshold_is_reproducible_and_does_not_predict_again(tmp_path, monkeypatch):
    paths, loaded, _, estimator = _setup(tmp_path, monkeypatch)
    output = tmp_path / "validation"
    validation.run_plain_mlp_validation(output, **paths)

    def forbidden(*args, **kwargs):
        raise AssertionError("Threshold selection must not load a bundle or protected features")

    monkeypatch.setattr(validation, "DatasetBSplitLoader", forbidden)
    monkeypatch.setattr(validation, "validate_plain_mlp_full_train", forbidden)
    monkeypatch.setattr(validation, "load_plain_mlp_bundle", forbidden)

    manifest = threshold.run_plain_mlp_validation_threshold(output)

    assert manifest["contract_id"] == "PMLP-04B"
    assert manifest["candidate_rows"] == 999
    assert estimator.predict_calls == 1
    assert loaded == ["validation"]
    selection = json.loads(
        (output / "threshold_selection/threshold_selection.json").read_text(encoding="utf-8")
    )
    assert selection["objective"] == "maximize_validation_balanced_accuracy"
    assert selection["tie_break"] == ["minimum_absolute_distance_to_0.5", "lower_threshold"]
    assert selection["data_boundary"]["model_fit_performed"] is False
    assert selection["data_boundary"]["predict_proba_calls"] == 0
    assert threshold.validate_plain_mlp_validation_threshold(output) == manifest


def test_threshold_validator_rejects_tampering(tmp_path, monkeypatch):
    paths, _, _, _ = _setup(tmp_path, monkeypatch)
    output = tmp_path / "validation"
    validation.run_plain_mlp_validation(output, **paths)
    threshold.run_plain_mlp_validation_threshold(output)
    with (output / "threshold_selection/threshold_candidates.csv").open("a", encoding="utf-8") as handle:
        handle.write("1,0.5,0.5,0.5,1,1,1,1\n")
    with pytest.raises(threshold.PlainMLPThresholdError, match="hash mismatch"):
        threshold.validate_plain_mlp_validation_threshold(output)
