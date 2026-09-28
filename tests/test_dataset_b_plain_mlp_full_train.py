"""Synthetic contract tests for the PlainMLP full-Train pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from src.dataset_b.runtime import plain_mlp_full_train as full_train


class _SyntheticPreprocessor:
    def __init__(self) -> None:
        self.fit_calls = 0
        self.transform_calls = 0

    @staticmethod
    def _matrix(frame: pd.DataFrame) -> csr_matrix:
        values = np.zeros((len(frame), 66), dtype=np.float32)
        values[:, 0] = np.arange(len(frame), dtype=np.float32)
        values[:, 1] = 1.0
        return csr_matrix(values)

    def fit_transform(self, frame: pd.DataFrame) -> csr_matrix:
        self.fit_calls += 1
        return self._matrix(frame)

    def transform(self, frame: pd.DataFrame) -> csr_matrix:
        self.transform_calls += 1
        return self._matrix(frame)


def _inputs(tmp_path: Path, monkeypatch):
    cv_preprocessing = tmp_path / "cv_preprocessing"
    cv_preprocessing.mkdir()
    (cv_preprocessing / "manifest.json").write_text("{}\n", encoding="utf-8")
    wrapper = tmp_path / "plain_mlp.py"
    wrapper.write_text("# synthetic wrapper\n", encoding="utf-8")
    cv_results = tmp_path / "cv_results.json"
    cv_results.write_text("{}\n", encoding="utf-8")
    cv_oof = tmp_path / "cv_oof_predictions.csv"
    cv_oof.write_text("record_id,p_approved\n", encoding="utf-8")
    cv_history = tmp_path / "cv_training_history.json"
    cv_history.write_text("{}\n", encoding="utf-8")
    config = tmp_path / "plain_mlp_v1.yaml"
    config.write_text(
        f"""schema_version: 1
model_version: plain_mlp_v1
status: provisional_starting_configuration
is_final_configuration: false
selected_by_train_cv: false
contracts:
  cv_artifact: "{cv_preprocessing.as_posix()}"
environment:
  device: cpu
  python_version: 3.12.13
  torch_release: 2.14.0
  torch_build: cpu
preprocessing:
  profile: plain_mlp_external_ohe
  encoded_feature_count: 66
  stored_format: csr
  stored_dtype: float32
architecture:
  hidden_layers: [4]
  activation: ReLU
  output_units: 1
training:
  loss: BCEWithLogitsLoss
  optimizer: AdamW
  learning_rate: 0.001
  weight_decay: 0.0001
  dropout: 0.0
  batch_size: 2
  epochs: 2
  random_state: 7
  deterministic_algorithms: true
class_handling:
  class_weight: false
  sample_weight: false
  resampling: false
""",
        encoding="utf-8",
    )
    selection = tmp_path / "configuration_selection.json"
    selection.write_text(
        json.dumps(
            {
                "stage": "configuration_selection_after_train_cv",
                "status": "complete",
                "decision": "accept_for_full_train_refit",
                "selected_for_run": True,
                "is_final_configuration_for_run": True,
                "external_validation_used": False,
                "test_used": False,
                "source_configuration_status": "provisional_starting_configuration",
                "feature_version": "temporary_training_lock_v0_2",
                "split_version": "split_v1",
                "preprocessing_version": "dataset_b_neural_preprocessing_v1",
                "locked_configuration": {
                    "architecture": {"hidden_layers": [4], "activation": "ReLU", "output_units": 1},
                    "training": {
                        "loss": "BCEWithLogitsLoss", "optimizer": "AdamW",
                        "learning_rate": 0.001, "weight_decay": 0.0001,
                        "dropout": 0.0, "batch_size": 2, "epochs": 2,
                        "random_state": 7, "deterministic_algorithms": True,
                    },
                    "class_handling": {
                        "class_weight": False, "sample_weight": False, "resampling": False,
                    },
                },
                "source_sha256": {
                    "plain_mlp_config": full_train._sha256_file(config),
                    "neural_preprocessing_manifest": full_train._sha256_file(cv_preprocessing / "manifest.json"),
                    "plain_mlp_wrapper": full_train._sha256_file(wrapper),
                    "cv_results": full_train._sha256_file(cv_results),
                    "cv_oof_predictions": full_train._sha256_file(cv_oof),
                    "cv_training_history": full_train._sha256_file(cv_history),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    split_manifest = tmp_path / "split_v1.json"
    split_manifest.write_text("{}\n", encoding="utf-8")
    split_dir = tmp_path / "split_v1"
    split_dir.mkdir()
    (split_dir / "train.csv").write_text("synthetic train\n", encoding="utf-8")
    feature = tmp_path / "feature_set_v1.yaml"
    feature.write_text("feature_version: synthetic\n", encoding="utf-8")

    partition = SimpleNamespace(
        X=pd.DataFrame({"x": ["0", "1", "2", "3"]}),
        y=pd.Series([0, 1, 0, 1], dtype="int8"),
        record_ids=pd.Series(["r1", "r2", "r3", "r4"]),
    )
    loaded = []

    class Loader:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load(self, name):
            loaded.append(name)
            assert name == "train"
            return partition

    preprocessors = []

    def factory():
        value = _SyntheticPreprocessor()
        preprocessors.append(value)
        return value

    monkeypatch.setattr(full_train, "EXPECTED_TRAIN_ROWS", 4)
    monkeypatch.setattr(full_train, "DEFAULT_WRAPPER_PATH", wrapper)
    monkeypatch.setattr(full_train, "DatasetBSplitLoader", Loader)
    monkeypatch.setattr(full_train, "make_plain_mlp_preprocessing_pipeline", factory)
    return {
        "selection_path": selection,
        "config_path": config,
        "split_manifest_path": split_manifest,
        "split_dir": split_dir,
        "feature_contract_path": feature,
    }, loaded, preprocessors


def test_full_train_fits_fresh_state_once_and_validates_reload(tmp_path, monkeypatch):
    paths, loaded, preprocessors = _inputs(tmp_path, monkeypatch)
    destination = tmp_path / "full_train"

    manifest = full_train.run_plain_mlp_full_train(destination, **paths)

    assert manifest["contract_id"] == "PMLP-03"
    assert manifest["status"] == "PASS"
    assert loaded == ["train", "train", "train"]
    assert len(preprocessors) == 1
    assert preprocessors[0].fit_calls == 1
    # Reload consistency is checked with the deserialized preprocessing object,
    # not by reusing the original fitted object in memory.
    assert preprocessors[0].transform_calls == 0
    result = json.loads((destination / "full_train_result.json").read_text(encoding="utf-8"))
    assert result["data_boundary"] == {
        "source_partition": "train",
        "train_rows": 4,
        "project_validation_accessed": False,
        "project_test_accessed": False,
        "preprocessing_fit_calls": 1,
        "model_fit_calls": 1,
    }
    assert result["execution"]["reload_predictions_identical"] is True
    _, estimator, scope = full_train.load_plain_mlp_bundle(
        destination / "checkpoints/fitted_bundle.joblib"
    )
    assert len(estimator.training_history_) == 2
    assert scope["validation_accessed"] is False
    assert scope["test_accessed"] is False
    assert full_train.validate_plain_mlp_full_train(destination, **paths) == manifest


def test_full_train_resume_skips_valid_bundle_and_rejects_tampering(tmp_path, monkeypatch):
    paths, loaded, _ = _inputs(tmp_path, monkeypatch)
    destination = tmp_path / "full_train"
    full_train.run_plain_mlp_full_train(destination, **paths)

    resumed = full_train.run_plain_mlp_full_train(destination, **paths)
    assert resumed["resume_action"] == "SKIPPED_VALID_COMPLETED"
    assert loaded == ["train", "train", "train", "train"]

    with (destination / "full_train_result.json").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(full_train.PlainMLPFullTrainError, match="hash mismatch"):
        full_train.validate_plain_mlp_full_train(destination, **paths)


def test_full_train_preserves_interrupted_working_directory(tmp_path, monkeypatch):
    paths, loaded, _ = _inputs(tmp_path, monkeypatch)
    destination = tmp_path / "full_train"
    working = tmp_path / "full_train.tmp"
    working.mkdir()
    (working / "trace.txt").write_text("preserve\n", encoding="utf-8")

    with pytest.raises(full_train.PlainMLPFullTrainError, match="preserved"):
        full_train.run_plain_mlp_full_train(destination, **paths)

    assert (working / "trace.txt").read_text(encoding="utf-8") == "preserve\n"
    assert loaded == []
