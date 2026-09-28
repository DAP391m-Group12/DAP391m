"""Kiểm tra hợp đồng dự đoán Validation của RealMLP."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.dataset_b.evaluation import realmlp_validation as validation
from src.dataset_b.evaluation import realmlp_validation_threshold as threshold


def _setup_inputs(tmp_path: Path, monkeypatch, probabilities=None):
    monkeypatch.setattr(validation, "EXPECTED_VALIDATION_ROWS", 3)
    full_train = tmp_path / "full_train"
    (full_train / "checkpoints").mkdir(parents=True)
    (full_train / "manifest.json").write_text('{"contract_id":"RMLP-07A"}\n')
    (full_train / "full_train_result.json").write_text('{"status":"PASS"}\n')
    (full_train / "checkpoints" / "fitted_model.pkl").write_bytes(b"checkpoint")
    split_manifest = tmp_path / "split_v1.json"
    split_manifest.write_text('{"split_version":"split_v1"}\n')
    split_dir = tmp_path / "split_v1"
    split_dir.mkdir()
    validation_csv = split_dir / "validation.csv"
    validation_csv.write_text("record_id\nr1\nr2\nr3\n", encoding="utf-8")
    feature_contract = tmp_path / "feature_set_v1.yaml"
    feature_contract.write_text("feature_version: test\n", encoding="utf-8")
    model_protocol = tmp_path / "model_protocol_v1.yaml"
    model_protocol.write_text("model_protocol_version: test\n", encoding="utf-8")

    ids = pd.Series(["r1", "r2", "r3"])
    labels = pd.Series([1, 0, 1], dtype="int8")
    partition = SimpleNamespace(
        record_ids=ids,
        y=labels,
        X=pd.DataFrame({"x": [1, 2, 3]}),
    )
    loaded_partitions = []

    class Loader:
        DEFAULT_SPLIT_DIR = split_dir

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load(self, partition_name):
            loaded_partitions.append(partition_name)
            assert partition_name == "validation"
            return partition

    class Preprocessing:
        def __init__(self):
            self.transform_calls = 0

        def transform(self, frame):
            self.transform_calls += 1
            return frame

    class Estimator:
        classes_ = np.asarray([0, 1])

        def __init__(self):
            self.predict_calls = 0

        def predict_proba(self, frame):
            self.predict_calls += 1
            if probabilities is not None:
                return np.asarray(probabilities)
            return np.asarray([[0.8, 0.2], [0.7, 0.3], [0.1, 0.9]])

    preprocessing = Preprocessing()
    estimator = Estimator()
    bundle = {
        "estimator": estimator,
        "preprocessing": preprocessing,
        "scope": "test-scope",
    }
    monkeypatch.setattr(validation, "DatasetBSplitLoader", Loader)
    monkeypatch.setattr(
        validation,
        "validate_realmlp_full_train",
        lambda path: {"contract_id": "RMLP-07A", "status": "PASS"},
    )
    monkeypatch.setattr(validation.joblib, "load", lambda path: bundle)
    paths = {
        "full_train_dir": full_train,
        "split_manifest_path": split_manifest,
        "split_dir": split_dir,
        "feature_contract_path": feature_contract,
        "model_protocol_path": model_protocol,
    }
    return paths, estimator, preprocessing, loaded_partitions


def test_run_predicts_once_and_publishes_ordered_validation_rows(tmp_path, monkeypatch):
    paths, estimator, preprocessing, loaded = _setup_inputs(tmp_path, monkeypatch)
    output = tmp_path / "artifacts" / "validation"

    manifest = validation.run_realmlp_validation(output, **paths)

    assert manifest["status"] == "PASS"
    assert manifest["rows"] == 3
    assert manifest["unique_record_ids"] == 3
    assert manifest["source_order_preserved"] is True
    assert estimator.predict_calls == 1
    assert preprocessing.transform_calls == 1
    assert loaded and set(loaded) == {"validation"}
    with (output / "validation_predictions.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == ["record_id", "y_approved", "p_approved"]
    assert [row["record_id"] for row in rows] == ["r1", "r2", "r3"]
    assert [row["y_approved"] for row in rows] == ["1", "0", "1"]
    result = json.loads((output / "validation_result.json").read_text())
    assert result["execution"] == {
        "model_fit_calls": 0,
        "preprocessing_transform_calls": 1,
        "predict_proba_calls": 1,
    }
    assert result["data_boundary"]["project_test_accessed"] is False
    assert result["prediction_summary"]["approved_rows"] == 2
    assert result["prediction_summary"]["denied_rows"] == 1


def test_resume_valid_artifact_does_not_predict_again(tmp_path, monkeypatch):
    paths, estimator, _, _ = _setup_inputs(tmp_path, monkeypatch)
    output = tmp_path / "validation"
    validation.run_realmlp_validation(output, **paths)

    resumed = validation.run_realmlp_validation(output, **paths)

    assert resumed["resume_action"] == "SKIPPED_VALID_COMPLETED"
    assert estimator.predict_calls == 1


def test_invalid_probabilities_do_not_publish_artifact(tmp_path, monkeypatch):
    paths, _, _, _ = _setup_inputs(
        tmp_path,
        monkeypatch,
        probabilities=[[0.5, 0.5], [np.nan, 0.5], [0.2, 0.8]],
    )
    output = tmp_path / "validation"

    with pytest.raises(validation.RealMLPValidationError, match="xác suất"):
        validation.run_realmlp_validation(output, **paths)

    assert not output.exists()
    assert not output.with_name("validation.tmp").exists()


def test_validator_rejects_changed_prediction_file(tmp_path, monkeypatch):
    paths, _, _, _ = _setup_inputs(tmp_path, monkeypatch)
    output = tmp_path / "validation"
    validation.run_realmlp_validation(output, **paths)
    with (output / "validation_predictions.csv").open("a", encoding="utf-8") as handle:
        handle.write("r4,0,0.2\n")

    with pytest.raises(validation.RealMLPValidationError, match="(?i)mã băm"):
        validation.validate_realmlp_validation(output, **paths)


def test_interrupted_temp_directory_is_preserved(tmp_path, monkeypatch):
    paths, estimator, _, _ = _setup_inputs(tmp_path, monkeypatch)
    output = tmp_path / "validation"
    temp = tmp_path / "validation.tmp"
    temp.mkdir()
    (temp / "trace.txt").write_text("preserve\n", encoding="utf-8")

    with pytest.raises(validation.RealMLPValidationError, match="Thư mục tạm"):
        validation.run_realmlp_validation(output, **paths)

    assert (temp / "trace.txt").read_text(encoding="utf-8") == "preserve\n"
    assert estimator.predict_calls == 0


def test_threshold_selects_balanced_accuracy_without_predicting_again(tmp_path, monkeypatch):
    paths, estimator, _, _ = _setup_inputs(tmp_path, monkeypatch)
    output = tmp_path / "validation"
    validation.run_realmlp_validation(output, **paths)
    monkeypatch.setattr(
        threshold,
        "validate_realmlp_validation",
        lambda directory: validation.validate_realmlp_validation(directory, **paths),
    )
    source_hash = validation._sha256_file(output / "validation_predictions.csv")

    manifest = threshold.run_realmlp_validation_threshold(output)

    assert manifest["contract_id"] == "RMLP-08B"
    assert manifest["candidate_rows"] == 999
    assert manifest["source_prediction_sha256"] == source_hash
    selection = json.loads(
        (output / "threshold_selection" / "threshold_selection.json").read_text()
    )
    assert selection["data_boundary"]["predict_proba_calls"] == 0
    assert selection["data_boundary"]["project_test_accessed"] is False
    assert estimator.predict_calls == 1
    assert threshold.validate_realmlp_validation_threshold(output) == manifest
    assert validation.validate_realmlp_validation(output, **paths)["status"] == "PASS"


def test_threshold_resume_skips_and_validator_rejects_tampering(tmp_path, monkeypatch):
    paths, estimator, _, _ = _setup_inputs(tmp_path, monkeypatch)
    output = tmp_path / "validation"
    validation.run_realmlp_validation(output, **paths)
    monkeypatch.setattr(
        threshold,
        "validate_realmlp_validation",
        lambda directory: validation.validate_realmlp_validation(directory, **paths),
    )
    threshold.run_realmlp_validation_threshold(output)

    resumed = threshold.run_realmlp_validation_threshold(output)
    assert resumed["resume_action"] == "SKIPPED_VALID_COMPLETED"
    assert estimator.predict_calls == 1

    candidate_path = output / "threshold_selection" / "threshold_candidates.csv"
    with candidate_path.open("a", encoding="utf-8") as handle:
        handle.write("1.0,0.5,0.5,0.5,1,1,1,1\n")
    with pytest.raises(threshold.RealMLPThresholdError, match="(?i)(mã băm|bảng ngưỡng)"):
        threshold.validate_realmlp_validation_threshold(output)
