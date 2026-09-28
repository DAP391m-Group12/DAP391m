"""Contract tests for the RMLP-07A Train-only full-fit runner."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.dataset_b.runtime import realmlp_full_train
from src.dataset_b.runtime.realmlp_full_train import (
    CHECKPOINT_SCOPE,
    FIXED_EPOCHS,
    RealMLPFullTrainError,
    _FullTrainInputs,
    _fit_full_train,
    run_realmlp_full_train,
    validate_realmlp_full_train,
)


def _history():
    return [
        {
            "epoch": epoch,
            "train_loss": 1.0 / epoch,
            "inner_val_loss": None,
            "learning_rate": 0.04 / epoch,
            "epoch_time_seconds": 1.0,
            "elapsed_time_seconds": float(epoch),
            "gpu_memory_used_mb": 512.0,
            "best_epoch_so_far": None,
            "status": "COMPLETED",
            "timestamp": f"2026-09-26T00:00:{epoch:02d}+00:00",
            "inner_val_status": "NOT_APPLICABLE_FIXED_EPOCH",
            "best_epoch_status": "NOT_APPLICABLE_FIXED_EPOCH",
        }
        for epoch in range(1, FIXED_EPOCHS + 1)
    ]


def _source_evidence(root: Path):
    pilot = root / "pilot_run"
    cv = root / "cv"
    runtime = root / "runtime"
    pilot.mkdir(parents=True)
    (cv / "summary").mkdir(parents=True)
    runtime.mkdir()
    (pilot / "decision.json").write_text("{}\n", encoding="utf-8")
    (cv / "summary" / "cv_results.json").write_text("{}\n", encoding="utf-8")
    (cv / "summary" / "manifest.json").write_text("{}\n", encoding="utf-8")
    (runtime / "runtime_manifest.json").write_text(
        '{"selected_batch_size": 256}\n', encoding="utf-8"
    )
    return pilot, cv, runtime


def _patch_no_training_run(monkeypatch, source_root: Path):
    pilot, cv, runtime = _source_evidence(source_root)
    ids = pd.Series(["r1", "r2", "r3"])
    train = SimpleNamespace(
        X=pd.DataFrame({"x": [1, 2, 3]}),
        y=pd.Series([0, 1, 0]),
        record_ids=ids,
    )
    metadata = {
        "source_partition": "train",
        "train_rows": 3,
        "unique_train_record_ids": 3,
        "train_record_ids_sha256": realmlp_full_train._sha256_ids(ids.tolist()),
        "preprocessing_fit_scope": "all_immutable_train_rows",
        "model_fit_scope": "all_immutable_train_rows",
        "inner_stop_set": "none",
        "project_validation_accessed": False,
        "project_test_accessed": False,
        "category_support_checked_cells": 1,
        "category_support_absent_both_warnings": 0,
        "category_support_failures": 0,
    }
    inputs = _FullTrainInputs(
        X_train=train.X,
        y_train=train.y.to_numpy(dtype=np.int64),
        preprocessing=object(),
        metadata=metadata,
    )
    fit_calls = []

    def fake_fit(directory, supplied, *, n_epochs, batch_size, device_index):
        fit_calls.append((supplied, n_epochs, batch_size, device_index))
        (directory / "checkpoints" / "fitted_model.pkl").write_bytes(b"checkpoint")
        return {
            "history": _history(),
            "metadata": {
                "elapsed_seconds": 51.0,
                "peak_gpu_memory_mb": 512.0,
                "runtime_metadata": {
                    "validation_source": "none",
                    "validation_rows": 0,
                    "use_best_epoch": False,
                    "train_rows": 3,
                    "requested_device": "cuda:0",
                    "allow_silent_cpu_fallback": False,
                    "backend_val_fraction": 0.0,
                },
                "fit_params": {"stop_epoch": {"cross_entropy": 51}},
                "fit_calls": 1,
                "predict_proba_calls": 0,
                "fit_validation_arguments": {"X_val": None, "y_val": None},
                "model_deserialization": "PASS_NO_PREDICTION",
            },
        }

    lineage = realmlp_full_train._lineage_hashes(pilot, cv, runtime)
    monkeypatch.setattr(realmlp_full_train, "EXPECTED_TRAIN_ROWS", 3)
    monkeypatch.setattr(realmlp_full_train, "_load_immutable_train", lambda: train)
    monkeypatch.setattr(realmlp_full_train, "_load_full_train_inputs", lambda: inputs)
    monkeypatch.setattr(realmlp_full_train, "_fit_full_train", fake_fit)
    monkeypatch.setattr(
        realmlp_full_train,
        "_load_run_contract",
        lambda *args, **kwargs: {
            "n_epochs": 51,
            "batch_size": 256,
            "lineage": lineage,
        },
    )
    monkeypatch.setattr(
        realmlp_full_train.joblib,
        "load",
        lambda path: {
            "estimator": SimpleNamespace(backend_=object()),
            "preprocessing": SimpleNamespace(named_steps={"native": object()}),
            "scope": CHECKPOINT_SCOPE,
        },
    )
    return pilot, cv, runtime, fit_calls


def _publish(monkeypatch, tmp_path: Path, name: str = "full_train"):
    pilot, cv, runtime, fit_calls = _patch_no_training_run(
        monkeypatch, tmp_path / f"sources_{name}"
    )
    destination = tmp_path / name
    manifest = run_realmlp_full_train(
        destination,
        pilot_run_dir=pilot,
        cv_root=cv,
        runtime_dir=runtime,
    )
    return destination, manifest, (pilot, cv, runtime), fit_calls


def _rehash_result(destination: Path):
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_sha256"]["full_train_result.json"] = (
        realmlp_full_train._sha256_file(destination / "full_train_result.json")
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_mocked_fit_runs_once_and_passes_no_validation(tmp_path, monkeypatch):
    class Recorder:
        def __init__(self, path, *, device_index, expected_epochs, stage):
            assert (device_index, expected_epochs, stage) == (0, 51, "full_train")
            self.records = []

    class Estimator:
        def __init__(self):
            self.fit_calls = []
            self.runtime_metadata_ = {
                "validation_source": "none",
                "validation_rows": 0,
                "use_best_epoch": False,
                "train_rows": 3,
                "requested_device": "cuda:0",
                "allow_silent_cpu_fallback": False,
                "backend_val_fraction": 0.0,
            }
            self.backend_ = SimpleNamespace(
                fit_params_={"stop_epoch": {"cross_entropy": 51}}
            )

        @classmethod
        def from_yaml(cls):
            return cls()

        def set_params(self, **params):
            assert params == {
                "device": "cuda:0",
                "n_epochs": 51,
                "batch_size": 256,
                "use_early_stopping": False,
                "use_best_epoch": False,
            }
            return self

        def attach_history_recorder(self, recorder):
            self.recorder = recorder
            return self

        def fit(self, X, y, **kwargs):
            self.fit_calls.append((X, y, kwargs))
            self.recorder.records = _history()
            return self

    monkeypatch.setattr(realmlp_full_train, "RealMLPHistoryRecorder", Recorder)
    monkeypatch.setattr(realmlp_full_train, "HistoryEnabledRealMLPClassifier", Estimator)
    monkeypatch.setattr("torch.cuda.empty_cache", lambda: None)
    monkeypatch.setattr("torch.cuda.reset_peak_memory_stats", lambda index: None)
    monkeypatch.setattr(
        realmlp_full_train.joblib,
        "dump",
        lambda bundle, path, compress: Path(path).write_bytes(b"checkpoint"),
    )
    saved = {}

    def fake_load(path):
        saved["path"] = Path(path)
        return {
            "estimator": SimpleNamespace(backend_=object()),
            "preprocessing": SimpleNamespace(named_steps={"native": object()}),
            "scope": CHECKPOINT_SCOPE,
        }

    monkeypatch.setattr(realmlp_full_train.joblib, "load", fake_load)
    directory = tmp_path / "full_train.tmp"
    (directory / "checkpoints").mkdir(parents=True)
    (directory / "run.log").write_text("", encoding="utf-8")
    inputs = _FullTrainInputs(
        X_train=pd.DataFrame({"x": [1, 2, 3]}),
        y_train=np.asarray([0, 1, 0]),
        preprocessing=object(),
        metadata={},
    )

    execution = _fit_full_train(
        directory, inputs, n_epochs=51, batch_size=256, device_index=0
    )

    assert execution["metadata"]["fit_calls"] == 1
    assert execution["metadata"]["predict_proba_calls"] == 0
    assert execution["metadata"]["fit_validation_arguments"] == {
        "X_val": None,
        "y_val": None,
    }
    assert len(execution["history"]) == 51
    assert saved["path"].name == "fitted_model.pkl"


def test_input_preparation_loads_only_train_with_fresh_preprocessing(monkeypatch):
    accessed = []
    train = SimpleNamespace(
        X=pd.DataFrame({"x": ["1", "2", "3"]}),
        y=pd.Series([0, 1, 0]),
        record_ids=pd.Series(["r1", "r2", "r3"]),
    )

    class Loader:
        def load(self, partition):
            accessed.append(partition)
            assert partition == "train"
            return train

    class Preprocessing:
        named_steps = {
            "native": SimpleNamespace(
                last_category_support_={
                    "category": {"A": {"outcome": "supported"}}
                }
            )
        }

        def fit_transform(self, frame):
            assert frame.equals(train.X)
            return pd.DataFrame({"x": np.asarray([1, 2, 3], dtype=np.float32)})

    created = []

    def factory():
        preprocessing = Preprocessing()
        created.append(preprocessing)
        return preprocessing

    monkeypatch.setattr(realmlp_full_train, "EXPECTED_TRAIN_ROWS", 3)
    monkeypatch.setattr(realmlp_full_train, "DatasetBSplitLoader", Loader)
    monkeypatch.setattr(
        realmlp_full_train, "make_realmlp_native_preprocessing_pipeline", factory
    )

    inputs = realmlp_full_train._load_full_train_inputs()

    assert accessed == ["train"]
    assert len(created) == 1
    assert inputs.preprocessing is created[0]
    assert inputs.metadata["project_validation_accessed"] is False
    assert inputs.metadata["project_test_accessed"] is False


def test_runner_atomically_publishes_complete_train_only_bundle(tmp_path, monkeypatch):
    destination, manifest, sources, fit_calls = _publish(monkeypatch, tmp_path)

    assert manifest["status"] == "PASS"
    assert len(fit_calls) == 1
    assert not destination.with_name("full_train.tmp").exists()
    published = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    }
    assert published == realmlp_full_train._FINAL_FILES
    result = json.loads(
        (destination / "full_train_result.json").read_text(encoding="utf-8")
    )
    assert result["configuration"]["n_epochs"] == 51
    assert result["data_boundary"]["train_rows"] == 3
    assert result["history"]["observed_records"] == 51
    assert all(record["inner_val_loss"] is None for record in _history())
    validate_realmlp_full_train(
        destination,
        pilot_run_dir=sources[0],
        cv_root=sources[1],
        runtime_dir=sources[2],
    )


def test_resume_skips_only_a_valid_completed_bundle(tmp_path, monkeypatch):
    destination, _, sources, fit_calls = _publish(monkeypatch, tmp_path)

    resumed = run_realmlp_full_train(
        destination,
        pilot_run_dir=sources[0],
        cv_root=sources[1],
        runtime_dir=sources[2],
    )

    assert resumed["resume_action"] == "SKIPPED_VALID_COMPLETED"
    assert len(fit_calls) == 1


def test_existing_invalid_bundle_is_not_overwritten(tmp_path, monkeypatch):
    destination, _, sources, fit_calls = _publish(monkeypatch, tmp_path)
    (destination / "run.log").write_text("corrupted\n", encoding="utf-8")

    with pytest.raises(RealMLPFullTrainError, match="hash mismatch"):
        run_realmlp_full_train(
            destination,
            pilot_run_dir=sources[0],
            cv_root=sources[1],
            runtime_dir=sources[2],
        )

    assert (destination / "run.log").read_text(encoding="utf-8") == "corrupted\n"
    assert len(fit_calls) == 1


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("missing_hash", "hash coverage"),
        ("wrong_hash", "hash mismatch"),
        ("wrong_epoch", "fixed-epoch"),
        ("wrong_cardinality", "cardinality"),
        ("missing_marker", "artifact set"),
    ],
)
def test_validator_rejects_broken_contract(
    tmp_path, monkeypatch, mutation, match
):
    destination, _, sources, _ = _publish(monkeypatch, tmp_path, mutation)
    manifest_path = destination / "manifest.json"
    result_path = destination / "full_train_result.json"
    if mutation == "missing_hash":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["artifact_sha256"].pop("run.log")
        manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    elif mutation == "wrong_hash":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["artifact_sha256"]["run.log"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    elif mutation == "wrong_epoch":
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["configuration"]["n_epochs"] = 50
        result_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
        _rehash_result(destination)
    elif mutation == "wrong_cardinality":
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["data_boundary"]["train_rows"] = 2
        result_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
        _rehash_result(destination)
    else:
        (destination / "COMPLETED").unlink()

    with pytest.raises(RealMLPFullTrainError, match=match):
        validate_realmlp_full_train(
            destination,
            pilot_run_dir=sources[0],
            cv_root=sources[1],
            runtime_dir=sources[2],
        )
