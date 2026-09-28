"""Contract tests for the Dataset B RealMLP GPU/history runtime gate."""

import json

import numpy as np
import pytest
import torch

from src.dataset_b.runtime.realmlp_smoke import (
    RealMLPSmokeExecution,
    RealMLPSmokeOOM,
    _Tee,
    _validate_history_records,
    run_realmlp_runtime_gate,
)
from src.dataset_b.runtime.realmlp_history import (
    RealMLPHistoryRecorder,
    _RealMLPHistoryCallback,
)


def _environment(*, ready=True):
    return {
        "timestamp": "2026-09-25T00:00:00+00:00",
        "python_version": "3.13.9",
        "python_executable": "python",
        "operating_system": "test",
        "packages": {"torch": "2.14.0", "pytabkit": "1.7.3"},
        "device_index": 0,
        "cuda_available": ready,
        "cuda_build_version": "13.0" if ready else None,
        "nvidia_driver_version": "999.0" if ready else None,
        "gpu_name": "Fake GPU" if ready else None,
        "total_vram_bytes": 16 * 1024**3 if ready else None,
        "finite_cuda_tensor_calculation": ready,
        "pytabkit_estimator_import": True,
        "errors": [] if ready else ["torch.cuda.is_available() is False."],
        "ready_for_smoke": ready,
    }


def _history():
    return tuple(
        {
            "epoch": epoch,
            "train_loss": 0.8 / epoch,
            "inner_val_loss": 0.9 / epoch,
            "learning_rate": 0.04,
            "epoch_time_seconds": 1.5,
            "elapsed_time_seconds": 1.5 * epoch,
            "gpu_memory_used_mb": 512.0,
            "best_epoch_so_far": epoch,
            "status": "COMPLETED",
            "timestamp": f"2026-09-25T00:00:0{epoch}+00:00",
        }
        for epoch in (1, 2)
    )


def _success(batch_size, run_log_path):
    run_log_path.write_text(f"batch_size={batch_size}\n", encoding="utf-8")
    return RealMLPSmokeExecution(
        probabilities=np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float64),
        history_records=_history(),
        history_method="public_backend_history",
        raw_log="two epochs",
        metadata={"requested_device": "cuda:0"},
    )


def test_runtime_gate_publishes_complete_hashed_evidence(tmp_path):
    output_dir = tmp_path / "runtime"
    manifest = run_realmlp_runtime_gate(
        output_dir,
        _environment_probe=lambda index: _environment(),
        _smoke_attempt=_success,
    )

    assert manifest["status"] == "PASS"
    assert manifest["selected_batch_size"] == 256
    assert {path.name for path in output_dir.iterdir()} == {
        "runtime_manifest.json",
        "gpu_compatibility.log",
        "smoke_result.json",
        "run.log",
        "training_history.jsonl",
        "training_history.json",
    }
    smoke = json.loads((output_dir / "smoke_result.json").read_text(encoding="utf-8"))
    assert smoke["probabilities"]["valid"] is True
    assert smoke["history"]["complete"] is True
    assert smoke["history"]["valid_records"] == 2
    assert len(manifest["artifact_sha256"]) == 5


def test_runtime_gate_reduces_batch_only_after_confirmed_oom(tmp_path):
    attempted = []

    def attempt(batch_size, run_log_path):
        attempted.append(batch_size)
        if batch_size == 256:
            raise RealMLPSmokeOOM("CUDA out of memory")
        return _success(batch_size, run_log_path)

    manifest = run_realmlp_runtime_gate(
        tmp_path / "runtime",
        _environment_probe=lambda index: _environment(),
        _smoke_attempt=attempt,
    )
    assert attempted == [256, 128]
    assert manifest["status"] == "PASS"
    assert manifest["selected_batch_size"] == 128


def test_runtime_gate_does_not_train_without_cuda_and_records_not_ready(tmp_path):
    called = False

    def attempt(batch_size, run_log_path):
        nonlocal called
        called = True
        return _success(batch_size, run_log_path)

    output_dir = tmp_path / "runtime"
    manifest = run_realmlp_runtime_gate(
        output_dir,
        _environment_probe=lambda index: _environment(ready=False),
        _smoke_attempt=attempt,
    )
    smoke = json.loads((output_dir / "smoke_result.json").read_text(encoding="utf-8"))

    assert called is False
    assert manifest["status"] == "NOT_READY"
    assert manifest["selected_batch_size"] is None
    assert smoke["attempts"] == []
    assert smoke["history"]["complete"] is False


def test_runtime_gate_stops_fallback_after_non_oom_failure(tmp_path):
    attempted = []

    def attempt(batch_size, run_log_path):
        attempted.append(batch_size)
        raise RuntimeError("unexpected backend failure")

    output_dir = tmp_path / "runtime"
    manifest = run_realmlp_runtime_gate(
        output_dir,
        _environment_probe=lambda index: _environment(),
        _smoke_attempt=attempt,
    )
    smoke = json.loads((output_dir / "smoke_result.json").read_text(encoding="utf-8"))

    assert attempted == [256]
    assert manifest["status"] == "NOT_READY"
    assert smoke["attempts"][0]["status"] == "FAILED"


def test_runtime_gate_rejects_incomplete_history(tmp_path):
    def incomplete(batch_size, run_log_path):
        return RealMLPSmokeExecution(
            probabilities=np.asarray([[0.5, 0.5]], dtype=np.float64),
            history_records=(
                {
                    "epoch": 1,
                    "inner_val_loss": 0.5,
                    "status": "INCOMPLETE",
                    "timestamp": "2026-09-25T00:00:01+00:00",
                },
            ),
            history_method="terminal_output_parser",
            raw_log="partial",
            metadata={},
        )

    output_dir = tmp_path / "runtime"
    manifest = run_realmlp_runtime_gate(
        output_dir,
        _environment_probe=lambda index: _environment(),
        _smoke_attempt=incomplete,
    )
    history = json.loads((output_dir / "training_history.json").read_text(encoding="utf-8"))

    assert manifest["status"] == "NOT_READY"
    assert history["complete"] is False
    assert "train_loss" in history["missing_fields_by_epoch"]["1"]


def test_history_recorder_appends_complete_truthful_epoch(tmp_path, monkeypatch):
    monkeypatch.setattr("torch.cuda.reset_peak_memory_stats", lambda index: None)
    monkeypatch.setattr("torch.cuda.synchronize", lambda index: None)
    monkeypatch.setattr("torch.cuda.max_memory_allocated", lambda index: 64 * 1024**2)
    ticks = iter((10.0, 11.0, 13.5))
    monkeypatch.setattr(
        "src.dataset_b.runtime.realmlp_history.time.perf_counter",
        lambda: next(ticks),
    )
    path = tmp_path / "training_history.jsonl"
    recorder = RealMLPHistoryRecorder(path, device_index=0, expected_epochs=1)

    recorder.start_fit()
    recorder.start_epoch()
    recorder.add_train_batch(0.8, 2)
    recorder.add_train_batch(0.4, 2)
    recorder.finish_training(0.01)
    recorder.record_validation(1, 0.5)
    record = recorder.complete_epoch(1, 1)

    assert record == recorder.records[0]
    assert record["train_loss"] == pytest.approx(0.6)
    assert record["inner_val_loss"] == 0.5
    assert record["learning_rate"] == 0.01
    assert record["epoch_time_seconds"] == 2.5
    assert record["elapsed_time_seconds"] == 3.5
    assert record["gpu_memory_used_mb"] == 64.0
    assert record["status"] == "COMPLETED"
    assert json.loads(path.read_text(encoding="utf-8")) == record


def test_tee_flush_ignores_already_closed_stream(tmp_path):
    path = tmp_path / "closed.log"
    stream = path.open("w", encoding="utf-8")
    tee = _Tee(stream)
    stream.close()

    tee.flush()


def test_history_callback_closes_training_metrics_before_validation_end(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("torch.cuda.reset_peak_memory_stats", lambda index: None)
    monkeypatch.setattr("torch.cuda.synchronize", lambda index: None)
    monkeypatch.setattr("torch.cuda.max_memory_allocated", lambda index: 1024**2)

    class _Optimizer:
        @staticmethod
        def get_hyper_values(name, index, use_hyper_factor):
            assert (name, index, use_hyper_factor) == ("lr", 0, False)
            return 0.02

    class _Progress:
        epoch = 1

    class _Module:
        progress = _Progress()
        val_metric_names = ["cross_entropy"]
        best_mean_val_epochs = {"cross_entropy": [1]}

        @staticmethod
        def optimizers(use_pl_optimizer):
            assert use_pl_optimizer is False
            return _Optimizer()

    recorder = RealMLPHistoryRecorder(
        tmp_path / "history.jsonl", device_index=0, expected_epochs=1
    )
    callback = _RealMLPHistoryCallback(recorder)
    module = _Module()
    callback.on_fit_start(None, module)
    callback.on_train_epoch_start(None, module)
    callback.on_train_batch_end(
        None,
        module,
        torch.tensor(0.4),
        {"y": torch.zeros((1, 3, 1))},
        0,
    )
    callback.on_validation_start(None, module)
    recorder.record_validation(1, 0.3)
    callback.on_validation_end(None, module)

    assert recorder.records[0]["train_loss"] == pytest.approx(0.4)
    assert recorder.records[0]["learning_rate"] == 0.02


def test_fixed_epoch_history_records_validation_as_not_applicable(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("torch.cuda.reset_peak_memory_stats", lambda index: None)
    monkeypatch.setattr("torch.cuda.synchronize", lambda index: None)
    monkeypatch.setattr("torch.cuda.max_memory_allocated", lambda index: 2 * 1024**2)
    ticks = iter((10.0, 11.0, 12.0))
    monkeypatch.setattr(
        "src.dataset_b.runtime.realmlp_history.time.perf_counter",
        lambda: next(ticks),
    )
    recorder = RealMLPHistoryRecorder(
        tmp_path / "history.jsonl",
        device_index=0,
        expected_epochs=1,
        stage="cv_smoke_fold_01",
    )
    recorder.start_fit()
    recorder.start_epoch()
    recorder.add_train_batch(0.4, 4)
    recorder.finish_training(0.01)
    record = recorder.complete_epoch(
        1, None, inner_validation_required=False
    )

    assert record["inner_val_loss"] is None
    assert record["best_epoch_so_far"] is None
    assert record["inner_val_status"] == "NOT_APPLICABLE_FIXED_EPOCH"
    assert record["best_epoch_status"] == "NOT_APPLICABLE_FIXED_EPOCH"
    summary = _validate_history_records(
        [record], 1, inner_validation_required=False
    )
    assert summary["complete"] is True


def test_history_callback_completes_epoch_without_validation(tmp_path, monkeypatch):
    monkeypatch.setattr("torch.cuda.reset_peak_memory_stats", lambda index: None)
    monkeypatch.setattr("torch.cuda.synchronize", lambda index: None)
    monkeypatch.setattr("torch.cuda.max_memory_allocated", lambda index: 1024**2)

    class _Optimizer:
        @staticmethod
        def get_hyper_values(name, index, use_hyper_factor):
            assert (name, index, use_hyper_factor) == ("lr", 0, False)
            return 0.02

    class _Module:
        @staticmethod
        def optimizers(use_pl_optimizer):
            assert use_pl_optimizer is False
            return _Optimizer()

    class _Trainer:
        current_epoch = 0

    recorder = RealMLPHistoryRecorder(
        tmp_path / "history.jsonl", device_index=0, expected_epochs=1
    )
    callback = _RealMLPHistoryCallback(
        recorder, inner_validation_enabled=False
    )
    module = _Module()
    callback.on_fit_start(_Trainer(), module)
    callback.on_train_epoch_start(_Trainer(), module)
    callback.on_train_batch_end(
        _Trainer(),
        module,
        torch.tensor(0.4),
        {"y": torch.zeros((1, 3, 1))},
        0,
    )
    callback.on_train_epoch_end(_Trainer(), module)

    assert len(recorder.records) == 1
    assert recorder.records[0]["epoch"] == 1
    assert recorder.records[0]["inner_val_loss"] is None
    assert recorder.records[0]["learning_rate"] == 0.02
