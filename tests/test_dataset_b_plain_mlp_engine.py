"""Tests for the PlainMLP CUDA smoke and Train-only epoch selector."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from src.dataset_b.runtime import plain_mlp_engine as engine


def _data(rows=40):
    dense = np.zeros((rows, 62), dtype=np.float64)
    dense[:, 0] = np.linspace(0.0, 1.0, rows)
    dense[:, 4] = 1.0
    y = np.asarray([0, 1] * (rows // 2), dtype=np.float64)
    ids = tuple(f"train_{index:03d}" for index in range(rows))
    return csr_matrix(dense), y, ids


def _preflight_pass():
    return {"status": "PASS", "device": "cuda:0"}


def _smoke_pass(batch_size):
    return {
        "epochs": 2,
        "batch_size": batch_size,
        "device": "cuda:0",
        "dtype": "float64",
    }


def test_cuda_preflight_is_read_only_and_validates_float64_device(monkeypatch, tmp_path):
    class FakeTensor:
        device = "cuda:0"
        dtype = engine.LOCKED_DTYPE

        def __mul__(self, other):
            return self

        def sum(self):
            return self

        def item(self):
            return True

    fake_torch = SimpleNamespace(
        __version__="test-torch",
        float64=engine.LOCKED_DTYPE,
        version=SimpleNamespace(cuda="test-cuda"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_properties=lambda index: SimpleNamespace(
                name="test-gpu", total_memory=16_000
            ),
        ),
        device=lambda value: value,
        asarray=lambda values, device, dtype: FakeTensor(),
        isfinite=lambda value: value,
    )
    monkeypatch.setattr(engine, "torch", fake_torch)

    result = engine.plain_mlp_cuda_preflight()

    assert result == {
        "status": "PASS",
        "torch_version": "test-torch",
        "cuda_build": "test-cuda",
        "device": "cuda:0",
        "device_index": 0,
        "gpu_name": "test-gpu",
        "total_memory_bytes": 16_000,
        "float64_probe": "PASS",
        "cpu_fallback_allowed": False,
    }
    assert list(tmp_path.iterdir()) == []


def test_cuda_preflight_rejects_unavailable_cuda_without_cpu_fallback(monkeypatch):
    fake_torch = SimpleNamespace(
        version=SimpleNamespace(cuda="test-cuda"),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setattr(engine, "torch", fake_torch)

    with pytest.raises(engine.PlainMLPEngineError, match="CPU fallback is forbidden"):
        engine.plain_mlp_cuda_preflight()


def test_smoke_falls_back_only_after_confirmed_cuda_oom(monkeypatch):
    X, y, _ = _data()
    calls = []
    empty_cache_calls = []

    def runner(batch_size):
        calls.append(batch_size)
        if batch_size == 256:
            raise RuntimeError("CUDA out of memory")
        return _smoke_pass(batch_size)

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: empty_cache_calls.append(True))
    result = engine.run_plain_mlp_cuda_smoke(
        X,
        y,
        attempt_runner=runner,
        preflight_runner=_preflight_pass,
    )

    assert calls == [256, 128]
    assert empty_cache_calls == [True]
    assert result.selected_batch_size == 128
    assert result.epochs == 2
    assert result.dtype == "float64"
    assert result.device == "cuda:0"
    assert result.cpu_fallback_used is False
    assert [attempt["status"] for attempt in result.attempts] == ["CUDA_OOM", "PASS"]


def test_non_oom_failure_does_not_trigger_smaller_batch():
    X, y, _ = _data()
    calls = []

    def runner(batch_size):
        calls.append(batch_size)
        raise RuntimeError("kernel contract failure")

    with pytest.raises(engine.PlainMLPEngineError, match="without confirmed CUDA OOM"):
        engine.run_plain_mlp_cuda_smoke(
            X,
            y,
            attempt_runner=runner,
            preflight_runner=_preflight_pass,
        )
    assert calls == [256]


def test_smoke_exhaustion_stops_without_cpu_fallback(monkeypatch):
    X, y, _ = _data()
    calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    def runner(batch_size):
        calls.append(batch_size)
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(engine.PlainMLPEngineError, match="exhausted"):
        engine.run_plain_mlp_cuda_smoke(
            X,
            y,
            attempt_runner=runner,
            preflight_runner=_preflight_pass,
        )
    assert calls == [256, 128, 64]


def test_inner_split_is_stratified_disjoint_complete_and_train_only():
    _, y, ids = _data()
    split = engine.make_plain_mlp_inner_split(
        ids,
        y,
        outer_validation_record_ids=("outer_1",),
        project_validation_record_ids=("validation_1",),
        project_test_record_ids=("test_1",),
    )

    assert len(split.fit_positions) == 36
    assert len(split.stop_positions) == 4
    assert not set(split.fit_positions) & set(split.stop_positions)
    assert set(split.fit_positions) | set(split.stop_positions) == set(range(40))
    assert set(split.fit_record_ids) | set(split.stop_record_ids) == set(ids)
    assert not set(split.fit_record_ids) & set(split.stop_record_ids)
    assert y[list(split.fit_positions)].tolist().count(0) == 18
    assert y[list(split.fit_positions)].tolist().count(1) == 18
    assert y[list(split.stop_positions)].tolist().count(0) == 2
    assert y[list(split.stop_positions)].tolist().count(1) == 2
    assert split == engine.make_plain_mlp_inner_split(ids, y)


@pytest.mark.parametrize(
    "forbidden_argument,error",
    [
        ("outer_validation_record_ids", "outer fold-validation"),
        ("project_validation_record_ids", "project Validation"),
        ("project_test_record_ids", "project Test"),
    ],
)
def test_inner_split_rejects_outer_validation_or_project_holdout_leakage(
    forbidden_argument, error
):
    _, y, ids = _data()
    with pytest.raises(ValueError, match=error):
        engine.make_plain_mlp_inner_split(
            ids,
            y,
            **{forbidden_argument: (ids[0],)},
        )


def test_best_epoch_tie_and_patience_behavior_are_deterministic():
    X, y, ids = _data()
    losses = [1.0, 0.8, *([0.8] * 10), 0.1]
    observed_shapes = []

    def stream_factory(X_fit, y_fit, X_stop, y_stop, batch_size):
        observed_shapes.append((X_fit.shape, y_fit.shape, X_stop.shape, y_stop.shape, batch_size))
        return iter(losses)

    first = engine.run_plain_mlp_inner_selector(
        X,
        y,
        ids,
        batch_size=256,
        epoch_loss_stream_factory=stream_factory,
    )
    second = engine.run_plain_mlp_inner_selector(
        X,
        y,
        ids,
        batch_size=256,
        epoch_loss_stream_factory=stream_factory,
    )

    assert first == second
    assert first.best_epoch == 2
    assert first.best_inner_stop_bce_loss == 0.8
    assert first.stopped_after_epoch == 12
    assert first.stop_reason == "patience"
    assert first.history[-1].epochs_without_improvement == 10
    assert all(shape == ((36, 62), (36,), (4, 62), (4,), 256) for shape in observed_shapes)


def test_selector_stops_at_locked_maximum_epoch():
    X, y, ids = _data()
    losses = [2.0 - (epoch * 0.001) for epoch in range(105)]

    result = engine.run_plain_mlp_inner_selector(
        X,
        y,
        ids,
        batch_size=64,
        epoch_loss_stream_factory=lambda *args: iter(losses),
    )

    assert result.best_epoch == 100
    assert result.stopped_after_epoch == 100
    assert result.stop_reason == "max_epochs"
    assert len(result.history) == 100


def test_selector_does_not_publish_temporary_checkpoint_and_final_refit_is_fresh(
    monkeypatch, tmp_path
):
    X, y, ids = _data()
    monkeypatch.setattr(
        torch,
        "save",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("temporary selector checkpoint publication is forbidden")
        ),
    )
    result = engine.run_plain_mlp_inner_selector(
        X,
        y,
        ids,
        batch_size=128,
        epoch_loss_stream_factory=lambda *args: iter([0.9, 0.7, *([0.7] * 10)]),
    )

    first_final = engine.make_plain_mlp_final_refit_estimator(
        result.best_epoch, batch_size=128
    )
    second_final = engine.make_plain_mlp_final_refit_estimator(
        result.best_epoch, batch_size=128
    )

    assert result.best_epoch == 2
    assert result.temporary_state_published is False
    assert not hasattr(result, "model") and not hasattr(result, "optimizer")
    assert list(tmp_path.iterdir()) == []
    assert first_final is not second_final
    assert first_final.epochs == second_final.epochs == result.best_epoch
    assert not hasattr(first_final, "model_") and not hasattr(second_final, "model_")
