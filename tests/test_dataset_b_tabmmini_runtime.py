"""Mocked CUDA, pilot membership, and epoch-selection tests for TabMmini."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from src.dataset_b.runtime import tabmmini_engine as engine
from src.dataset_b.splits.tabmmini_pilot import make_tabmmini_pilot_split


def _smoke_pass(batch_size):
    return {
        "epochs": 2,
        "batch_size": batch_size,
        "device": "cuda:0",
        "dtype": "float32",
        "logits_shape": [batch_size, 32, 1],
        "probability_shape": [batch_size],
        "serialization_round_trip": "PASS",
    }


def _preflight_pass():
    return {"status": "PASS", "device": "cuda:0", "cpu_fallback_allowed": False}


def test_cuda_preflight_is_cuda_only_and_checks_pinned_public_constructor(monkeypatch):
    class FakeTensor:
        device = "cuda:0"

        def __mul__(self, other):
            return self

        def sum(self):
            return self

    fake_tabm = ModuleType("tabm")
    fake_tabm.TabM = SimpleNamespace(make=lambda **kwargs: None)
    fake_torch = SimpleNamespace(
        __version__="test-torch",
        float32="float32",
        version=SimpleNamespace(cuda="test-cuda"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_properties=lambda index: SimpleNamespace(
                name="test-gpu", total_memory=16_000
            ),
        ),
        device=lambda value: value,
        tensor=lambda values, dtype, device: FakeTensor(),
        isfinite=lambda value: SimpleNamespace(item=lambda: True),
    )
    monkeypatch.setitem(sys.modules, "tabm", fake_tabm)
    monkeypatch.setattr(engine, "version", lambda package: "0.0.3")
    monkeypatch.setattr(engine, "torch", fake_torch)

    result = engine.tabmmini_cuda_preflight()

    assert result["status"] == "PASS"
    assert result["tabm_version"] == "0.0.3"
    assert result["device"] == "cuda:0"
    assert result["float32_probe"] == "PASS"
    assert result["cpu_fallback_allowed"] is False


def test_mocked_smoke_uses_locked_oom_sequence_and_whole_run_batch_lock(monkeypatch):
    calls = []
    cache_calls = []

    def attempt(batch_size):
        calls.append(batch_size)
        if batch_size == 256:
            raise RuntimeError("CUDA out of memory")
        return _smoke_pass(batch_size)

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cache_calls.append(True))
    lock = engine.run_tabmmini_cuda_smoke(
        attempt_runner=attempt,
        preflight_runner=_preflight_pass,
    )

    assert calls == [256, 128]
    assert cache_calls == [True]
    assert lock.effective_batch_size == 128
    assert lock.device == "cuda:0"
    assert lock.cpu_fallback_used is False
    assert [item["status"] for item in lock.attempts] == ["CUDA_OOM", "PASS"]


def test_smoke_rejects_cpu_preflight_non_oom_failure_and_oom_exhaustion(monkeypatch):
    with pytest.raises(engine.TabMMiniRuntimeError, match="locked PASS"):
        engine.run_tabmmini_cuda_smoke(
            attempt_runner=_smoke_pass,
            preflight_runner=lambda: {
                "status": "PASS",
                "device": "cpu",
                "cpu_fallback_allowed": True,
            },
        )

    calls = []

    def non_oom(batch_size):
        calls.append(batch_size)
        raise RuntimeError("kernel contract failure")

    with pytest.raises(engine.TabMMiniRuntimeError, match="without confirmed CUDA OOM"):
        engine.run_tabmmini_cuda_smoke(
            attempt_runner=non_oom, preflight_runner=_preflight_pass
        )
    assert calls == [256]

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    calls.clear()

    def always_oom(batch_size):
        calls.append(batch_size)
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(engine.TabMMiniRuntimeError, match="exhausted"):
        engine.run_tabmmini_cuda_smoke(
            attempt_runner=always_oom, preflight_runner=_preflight_pass
        )
    assert calls == [256, 128, 64]


def test_pilot_membership_is_deterministic_stratified_20_percent_then_80_20():
    ids = tuple(f"train_{index:03d}" for index in range(100))
    labels = np.asarray([0, 1] * 50, dtype=np.int64)

    first = make_tabmmini_pilot_split(ids, labels)
    second = make_tabmmini_pilot_split(ids, labels)

    assert first == second
    assert len(first.pilot_positions) == 20
    assert len(first.fit_positions) == 16
    assert len(first.stop_positions) == 4
    assert not set(first.fit_positions) & set(first.stop_positions)
    assert set(first.fit_positions) | set(first.stop_positions) == set(first.pilot_positions)
    assert set(first.pilot_record_ids) <= set(ids)
    assert labels[list(first.pilot_positions)].tolist().count(1) == 10
    assert labels[list(first.fit_positions)].tolist().count(1) == 8
    assert labels[list(first.stop_positions)].tolist().count(1) == 2


def test_epoch_selection_uses_weighted_stop_loss_tolerance_and_earliest_tie():
    decision = engine.select_tabmmini_fixed_epoch([0.9, 0.7, 0.7000005, 0.8])

    assert decision.status == "LOCKED"
    assert decision.fixed_epoch == 2
    assert decision.best_loss == 0.7
    assert decision.history[2].tied_with_best is True


def test_epoch_100_boundary_can_return_unresolved_and_invalid_losses_fail():
    losses = [1.0 - epoch * 0.001 for epoch in range(100)]
    decision = engine.select_tabmmini_fixed_epoch(losses)

    assert decision.status == "UNRESOLVED"
    assert decision.fixed_epoch is None
    assert decision.history[-1].epoch == 100
    with pytest.raises(engine.TabMMiniRuntimeError, match="finite and non-negative"):
        engine.select_tabmmini_fixed_epoch([0.5, np.nan])
