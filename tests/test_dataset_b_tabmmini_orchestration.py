"""Synthetic Train-only orchestration tests for TabMmini pilot/CV/full Train."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from src.dataset_b.runtime.tabmmini_engine import (
    TabMMiniBatchLock,
    TabMMiniEpochDecision,
    TabMMiniEpochRecord,
    TabMMiniRuntimeError,
)
from src.dataset_b.runtime.tabmmini_training import (
    TabMMiniRuntimeDependencies,
    TabMMiniTrainPopulation,
    run_tabmmini_cv,
    run_tabmmini_full_train,
    run_tabmmini_pilot,
)


@dataclass
class _State:
    kind: str
    identity: int


class _Preprocessor(_State):
    def fit_transform(self, frame, labels):
        self.fit_rows = len(frame)
        return frame["value"].to_numpy(dtype=np.float32).reshape(-1, 1)

    def transform(self, frame):
        if not getattr(self, "fit_rows", 0):
            raise AssertionError("unfitted preprocessing state")
        return frame["value"].to_numpy(dtype=np.float32).reshape(-1, 1)


def _population(rows=100):
    return TabMMiniTrainPopulation(
        X=pd.DataFrame({"value": np.arange(rows, dtype=np.float32)}),
        y=np.asarray([0, 1] * (rows // 2), dtype=np.int64),
        record_ids=tuple(f"train_{index:03d}" for index in range(rows)),
    )


def _locked_epoch(epoch=3):
    return TabMMiniEpochDecision(
        status="LOCKED",
        fixed_epoch=epoch,
        best_loss=0.6,
        history=(TabMMiniEpochRecord(epoch, 0.6, False, epoch),),
        reason="synthetic",
    )


def _batch_lock(size):
    selected = (256, 128, 64).index(size)
    attempts = tuple(
        {
            "attempt": index + 1,
            "batch_size": batch,
            "status": "PASS" if index == selected else "CUDA_OOM",
        }
        for index, batch in enumerate((256, 128, 64)[: selected + 1])
    )
    return TabMMiniBatchLock(size, attempts)


def _dependencies(*, fail_production_once=False):
    trace = {
        "preprocessors": [],
        "models": [],
        "optimizers": [],
        "pilot_calls": [],
        "production_calls": [],
        "prediction_calls": [],
        "state_writes": [],
        "fail_remaining": int(fail_production_once),
    }

    def preprocessor_factory():
        value = _Preprocessor("preprocessor", len(trace["preprocessors"]) + 1)
        trace["preprocessors"].append(value)
        return value

    def model_factory():
        value = _State("model", len(trace["models"]) + 1)
        trace["models"].append(value)
        return value

    def optimizer_factory(model):
        value = _State("optimizer", len(trace["optimizers"]) + 1)
        value.model_identity = model.identity
        trace["optimizers"].append(value)
        return value

    def pilot_runner(model, optimizer, fit_X, fit_y, stop_X, stop_y, pos_weight, batch, maximum):
        trace["pilot_calls"].append(
            {
                "model": model.identity,
                "optimizer": optimizer.identity,
                "fit_rows": len(fit_y),
                "stop_rows": len(stop_y),
                "pos_weight": pos_weight,
                "batch": batch,
                "maximum": maximum,
            }
        )
        return [0.9, 0.6, 0.6000005, 0.7]

    def production_runner(model, optimizer, native, labels, pos_weight, batch, epochs, early_stopping):
        trace["production_calls"].append(
            {
                "model": model.identity,
                "optimizer": optimizer.identity,
                "rows": len(labels),
                "pos_weight": pos_weight,
                "batch": batch,
                "epochs": epochs,
                "early_stopping": early_stopping,
            }
        )
        if trace["fail_remaining"]:
            trace["fail_remaining"] -= 1
            raise RuntimeError("synthetic interruption")

    def prediction_runner(model, native, batch):
        trace["prediction_calls"].append(
            {"model": model.identity, "rows": len(native), "batch": batch}
        )
        return (0.1 + native[:, 0] / 200.0).astype(np.float32)

    def state_writer(directory, model, preprocessor, metadata):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "state.json").write_text(
            json.dumps(
                {
                    "model_identity": model.identity,
                    "preprocessor_identity": preprocessor.identity,
                    "metadata": metadata,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        trace["state_writes"].append((model.identity, preprocessor.identity))

    return (
        TabMMiniRuntimeDependencies(
            preprocessor_factory=preprocessor_factory,
            model_factory=model_factory,
            optimizer_factory=optimizer_factory,
            pilot_epoch_runner=pilot_runner,
            production_fit_runner=production_runner,
            prediction_runner=prediction_runner,
            state_writer=state_writer,
        ),
        trace,
    )


def test_pilot_cv_full_train_use_one_batch_lock_fixed_epoch_and_fresh_state(tmp_path):
    dependencies, trace = _dependencies()
    population = _population()
    batch_lock = _batch_lock(128)

    pilot = run_tabmmini_pilot(
        population,
        tmp_path / "pilot",
        run_id="tabmmini_synthetic",
        batch_lock=batch_lock,
        dependencies=dependencies,
    )
    cv = run_tabmmini_cv(
        population,
        tmp_path / "cv",
        run_id="tabmmini_synthetic",
        batch_lock=batch_lock,
        pilot_decision=pilot.decision,
        dependencies=dependencies,
    )
    full = run_tabmmini_full_train(
        population,
        tmp_path / "full_train",
        run_id="tabmmini_synthetic",
        batch_lock=batch_lock,
        pilot_decision=pilot.decision,
        dependencies=dependencies,
    )

    assert pilot.decision.fixed_epoch == 2
    assert pilot.n_approved == pilot.n_denied == 8
    assert trace["pilot_calls"] == [
        {
            "model": 1,
            "optimizer": 1,
            "fit_rows": 16,
            "stop_rows": 4,
            "pos_weight": 1.0,
            "batch": 128,
            "maximum": 100,
        }
    ]
    assert len(trace["preprocessors"]) == len(trace["models"]) == len(trace["optimizers"]) == 7
    assert len({id(value) for value in trace["preprocessors"]}) == 7
    assert len({id(value) for value in trace["models"]}) == 7
    assert len({id(value) for value in trace["optimizers"]}) == 7
    assert all(call["batch"] == 128 for call in trace["production_calls"])
    assert all(call["epochs"] == 2 for call in trace["production_calls"])
    assert all(call["early_stopping"] is False for call in trace["production_calls"])
    assert cv["fresh_state_counts"] == {"preprocessors": 5, "models": 5, "optimizers": 5}
    assert full["fresh_state_counts"] == {"preprocessors": 1, "models": 1, "optimizers": 1}
    assert not (tmp_path / "pilot.working").exists()
    assert not (tmp_path / "cv.working").exists()
    assert not (tmp_path / "full_train.working").exists()


def test_oof_predictions_preserve_source_order_and_exactly_once_coverage(tmp_path):
    dependencies, _ = _dependencies()
    population = _population()
    destination = tmp_path / "cv"
    run_tabmmini_cv(
        population,
        destination,
        run_id="tabmmini_oof",
        batch_lock=_batch_lock(256),
        pilot_decision=_locked_epoch(4),
        dependencies=dependencies,
    )

    with np.load(destination / "oof_predictions.npz", allow_pickle=False) as artifact:
        assert artifact["record_ids"].astype(str).tolist() == list(population.record_ids)
        np.testing.assert_array_equal(artifact["coverage"], np.ones(100, dtype=np.int8))
        np.testing.assert_allclose(
            artifact["probabilities"],
            0.1 + population.X["value"].to_numpy(dtype=np.float32) / 200.0,
        )


def test_unresolved_pilot_blocks_cv_and_full_train_before_state_creation(tmp_path):
    dependencies, trace = _dependencies()
    unresolved = TabMMiniEpochDecision(
        status="UNRESOLVED",
        fixed_epoch=None,
        best_loss=0.1,
        history=(TabMMiniEpochRecord(100, 0.1, False, 100),),
        reason="boundary",
    )
    for runner, name in (
        (run_tabmmini_cv, "cv"),
        (run_tabmmini_full_train, "full"),
    ):
        with pytest.raises(TabMMiniRuntimeError, match="requires a locked pilot epoch"):
            runner(
                _population(),
                tmp_path / name,
                run_id="tabmmini_unresolved",
                batch_lock=_batch_lock(64),
                pilot_decision=unresolved,
                dependencies=dependencies,
            )
    assert trace["preprocessors"] == trace["models"] == trace["optimizers"] == []


def test_interrupted_stage_is_preserved_and_restart_uses_fresh_state(tmp_path):
    failing_dependencies, failing_trace = _dependencies(fail_production_once=True)
    destination = tmp_path / "full_train"
    kwargs = {
        "population": _population(),
        "destination": destination,
        "run_id": "tabmmini_restart",
        "batch_lock": _batch_lock(128),
        "pilot_decision": _locked_epoch(3),
    }
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run_tabmmini_full_train(**kwargs, dependencies=failing_dependencies)
    working = tmp_path / "full_train.working"
    assert working.is_dir()
    assert len(failing_trace["models"]) == len(failing_trace["optimizers"]) == 1

    successful_dependencies, successful_trace = _dependencies()
    result = run_tabmmini_full_train(**kwargs, dependencies=successful_dependencies)

    failed_attempt = tmp_path / "full_train.working.failed-001"
    assert failed_attempt.is_dir()
    assert destination.is_dir()
    assert not working.exists()
    assert result["resume_model_state"] is False
    assert result["resume_optimizer_state"] is False
    assert len(successful_trace["models"]) == len(successful_trace["optimizers"]) == 1
    assert successful_trace["models"][0] is not failing_trace["models"][0]


def test_published_stage_is_never_overwritten_and_locked_batch_oom_does_not_fallback(tmp_path):
    dependencies, trace = _dependencies()
    destination = tmp_path / "full_train"
    kwargs = {
        "population": _population(),
        "destination": destination,
        "run_id": "tabmmini_no_overwrite",
        "batch_lock": _batch_lock(128),
        "pilot_decision": _locked_epoch(3),
        "dependencies": dependencies,
    }
    run_tabmmini_full_train(**kwargs)
    calls_before = len(trace["production_calls"])
    with pytest.raises(TabMMiniRuntimeError, match="refuses to overwrite"):
        run_tabmmini_full_train(**kwargs)
    assert len(trace["production_calls"]) == calls_before

    oom_dependencies, oom_trace = _dependencies()

    def oom_runner(*args, **kwargs):
        oom_trace["production_calls"].append({"batch": args[5]})
        raise RuntimeError("CUDA out of memory")

    oom_dependencies = TabMMiniRuntimeDependencies(
        **{
            **oom_dependencies.__dict__,
            "production_fit_runner": oom_runner,
        }
    )
    with pytest.raises(TabMMiniRuntimeError, match="fallback is forbidden"):
        run_tabmmini_full_train(
            _population(),
            tmp_path / "oom",
            run_id="tabmmini_locked_oom",
            batch_lock=_batch_lock(128),
            pilot_decision=_locked_epoch(3),
            dependencies=oom_dependencies,
        )
    assert oom_trace["production_calls"] == [{"batch": 128}]
