from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import src.dataset_b.gsv_sl.runner as runner_module
from src.dataset_b.gsv_sl.exact_methods import (
    agreement_metrics,
    bootstrap_cached_agreement,
    gsv_from_prediction_cube,
    rank_importance,
    shapley_lorenz_from_prediction_cube,
)
from src.dataset_b.gsv_sl.predictors import FEATURE_COLUMNS, TimedPrediction
from src.dataset_b.gsv_sl.runner import (
    DEFAULT_PROTOCOL_PATH,
    RQ2RunnerError,
    _cross_model_rows,
    _read_protocol,
    build_coalition_prediction_cube,
    prepare_sampling_inputs,
    run_rq2,
)


class _TimedFakePredictor:
    model_key = "fake"
    run_id = "fake_frozen_run"
    execution_device = "cpu"

    def predict_p_approved_timed(self, frame: pd.DataFrame) -> TimedPrediction:
        values = (0.1 + 0.2 * frame["loan_type"].eq("E").to_numpy(dtype=np.float64)
                  + 0.3 * frame["occupancy_type"].eq("E").to_numpy(dtype=np.float64))
        return TimedPrediction(values, 0.001, 0.002)


class _MiniPredictor:
    def __init__(self, model_key: str) -> None:
        self.model_key = model_key
        self.run_id = f"{model_key}_frozen"
        self.execution_device = "cpu"
        self.smoke_calls = 0

    def smoke_compare_stored(self, validation_rows: pd.DataFrame, *, atol: float) -> None:
        assert set(validation_rows["record_id"]) <= {"validation-0", "validation-1", "validation-2", "validation-3", "validation-4", "validation-5", "validation-6", "validation-7"}
        assert atol == pytest.approx(1.0e-6)
        self.smoke_calls += 1

    def predict_p_approved_timed(self, frame: pd.DataFrame) -> TimedPrediction:
        values = (0.02 * pd.to_numeric(frame["loan_type"]).to_numpy(dtype=np.float64)
                  + 0.001 * pd.to_numeric(frame["income"]).to_numpy(dtype=np.float64)
                  + 0.03 * pd.to_numeric(frame["occupancy_type"]).to_numpy(dtype=np.float64))
        return TimedPrediction(values, 0.0001, 0.0002)


def _frame(value: str, rows: int = 4) -> pd.DataFrame:
    return pd.DataFrame({column: [value] * rows for column in FEATURE_COLUMNS},
                        columns=FEATURE_COLUMNS, dtype=object)


def _checkpoint_fixture(
    tmp_path: Path, *, replicate_count: int = 4, chunk_size: int = 2
) -> tuple[Path, Path]:
    protocol = copy.deepcopy(_read_protocol(DEFAULT_PROTOCOL_PATH))
    protocol["scope"]["model_keys"] = ["model_a"]
    protocol["scope"]["ordered_players"] = [
        {"key": "application", "raw_fields": list(FEATURE_COLUMNS[:7])},
        {"key": "terms", "raw_fields": list(FEATURE_COLUMNS[7:])},
    ]
    protocol["eligible_final_runs"] = {"model_a": "model_a_frozen"}
    protocol["sampling"]["explanation"]["n_rows"] = 4
    protocol["sampling"]["background"]["pool_rows"] = 4
    protocol["smoke"]["rows"] = 2
    protocol["execution"]["coalition_batch_size"] = 2
    protocol["uncertainty"]["bootstrap_replicates"] = replicate_count
    protocol["uncertainty"]["checkpoint"]["chunk_size"] = chunk_size
    protocol["uncertainty"]["checkpoint"]["chunk_count"] = replicate_count // chunk_size
    protocol["uncertainty"]["undefined_metrics"]["minimum_valid_replicates"] = 1
    protocol["toy_fixtures"]["production_bootstrap_gate"]["expected_shape"] = [
        replicate_count,
        4,
    ]

    split_dir = tmp_path / "split_v1"
    split_dir.mkdir()
    for partition in ("train", "validation"):
        rows = []
        for index in range(8):
            rows.append({
                "record_id": f"{partition}-{index}", "y_approved": str(index // 4),
                "loan_type": str(index % 4 + 1), "loan_purpose": "1", "lien_status": "1",
                "reverse_mortgage": "2", "open-end_line_of_credit": "2",
                "business_or_commercial_purpose": "2", "income": str(50 + 5 * index),
                "debt_to_income_ratio": "42", "loan_amount": str(100 + 10 * index),
                "loan_to_value_ratio": "75", "loan_term": "360",
                "occupancy_type": str(index % 3 + 1), "preapproval": "1",
                "interest_only_payment": "2", "balloon_payment": "2",
            })
        pd.DataFrame(rows, columns=["record_id", "y_approved", *FEATURE_COLUMNS]).to_csv(
            split_dir / f"{partition}.csv", index=False
        )
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    return protocol_path, split_dir


def test_locked_sampling_is_materialized_without_hashes() -> None:
    observed = prepare_sampling_inputs(_read_protocol(DEFAULT_PROTOCOL_PATH))
    assert observed.explanation_X.shape == (256, 15)
    assert observed.paired_donor_X.shape == (256, 15)
    assert observed.sample_output["partition"].eq("validation").all()
    assert observed.sample_output["seed"].eq(4201).all()
    assert observed.background_count == 128
    assert observed.donors_output["background_partition"].eq("train").all()
    assert observed.donors_output["assignment_seed"].eq(4203).all()


def test_coalition_engine_batches_masks_without_changing_canonical_order() -> None:
    players = (
        {"key": "first", "raw_fields": list(FEATURE_COLUMNS[:7])},
        {"key": "second", "raw_fields": list(FEATURE_COLUMNS[7:])},
    )
    cube, timing = build_coalition_prediction_cube(
        _TimedFakePredictor(), _frame("E"), _frame("D"), players, coalition_batch_size=2
    )
    assert cube.shape == (4, 4)
    np.testing.assert_allclose(cube, np.asarray([[0.1] * 4, [0.3] * 4, [0.4] * 4, [0.6] * 4]))
    assert timing.coalition_construction_seconds >= 0.0
    assert timing.frozen_preprocessing_seconds == pytest.approx(0.002)
    assert timing.frozen_inference_seconds == pytest.approx(0.004)


def test_cross_model_stability_has_locked_sixty_row_order() -> None:
    model_keys = ("lr", "rf", "xgboost", "plain_mlp", "realmlp", "tabmmini")
    base = np.linspace(0.01, 0.12, 12, dtype=np.float64)
    normalized = {
        key: {"gsv": np.roll(base, index) / base.sum(),
              "shapley_lorenz": np.roll(base[::-1], index) / base.sum()}
        for index, key in enumerate(model_keys)
    }
    rows = _cross_model_rows(normalized, analysis_run_id="run", model_keys=model_keys,
                             top_k=3, tie_atol=1.0e-10)
    assert len(rows) == 60
    assert rows[0]["metric"] == "spearman"
    assert rows[-1]["method"] == "shapley_lorenz"


def test_runner_refuses_existing_destination_before_loading_models(tmp_path: Path) -> None:
    destination = tmp_path / "existing"
    destination.mkdir()
    calls: list[str] = []

    def forbidden_loader(model_key: str, **kwargs):
        calls.append(model_key)
        raise AssertionError("predictor loader must not be called")

    with pytest.raises(RQ2RunnerError, match="destination already exists"):
        run_rq2(output_directory=destination, predictor_loader=forbidden_loader)
    assert calls == []


def test_runner_rejects_a_batch_size_that_differs_from_protocol(tmp_path: Path) -> None:
    destination = tmp_path / "wrong-batch"

    with pytest.raises(RQ2RunnerError, match="coalition batch size must equal"):
        run_rq2(
            output_directory=destination,
            coalition_batch_size=8,
            predictor_loader=lambda model_key, **kwargs: _MiniPredictor(model_key),
        )

    assert not destination.exists()


def test_runner_reports_the_failing_model_and_creates_no_fake_model_result(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "failed-run"

    def failing_loader(model_key: str, **kwargs):
        raise RuntimeError("fixture load failure")

    with pytest.raises(
        RQ2RunnerError,
        match=r"model 'lr' failed during frozen predictor load: fixture load failure",
    ):
        run_rq2(output_directory=destination, predictor_loader=failing_loader)

    assert destination.is_dir()
    status = pd.read_json(destination / "models" / "lr" / "status.json", typ="series")
    assert status["status"] == "FAILED"
    assert status["stage"] == "frozen predictor load"
    assert not list(destination.rglob("cube.npy"))
    assert not list(destination.rglob("importance.csv"))
    assert not list(destination.rglob("bootstrap.csv"))


def test_runner_rejects_a_non_cpu_predictor_before_smoke(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "wrong-device"
    predictor = _MiniPredictor("lr")
    predictor.execution_device = "cuda:0"

    with pytest.raises(RQ2RunnerError, match="predictor device differs for lr"):
        run_rq2(
            output_directory=destination,
            predictor_loader=lambda model_key, **kwargs: predictor,
    )

    assert predictor.smoke_calls == 0
    status = pd.read_json(destination / "models" / "lr" / "status.json", typ="series")
    assert status["status"] == "FAILED"
    assert status["stage"] == "frozen predictor load"


def test_interrupted_bootstrap_resumes_without_reinference_or_recomputing_valid_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol_path, split_dir = _checkpoint_fixture(tmp_path)
    destination = tmp_path / "resume-run"
    messages: list[str] = []

    def interrupt_after_first_chunk(model_key: str, chunk_id: int) -> None:
        assert model_key == "model_a"
        if chunk_id == 0:
            raise RuntimeError("simulated interruption")

    with pytest.raises(RQ2RunnerError, match="simulated interruption"):
        run_rq2(
            protocol_path=protocol_path,
            split_dir=split_dir,
            output_directory=destination,
            coalition_batch_size=2,
            predictor_loader=lambda model_key, **kwargs: _MiniPredictor(model_key),
            progress_callback=messages.append,
            bootstrap_chunk_hook=interrupt_after_first_chunk,
        )

    checkpoint_dir = destination / "models" / "model_a" / "bootstrap_checkpoints"
    assert (checkpoint_dir / "chunk_000.npz").is_file()
    assert (checkpoint_dir / "chunk_000.json").is_file()
    assert not (checkpoint_dir / "chunk_001.npz").exists()
    failed_status = json.loads(
        (destination / "models" / "model_a" / "status.json").read_text(encoding="utf-8")
    )
    assert failed_status["status"] == "FAILED"
    assert failed_status["completed_chunks"] == 1

    original_bootstrap = runner_module.bootstrap_cached_agreement
    bootstrap_calls = 0

    def counted_bootstrap(*args, **kwargs):
        nonlocal bootstrap_calls
        bootstrap_calls += 1
        return original_bootstrap(*args, **kwargs)

    monkeypatch.setattr(runner_module, "bootstrap_cached_agreement", counted_bootstrap)

    def forbidden_loader(model_key: str, **kwargs):
        raise AssertionError("resume must not load a predictor when the cube cache is valid")

    result = run_rq2(
        protocol_path=protocol_path,
        split_dir=split_dir,
        output_directory=destination,
        coalition_batch_size=2,
        resume=True,
        predictor_loader=forbidden_loader,
        progress_callback=messages.append,
    )
    assert result.validation_status == "PASS"
    assert bootstrap_calls == 1
    assert any("event=bootstrap_chunk_skipped" in message for message in messages)
    assert any("event=bootstrap_chunk_passed" in message for message in messages)
    assert any("event=run_passed" in message for message in messages)
    passed_status = json.loads(
        (destination / "models" / "model_a" / "status.json").read_text(encoding="utf-8")
    )
    assert passed_status["status"] == "PASS"
    assert passed_status["completed_chunks"] == 2


@pytest.mark.parametrize("corruption", ["identity", "payload", "dtype", "unexpected"])
def test_resume_rejects_wrong_or_corrupt_checkpoint(
    tmp_path: Path, corruption: str,
) -> None:
    protocol_path, split_dir = _checkpoint_fixture(tmp_path)
    destination = tmp_path / f"corrupt-{corruption}"

    def interrupt_after_first_chunk(model_key: str, chunk_id: int) -> None:
        if chunk_id == 0:
            raise RuntimeError("simulated interruption")

    with pytest.raises(RQ2RunnerError, match="simulated interruption"):
        run_rq2(
            protocol_path=protocol_path,
            split_dir=split_dir,
            output_directory=destination,
            coalition_batch_size=2,
            predictor_loader=lambda model_key, **kwargs: _MiniPredictor(model_key),
            progress_callback=None,
            bootstrap_chunk_hook=interrupt_after_first_chunk,
        )

    checkpoint_dir = destination / "models" / "model_a" / "bootstrap_checkpoints"
    if corruption == "identity":
        metadata_path = checkpoint_dir / "chunk_000.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["model_run_id"] = "wrong_frozen_run"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        expected_error = "identity differs"
    elif corruption == "payload":
        (checkpoint_dir / "chunk_000.npz").write_bytes(b"not-an-npz")
        expected_error = "Cannot read bootstrap checkpoint"
    elif corruption == "dtype":
        npz_path = checkpoint_dir / "chunk_000.npz"
        with np.load(npz_path, allow_pickle=False) as payload:
            arrays = {key: np.asarray(payload[key], dtype=np.float32) for key in payload.files}
        np.savez(npz_path, **arrays)
        expected_error = "payload is invalid"
    else:
        (checkpoint_dir / "unexpected.txt").write_text("stale", encoding="utf-8")
        expected_error = "contains unexpected entries"

    with pytest.raises(RQ2RunnerError, match=expected_error):
        run_rq2(
            protocol_path=protocol_path,
            split_dir=split_dir,
            output_directory=destination,
            coalition_batch_size=2,
            resume=True,
            predictor_loader=lambda model_key, **kwargs: pytest.fail(
                "corrupt checkpoint must fail before predictor loading"
            ),
            progress_callback=None,
        )


def test_mini_runner_writes_only_the_minimal_package(tmp_path: Path) -> None:
    protocol = copy.deepcopy(_read_protocol(DEFAULT_PROTOCOL_PATH))
    protocol["scope"]["model_keys"] = ["model_a", "model_b"]
    protocol["scope"]["ordered_players"] = [
        {"key": "application", "raw_fields": list(FEATURE_COLUMNS[:7])},
        {"key": "terms", "raw_fields": list(FEATURE_COLUMNS[7:])},
    ]
    protocol["eligible_final_runs"] = {"model_a": "model_a_frozen", "model_b": "model_b_frozen"}
    protocol["sampling"]["explanation"]["n_rows"] = 4
    protocol["sampling"]["background"]["pool_rows"] = 4
    protocol["smoke"]["rows"] = 2
    protocol["execution"]["coalition_batch_size"] = 2
    protocol["uncertainty"]["bootstrap_replicates"] = 3
    protocol["uncertainty"]["checkpoint"]["chunk_size"] = 1
    protocol["uncertainty"]["checkpoint"]["chunk_count"] = 3
    protocol["uncertainty"]["undefined_metrics"]["minimum_valid_replicates"] = 1
    protocol["toy_fixtures"]["production_bootstrap_gate"]["expected_shape"] = [3, 4]

    split_dir = tmp_path / "split_v1"
    split_dir.mkdir()
    for partition in ("train", "validation"):
        rows = []
        for index in range(8):
            row = {"record_id": f"{partition}-{index}", "y_approved": str(index // 4),
                   "loan_type": str(index % 4 + 1), "loan_purpose": "1", "lien_status": "1",
                   "reverse_mortgage": "2", "open-end_line_of_credit": "2",
                   "business_or_commercial_purpose": "2", "income": str(50 + 5 * index),
                   "debt_to_income_ratio": "42", "loan_amount": str(100 + 10 * index),
                   "loan_to_value_ratio": "75", "loan_term": "360",
                   "occupancy_type": str(index % 3 + 1), "preapproval": "1",
                   "interest_only_payment": "2", "balloon_payment": "2"}
            rows.append(row)
        pd.DataFrame(rows, columns=["record_id", "y_approved", *FEATURE_COLUMNS]).to_csv(
            split_dir / f"{partition}.csv", index=False
        )
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    destination = tmp_path / "mini-run"
    result = run_rq2(protocol_path=protocol_path, split_dir=split_dir,
                     output_directory=destination, coalition_batch_size=2,
                     predictor_loader=lambda model_key, **kwargs: _MiniPredictor(model_key))
    assert result.validation_status == "PASS"
    assert {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()} == {
        "sample.csv", "donors.csv", "bootstrap_indices.npy", "runtime.csv",
        "cross_model_stability.csv", "models/model_a/cube.npy",
        "models/model_a/gsv_local.npy", "models/model_a/importance.csv",
        "models/model_a/bootstrap.csv", "models/model_a/status.json",
        "models/model_a/bootstrap_checkpoints/chunk_000.npz",
        "models/model_a/bootstrap_checkpoints/chunk_000.json",
        "models/model_a/bootstrap_checkpoints/chunk_001.npz",
        "models/model_a/bootstrap_checkpoints/chunk_001.json",
        "models/model_a/bootstrap_checkpoints/chunk_002.npz",
        "models/model_a/bootstrap_checkpoints/chunk_002.json",
        "models/model_b/cube.npy",
        "models/model_b/gsv_local.npy", "models/model_b/importance.csv",
        "models/model_b/bootstrap.csv", "models/model_b/status.json",
        "models/model_b/bootstrap_checkpoints/chunk_000.npz",
        "models/model_b/bootstrap_checkpoints/chunk_000.json",
        "models/model_b/bootstrap_checkpoints/chunk_001.npz",
        "models/model_b/bootstrap_checkpoints/chunk_001.json",
        "models/model_b/bootstrap_checkpoints/chunk_002.npz",
        "models/model_b/bootstrap_checkpoints/chunk_002.json",
    }

    cube = np.load(destination / "models" / "model_a" / "cube.npy", allow_pickle=False)
    gsv_local = np.load(
        destination / "models" / "model_a" / "gsv_local.npy", allow_pickle=False
    )
    expected_gsv = gsv_from_prediction_cube(cube, player_count=2)
    expected_sl = shapley_lorenz_from_prediction_cube(cube, player_count=2)
    np.testing.assert_allclose(gsv_local, expected_gsv["local_contributions"])

    importance = pd.read_csv(destination / "models" / "model_a" / "importance.csv")
    for method, expected in (("gsv", expected_gsv), ("shapley_lorenz", expected_sl)):
        observed = importance.loc[importance["method"].eq(method)].sort_values("player_index")
        np.testing.assert_allclose(observed["global_importance"], expected["global_importance"])
        np.testing.assert_allclose(
            observed["normalized_importance"], expected["normalized_importance"]
        )
        np.testing.assert_allclose(
            observed["rank"], rank_importance(expected["normalized_importance"])
        )

    bootstrap_indices = np.load(destination / "bootstrap_indices.npy", allow_pickle=False)
    expected_bootstrap = bootstrap_cached_agreement(
        expected_gsv["local_contributions"],
        cube,
        bootstrap_indices,
        player_count=2,
        minimum_valid_replicates=1,
    )
    expected_point = agreement_metrics(
        expected_gsv["normalized_importance"],
        expected_sl["normalized_importance"],
        top_k=3,
    )
    bootstrap = pd.read_csv(destination / "models" / "model_a" / "bootstrap.csv")
    for row in bootstrap.itertuples(index=False):
        interval = expected_bootstrap["confidence_intervals_95_percentile"][row.metric]
        assert row.point_estimate == pytest.approx(expected_point[row.metric])
        assert row.ci_lower == pytest.approx(interval["lower"])
        assert row.ci_upper == pytest.approx(interval["upper"])
        assert row.valid_replicates == interval["valid_replicates"]
        assert row.invalid_replicates == interval["invalid_replicates"]
