from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.dataset_b.gsv_sl import predictors
from src.dataset_b.gsv_sl import cpu_inference
from src.dataset_b.gsv_sl.predictors import (
    FrozenPredictorError,
    available_model_keys,
    load_frozen_predictor,
)
FEATURE_COLUMNS = predictors.FEATURE_COLUMNS


def _raw_row(**changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "loan_type": "1",
        "loan_purpose": "1",
        "lien_status": "1",
        "reverse_mortgage": "2",
        "open-end_line_of_credit": "2",
        "business_or_commercial_purpose": "2",
        "income": "100",
        "debt_to_income_ratio": "42",
        "loan_amount": "200",
        "loan_to_value_ratio": "75",
        "loan_term": "360",
        "occupancy_type": "1",
        "preapproval": "1",
        "interest_only_payment": "2",
        "balloon_payment": "2",
    }
    row.update(changes)
    return row


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _raw_row(reverse_mortgage="1111"),
            _raw_row(
                reverse_mortgage="Exempt",
                income=120.0,
                loan_amount=250.0,
                loan_to_value_ratio=80.0,
                loan_term=360.0,
            ),
        ],
        columns=FEATURE_COLUMNS,
        dtype=object,
    )


def _materialize_preflight(
    artifact_root: Path, model_key: str, *, feature_version: str = "feature_set_v1"
) -> Path:
    run_id, relative_root, relative_bundle = predictors._RUNS[model_key]
    run_root = artifact_root / relative_root
    bundle = run_root / relative_bundle
    bundle.parent.mkdir(parents=True, exist_ok=True)
    bundle.write_bytes(b"trusted-test-bundle")
    (run_root / "FROZEN").write_text("FROZEN\n", encoding="utf-8")
    reporting = run_root / "reporting"
    reporting.mkdir()
    (reporting / "model_summary.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "model_key": model_key,
                "reporting_status": "COMPLETE",
                "feature_version": feature_version,
                "split_version": "split_v1",
            }
        ),
        encoding="utf-8",
    )
    return run_root


class _FakePipeline:
    classes_ = np.asarray([0, 1])

    def __init__(self, probabilities: list[float] | None = None) -> None:
        self.probabilities = probabilities or [0.25, 0.75]
        self.last_frame: pd.DataFrame | None = None

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        self.last_frame = frame.copy(deep=True)
        approved = np.asarray(self.probabilities[: len(frame)], dtype=np.float64)
        return np.column_stack((1.0 - approved, approved))


class _FakePreprocessing:
    def __init__(self) -> None:
        self.last_frame: pd.DataFrame | None = None

    def transform(self, frame: pd.DataFrame):
        self.last_frame = frame.copy(deep=True)
        return frame


class _FakeEstimator:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        approved = np.linspace(0.3, 0.7, len(frame), dtype=np.float64)
        return np.column_stack((1.0 - approved, approved))


def test_protocol_order_is_the_common_public_model_order() -> None:
    assert available_model_keys() == (
        "lr",
        "rf",
        "xgboost",
        "plain_mlp",
        "realmlp",
        "tabmmini",
    )


def test_plain_mlp_cpu_loader_rehydrates_checkpoint_without_fit_or_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from src.dataset_b.models.plain_mlp import (
        EXPECTED_INPUT_WIDTH,
        LOCKED_DTYPE,
        PlainMLPClassifier,
        _PlainMLPModule,
    )

    configuration = {
        "activation": "ReLU",
        "batch_size": 2,
        "deterministic_algorithms": True,
        "device": "cuda:0",
        "dropout": 0.1,
        "epochs": 1,
        "hidden_layers": (256, 128, 64),
        "learning_rate": 0.001,
        "random_state": 42,
        "weight_decay": 0.0001,
    }
    source = _PlainMLPModule(EXPECTED_INPUT_WIDTH, (256, 128, 64), 0.1).to(
        dtype=LOCKED_DTYPE
    )
    bundle = {
        "preprocessing": "frozen-preprocessing",
        "scope": {
            "run_id": "plain-run",
            "stage": "full_train_final_estimator",
            "validation_accessed": False,
            "test_accessed": False,
        },
        "model_checkpoint": {
            "payload_version": 2,
            "model_state_dict": source.state_dict(),
            "schema": {
                "raw_feature_count": 15,
                "semantic_feature_count": 19,
                "encoded_feature_count": 62,
                "input_width": 62,
                "classes": [0, 1],
            },
            "runtime_policy": {
                "requested_device": "cuda:0",
                "device_index": 0,
                "dtype": "float64",
                "allow_cpu_fallback": False,
            },
            "model_policy": {
                "hidden_layers": [256, 128, 64],
                "activation": "ReLU",
                "loss": "BCEWithLogitsLoss",
                "positive_class": "approved",
                "pos_weight_strategy": "fit_population_n_denied_over_n_approved",
                "pos_weight": 1.0,
            },
            "configuration": configuration,
            "training_history": [0.5],
        },
    }
    monkeypatch.setattr(cpu_inference.joblib, "load", lambda path: bundle)
    monkeypatch.setattr(
        PlainMLPClassifier,
        "fit",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("fit called")),
    )

    preprocessing, estimator = cpu_inference.load_plain_mlp_cpu(
        "unused.joblib", expected_run_id="plain-run"
    )

    assert preprocessing == "frozen-preprocessing"
    assert estimator.runtime_device_.type == "cpu"
    assert all(parameter.device.type == "cpu" for parameter in estimator.model_.parameters())
    assert all(parameter.dtype == torch.float64 for parameter in estimator.model_.parameters())


def test_realmlp_cpu_loader_retargets_only_the_loaded_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interface = SimpleNamespace(device="cuda:0", config={"device": "cuda:0"})
    backend = SimpleNamespace(
        device="cuda:0", alg_interface_=interface, cv_alg_interface_=interface
    )
    estimator = SimpleNamespace(device="cuda:0", backend_=backend)
    bundle = {
        "estimator": estimator,
        "preprocessing": "frozen-preprocessing",
        "scope": "all_immutable_train_rows",
    }
    monkeypatch.setattr(cpu_inference.joblib, "load", lambda path: bundle)

    preprocessing, observed = cpu_inference.load_realmlp_cpu("unused.joblib")

    assert preprocessing == "frozen-preprocessing"
    assert observed is estimator
    assert estimator.device == "cpu"
    assert backend.device == "cpu"
    assert interface.device == "cpu"
    assert interface.config["device"] == "cpu"


def test_classical_factory_preflights_and_returns_float64_p_approved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _materialize_preflight(tmp_path, "lr")
    pipeline = _FakePipeline()
    monkeypatch.setattr(predictors.joblib, "load", lambda path: pipeline)

    adapter = load_frozen_predictor("lr", artifact_root=tmp_path)
    timed = adapter.predict_p_approved_timed(_valid_frame())
    observed = timed.probabilities

    np.testing.assert_array_equal(observed, np.asarray([0.25, 0.75]))
    assert observed.dtype == np.float64
    assert timed.preprocessing_seconds >= 0.0
    assert timed.inference_seconds >= 0.0
    assert adapter.run_id == "lr_final_15f_20261002_01"
    assert adapter.execution_device == "cpu"
    assert pipeline.last_frame is not None
    assert pipeline.last_frame["reverse_mortgage"].tolist() == ["Exempt", "Exempt"]
    assert pipeline.last_frame["income"].tolist() == [100.0, 120.0]


def test_smoke_comparison_matches_frozen_validation_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = _materialize_preflight(tmp_path, "lr")
    validation = run_root / "validation"
    validation.mkdir()
    pd.DataFrame(
        {
            "record_id": ["validation-1", "validation-2"],
            "p_approved": [0.25, 0.75],
        }
    ).to_csv(validation / "validation_predictions.csv", index=False)
    monkeypatch.setattr(predictors.joblib, "load", lambda path: _FakePipeline())
    adapter = load_frozen_predictor("lr", artifact_root=tmp_path)
    rows = _valid_frame().copy()
    rows.insert(0, "record_id", ["validation-1", "validation-2"])

    adapter.smoke_compare_stored(rows, atol=1.0e-12)

    pd.DataFrame(
        {"record_id": ["validation-1", "validation-2"], "p_approved": [0.25, 0.70]}
    ).to_csv(validation / "validation_predictions.csv", index=False)
    with pytest.raises(FrozenPredictorError, match="differ from frozen stored predictions"):
        adapter.smoke_compare_stored(rows, atol=1.0e-12)


@pytest.mark.parametrize(
    ("model_key", "loader_name"),
    [
        ("plain_mlp", "_load_plain_mlp_components"),
        ("realmlp", "_load_realmlp_components"),
    ],
)
def test_neural_joblib_routes_use_frozen_preprocessing_then_predict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_key: str,
    loader_name: str,
) -> None:
    _materialize_preflight(tmp_path, model_key)
    preprocessing = _FakePreprocessing()
    estimator = _FakeEstimator()
    if model_key == "plain_mlp":
        monkeypatch.setattr(
            predictors,
            loader_name,
            lambda path, expected_run_id: (preprocessing, estimator),
        )
    else:
        monkeypatch.setattr(
            predictors, loader_name, lambda path: (preprocessing, estimator)
        )

    timed = load_frozen_predictor(
        model_key, artifact_root=tmp_path
    ).predict_p_approved_timed(_valid_frame())
    observed = timed.probabilities

    np.testing.assert_allclose(observed, [0.3, 0.7])
    assert observed.dtype == np.float64
    assert timed.preprocessing_seconds >= 0.0
    assert timed.inference_seconds >= 0.0
    assert preprocessing.last_frame is not None
    assert tuple(preprocessing.last_frame.columns) == FEATURE_COLUMNS


def test_tabmmini_route_batches_on_the_bundle_setting_and_returns_float64(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    _materialize_preflight(tmp_path, "tabmmini")

    class _Native:
        x_num = torch.zeros((2, 4), dtype=torch.float32)
        x_cat = torch.zeros((2, 15), dtype=torch.int64)

    class _Preprocessing:
        def transform(self, frame: pd.DataFrame) -> _Native:
            return _Native()

    class _Model:
        eval_calls = 0
        inference_thread_counts: list[int] = []

        def eval(self) -> None:
            self.eval_calls += 1

        def predict_approved_probability(self, x_num, x_cat):
            assert x_num.device.type == "cpu"
            assert x_cat.device.type == "cpu"
            self.inference_thread_counts.append(torch.get_num_threads())
            return torch.full((len(x_num),), 0.6, dtype=torch.float32)

    model = _Model()
    monkeypatch.setattr(
        predictors,
        "_load_tabmmini_components",
        lambda spec, device: (_Preprocessing(), model, "cpu", 1),
    )

    adapter = load_frozen_predictor(
        "tabmmini", artifact_root=tmp_path, tabmmini_device="cpu"
    )
    original_thread_count = torch.get_num_threads()
    timed = adapter.predict_p_approved_timed(_valid_frame())
    observed = timed.probabilities

    np.testing.assert_allclose(observed, [0.6, 0.6], rtol=0.0, atol=1.0e-7)
    assert observed.dtype == np.float64
    assert timed.preprocessing_seconds >= 0.0
    assert timed.inference_seconds >= 0.0
    assert adapter.device == "cpu"
    assert adapter.execution_device == "cpu"
    assert adapter.batch_size == 1
    assert model.eval_calls == 1
    assert model.inference_thread_counts == [1, 1]
    assert torch.get_num_threads() == original_thread_count


def test_preflight_rejects_missing_freeze_and_wrong_lineage(tmp_path: Path) -> None:
    run_root = _materialize_preflight(tmp_path, "rf")
    (run_root / "FROZEN").unlink()
    with pytest.raises(FrozenPredictorError, match="Frozen marker"):
        load_frozen_predictor("rf", artifact_root=tmp_path)

    _materialize_preflight(tmp_path, "xgboost", feature_version="legacy_v0_2")
    with pytest.raises(FrozenPredictorError, match="metadata differs"):
        load_frozen_predictor("xgboost", artifact_root=tmp_path)


def test_realmlp_loader_rejects_non_final_bundle_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_path = tmp_path / "fitted_model.joblib"
    bundle_path.write_bytes(b"trusted-test-bundle")
    monkeypatch.setattr(
        predictors.joblib,
        "load",
        lambda path: {
            "estimator": _FakeEstimator(),
            "preprocessing": _FakePreprocessing(),
            "scope": "cv_fold_01",
        },
    )

    with pytest.raises(FrozenPredictorError, match="scope is invalid"):
        predictors._load_realmlp_components(bundle_path)


def test_input_and_output_gates_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _materialize_preflight(tmp_path, "lr")
    pipeline = _FakePipeline(probabilities=[0.2, 1.1])
    monkeypatch.setattr(predictors.joblib, "load", lambda path: pipeline)
    adapter = load_frozen_predictor("lr", artifact_root=tmp_path)

    with pytest.raises(FrozenPredictorError, match="ordered final 15 fields"):
        adapter.predict_p_approved(_valid_frame().loc[:, reversed(FEATURE_COLUMNS)])
    with pytest.raises(FrozenPredictorError, match="invalid class probabilities"):
        adapter.predict_p_approved(_valid_frame())


def test_unknown_model_key_is_rejected_before_artifact_access(tmp_path: Path) -> None:
    with pytest.raises(FrozenPredictorError, match="Unknown RQ2 model key"):
        load_frozen_predictor("legacy_model", artifact_root=tmp_path)
