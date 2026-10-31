"""Synthetic XGB-08 boundary tests; no real XGBoost model is inferred."""

from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.dataset_b.evaluation import xgb_validation_threshold as compatibility
from src.dataset_b.models.classical.xgboost import validation


def test_legacy_validation_module_delegates_to_canonical():
    assert compatibility.XGBValidationError is validation.XGBValidationError
    assert compatibility.run_xgb_validation_threshold is validation.run_xgb_validation_threshold


class CountingPipeline:
    def __init__(self):
        self.predict_calls = 0
        self.fit_calls = 0

    def fit(self, *_args, **_kwargs):
        self.fit_calls += 1
        raise AssertionError("XGB-08 must not fit")

    def predict_proba(self, frame):
        self.predict_calls += 1
        approved = np.linspace(0.1, 0.9, len(frame))
        return np.column_stack((1 - approved, approved))


def test_validation_inference_occurs_once_and_never_fits():
    pipeline = CountingPipeline()
    values = validation._predict_validation_once(pipeline, pd.DataFrame({"x": [1, 2, 3]}))
    assert pipeline.predict_calls == 1
    assert pipeline.fit_calls == 0
    assert values.tolist() == pytest.approx([0.1, 0.5, 0.9])


def test_fixed_probability_sum_tolerance_accepts_complement_error_without_changing_probabilities():
    approved = np.asarray([0.0001, 0.8], dtype=np.float32)
    raw_probabilities = np.column_stack((np.float32(1.0) - approved, approved))

    class Float32Pipeline:
        def predict_proba(self, _frame):
            return raw_probabilities

    assert abs(float(raw_probabilities[0, 0]) + float(raw_probabilities[0, 1]) - 1.0) > 1e-12
    values = validation._predict_validation_once(Float32Pipeline(), pd.DataFrame({"x": [1, 2]}))
    assert values.dtype == np.float64
    assert np.array_equal(values, raw_probabilities[:, 1].astype(np.float64))


def test_materially_invalid_float32_probability_sum_is_rejected():
    class InvalidFloat32Pipeline:
        def predict_proba(self, _frame):
            return np.asarray([[0.5, 0.6]], dtype=np.float32)

    with pytest.raises(validation.XGBValidationError, match="do not sum to one"):
        validation._predict_validation_once(InvalidFloat32Pipeline(), pd.DataFrame({"x": [1]}))


def test_tie_rule_prefers_nearest_half_then_lower():
    selected, maximizing = validation._select_threshold(
        [
            {"threshold": 0.4, "balanced_accuracy": 0.8},
            {"threshold": 0.6, "balanced_accuracy": 0.8},
            {"threshold": 0.7, "balanced_accuracy": 0.7},
        ]
    )
    assert len(maximizing) == 2
    assert selected["threshold"] == 0.4


def _run_synthetic(tmp_path, monkeypatch):
    run_id = "xgb_run"
    run_dir = tmp_path / run_id
    (run_dir / "checkpoints").mkdir(parents=True)
    for relative in ("manifest.json", "cv_results.json", "full_train_result.json", "checkpoints/fitted_pipeline.joblib"):
        (run_dir / relative).write_text("{}\n", encoding="utf-8")
    split_manifest = tmp_path / "split.json"
    split_manifest.write_text("{}\n", encoding="utf-8")
    feature = tmp_path / "feature.yaml"
    feature.write_text("{}\n", encoding="utf-8")
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    (split_dir / "validation.csv").write_text("source-v1\n", encoding="utf-8")
    ids = ("r1", "r2", "r3", "r4")
    labels = np.asarray([0, 0, 1, 1], dtype=np.int8)

    class Audit:
        def as_dict(self):
            return {"status": "PASS", "retained_rows": 4}

    gated = SimpleNamespace(
        X=pd.DataFrame({"x": range(4)}),
        y=pd.Series(labels),
        record_ids=pd.Series(ids),
        quality_audit=Audit(),
    )
    loaded = []

    class Loader:
        def __init__(self, **_kwargs):
            pass

        def load(self, partition):
            loaded.append(partition)
            assert partition == "validation"
            return gated

    class Gate:
        def apply(self, value):
            return value

    pipeline = CountingPipeline()
    training = {
        "run_id": run_id,
        "model_key": "xgb",
        "reporting_key": "xgboost",
        "feature_version": "feature_set_v1",
        "split_version": "split_v1",
        "preprocessing_version": "dataset_b_preprocessing_v2",
        "preprocessing_manifest": str((run_dir / "manifest.json").resolve()),
        "full_train_effective_parameters": {"scale_pos_weight": 0.5},
        "validation_accessed": False,
        "test_accessed": False,
    }
    monkeypatch.setattr(validation, "EXPECTED_VALIDATION_ROWS", 4)
    monkeypatch.setattr(validation, "CANDIDATE_THRESHOLDS", (0.25, 0.5, 0.75))
    monkeypatch.setattr(validation, "DatasetBSplitLoader", Loader)
    monkeypatch.setattr(validation, "DatasetBInputQualityGate", Gate)
    monkeypatch.setattr(validation, "validate_xgb_final_training", lambda *_a, **_k: training)
    monkeypatch.setattr(validation, "_load_pipeline", lambda _path, _weight: pipeline)
    result = validation.run_xgb_validation_threshold(
        run_id,
        run_dir=run_dir,
        split_manifest_path=split_manifest,
        split_dir=split_dir,
        feature_contract_path=feature,
    )
    return {
        "result": result,
        "run_id": run_id,
        "run_dir": run_dir,
        "split_manifest": split_manifest,
        "split_dir": split_dir,
        "feature": feature,
        "pipeline": pipeline,
        "loaded": loaded,
        "training": training,
    }


def test_full_runner_self_validates_once_without_fit_or_test_access(tmp_path, monkeypatch):
    state = _run_synthetic(tmp_path, monkeypatch)
    assert state["result"]["status"] == "PASS"
    assert state["result"]["rows"] == 4
    assert state["pipeline"].predict_calls == 1
    assert state["pipeline"].fit_calls == 0
    assert set(state["loaded"]) == {"validation"}
    prediction = json.loads((state["run_dir"] / "validation/prediction_manifest.json").read_text(encoding="utf-8"))
    selection = json.loads((state["run_dir"] / "validation/threshold_selection.json").read_text(encoding="utf-8"))
    assert prediction["execution"] == {"model_fit_calls": 0, "preprocessing_fit_calls": 0, "predict_proba_calls": 1}
    assert prediction["scale_pos_weight"] == 0.5
    assert selection["baseline_threshold_reused"] is False
    assert selection["selected_threshold"] != 0.744


def test_training_lineage_mismatch_is_rejected(tmp_path, monkeypatch):
    run_id = "xgb_run"
    run_dir = tmp_path / run_id
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints/fitted_pipeline.joblib").write_bytes(b"synthetic")
    training = {
        "run_id": "other_run",
        "model_key": "xgb",
        "reporting_key": "xgboost",
        "feature_version": "feature_set_v1",
        "split_version": "split_v1",
        "preprocessing_version": "dataset_b_preprocessing_v2",
        "full_train_effective_parameters": {"scale_pos_weight": 0.5},
        "validation_accessed": False,
        "test_accessed": False,
    }
    monkeypatch.setattr(validation, "validate_xgb_final_training", lambda *_a, **_k: training)
    with pytest.raises(validation.XGBValidationError, match="training run_id lineage"):
        validation.run_xgb_validation_threshold(run_id, run_dir=run_dir)


@pytest.mark.parametrize("tamper", ["reorder", "duplicate"])
def test_reordered_or_duplicate_validation_ids_are_rejected(tmp_path, monkeypatch, tamper):
    state = _run_synthetic(tmp_path, monkeypatch)
    predictions_path = state["run_dir"] / "validation/validation_predictions.csv"
    with predictions_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if tamper == "reorder":
        rows[0], rows[1] = rows[1], rows[0]
    else:
        rows[1]["record_id"] = rows[0]["record_id"]
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=validation.PREDICTION_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(validation.XGBValidationError, match="align|population"):
        validation.validate_xgb_validation_threshold(
            state["run_dir"] / "validation",
            run_dir=state["run_dir"],
            run_id=state["run_id"],
            split_manifest_path=state["split_manifest"],
            split_dir=state["split_dir"],
            feature_contract_path=state["feature"],
        )


def test_source_has_no_test_access_fit_or_baseline_threshold():
    source = inspect.getsource(validation)
    assert '.load("test")' not in source
    assert ".fit(" not in source
    assert ".fit_transform(" not in source
    assert "0.744" not in source
    assert "rf_validation_threshold" not in source
    assert "lr_validation_threshold" not in source
