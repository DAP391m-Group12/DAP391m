"""Synthetic Validation and threshold tests for PlainMLP."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from src.dataset_b.evaluation import plain_mlp_validation as validation
from src.dataset_b.evaluation import plain_mlp_validation_threshold as threshold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_FEATURE_CONTRACT = PROJECT_ROOT / "config/dataset_b/feature_set_v1.yaml"


def _setup(tmp_path: Path, monkeypatch, probabilities=None):
    full = tmp_path / "full_train"
    (full / "checkpoints").mkdir(parents=True)
    (full / "manifest.json").write_text("{}\n", encoding="utf-8")
    (full / "full_train_result.json").write_text("{}\n", encoding="utf-8")
    (full / "checkpoints/fitted_bundle.joblib").write_bytes(b"bundle")
    split_manifest = tmp_path / "split_v1.json"
    split_manifest.write_text("{}\n", encoding="utf-8")
    split_dir = tmp_path / "split_v1"
    split_dir.mkdir()
    (split_dir / "validation.csv").write_text("record_id\nr1\nr2\nr3\nr4\n", encoding="utf-8")
    feature = tmp_path / "feature.yaml"
    feature.write_bytes(PRODUCTION_FEATURE_CONTRACT.read_bytes())
    config = tmp_path / "plain.yaml"
    config.write_text("model_version: plain_mlp_v1\n", encoding="utf-8")

    partition = SimpleNamespace(
        X=pd.DataFrame({"x": [1, 2, 3, 4]}),
        y=pd.Series([1, 0, 1, 0], dtype="int8"),
        record_ids=pd.Series(["r1", "r2", "r3", "r4"]),
    )
    loaded = []
    quality_gate_calls = []

    class Loader:
        def __init__(self, **kwargs): pass
        def load(self, name):
            loaded.append(name)
            assert name == "validation"
            return partition

    class QualityGate:
        def apply(self, raw_partition):
            quality_gate_calls.append("validation")
            return SimpleNamespace(
                X=raw_partition.X,
                y=raw_partition.y,
                record_ids=raw_partition.record_ids,
                quality_audit=SimpleNamespace(
                    as_dict=lambda: {
                        "status": "PASS",
                        "input_rows": 4,
                        "retained_rows": 4,
                        "rejected_rows": 0,
                        "rejected_record_ids": [],
                        "blank_counts": {},
                        "raw_1111_counts": {},
                        "normalized_exempt_counts": {},
                        "semantic_1111_count": 0,
                        "row_count_preserved_after_normalization": True,
                        "row_order_preserved": True,
                        "target_alignment_preserved": True,
                        "record_id_alignment_preserved": True,
                    }
                ),
            )

    class Preprocessor:
        def __init__(self): self.transform_calls = 0
        def fit(self, *args, **kwargs): raise AssertionError("Validation must not fit")
        def fit_transform(self, *args, **kwargs): raise AssertionError("Validation must not fit")
        def transform(self, frame):
            self.transform_calls += 1
            return frame

    class Estimator:
        classes_ = np.asarray([0, 1])
        def __init__(self): self.predict_calls = 0
        def fit(self, *args, **kwargs): raise AssertionError("Validation must not fit")
        def predict_proba(self, frame):
            self.predict_calls += 1
            return np.asarray(probabilities if probabilities is not None else [
                [0.2, 0.8], [0.8, 0.2], [0.4, 0.6], [0.7, 0.3]
            ])

    preprocessor, estimator = Preprocessor(), Estimator()
    estimator.quality_gate_calls = quality_gate_calls
    monkeypatch.setattr(validation, "EXPECTED_VALIDATION_ROWS", 4)
    monkeypatch.setattr(validation, "DatasetBSplitLoader", Loader)
    monkeypatch.setattr(validation, "DatasetBInputQualityGate", QualityGate)
    monkeypatch.setattr(
        validation, "validate_plain_mlp_full_train",
        lambda path, **kwargs: {
            "contract_id": "PMLP-F04-FULL-TRAIN",
            "run_id": "plain_mlp_test_run",
            "status": "PASS",
        },
    )
    monkeypatch.setattr(
        validation, "load_plain_mlp_bundle",
        lambda path: (
            preprocessor,
            estimator,
            {
                "description": "synthetic",
                "stage": "full_train_final_estimator",
                "validation_accessed": False,
                "test_accessed": False,
                "temporary_selector_state_reused": False,
            },
        ),
    )
    paths = {
        "full_train_dir": full,
        "split_manifest_path": split_manifest,
        "split_dir": split_dir,
        "feature_contract_path": feature,
        "config_path": config,
        "run_id": "plain_mlp_test_run",
    }
    return paths, loaded, preprocessor, estimator


def test_validation_predicts_once_in_order_without_fit(tmp_path, monkeypatch):
    paths, loaded, preprocessor, estimator = _setup(tmp_path, monkeypatch)
    output = tmp_path / "validation"

    manifest = validation.run_plain_mlp_validation(output, **paths)

    assert manifest["contract_id"] == "PMLP-F05-VALIDATION"
    assert estimator.predict_calls == 1
    assert preprocessor.transform_calls == 1
    assert estimator.quality_gate_calls == ["validation"]
    assert loaded == ["validation"]
    with (output / "validation_predictions.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["record_id"] for row in rows] == ["r1", "r2", "r3", "r4"]
    result = json.loads((output / "validation_result.json").read_text(encoding="utf-8"))
    assert result["execution"]["model_fit_calls"] == 0
    assert result["execution"]["preprocessing_fit_calls"] == 0
    assert result["data_boundary"]["project_test_accessed"] is False
    assert result["data_boundary"]["quality_gate"]["status"] == "PASS"
    assert result["data_boundary"]["quality_gate"]["rejected_rows"] == 0
    assert result["data_boundary"]["quality_gate"]["row_order_preserved"] is True
    assert result["data_boundary"]["quality_gate"]["target_alignment_preserved"] is True


def test_validation_stops_before_transform_when_quality_gate_rejects(tmp_path, monkeypatch):
    paths, _, preprocessor, estimator = _setup(tmp_path, monkeypatch)

    class RejectingQualityGate:
        def apply(self, partition):
            raise validation.InputQualityError("synthetic rejection")

    monkeypatch.setattr(validation, "DatasetBInputQualityGate", RejectingQualityGate)
    output = tmp_path / "validation"

    with pytest.raises(validation.PlainMLPValidationError, match="quality gate failed"):
        validation.run_plain_mlp_validation(output, **paths)

    assert preprocessor.transform_calls == 0
    assert estimator.predict_calls == 0
    assert not output.exists()


def test_artifact_only_validator_does_not_reopen_sources(tmp_path, monkeypatch):
    paths, loaded, _, _ = _setup(tmp_path, monkeypatch)
    output = tmp_path / "validation"
    manifest = validation.run_plain_mlp_validation(output, **paths)

    def forbidden(*args, **kwargs):
        raise AssertionError("Artifact-only validation must not reopen protected sources")

    monkeypatch.setattr(validation, "DatasetBSplitLoader", forbidden)
    monkeypatch.setattr(validation, "validate_plain_mlp_full_train", forbidden)
    monkeypatch.setattr(validation, "load_plain_mlp_bundle", forbidden)

    assert validation.validate_plain_mlp_validation_artifact(output) == manifest
    assert loaded == ["validation"]


def test_validation_rejects_invalid_probabilities_without_publication(tmp_path, monkeypatch):
    paths, _, _, _ = _setup(
        tmp_path, monkeypatch,
        probabilities=[[0.2, 0.8], [np.nan, 0.5], [0.4, 0.6], [0.7, 0.3]],
    )
    output = tmp_path / "validation"
    with pytest.raises(validation.PlainMLPValidationError, match="probability"):
        validation.run_plain_mlp_validation(output, **paths)
    assert not output.exists()


def test_validation_preserves_float64_probability_buffer(tmp_path, monkeypatch):
    probabilities = np.asarray(
        [[0.2, 0.8], [0.8, 0.2], [0.4, 0.6], [0.7, 0.3]], dtype=np.float64
    )
    original = probabilities.copy()
    paths, _, _, _ = _setup(tmp_path, monkeypatch, probabilities=probabilities)
    output = tmp_path / "validation"

    validation.run_plain_mlp_validation(output, **paths)

    assert probabilities.dtype == np.float64
    np.testing.assert_array_equal(probabilities, original)
    frame = pd.read_csv(output / "validation_predictions.csv")
    assert frame[["p_denied", "p_approved"]].dtypes.tolist() == [
        np.dtype("float64"),
        np.dtype("float64"),
    ]


def test_validation_accepts_complement_rounding_at_locked_tolerance(tmp_path, monkeypatch):
    paths, _, _, _ = _setup(
        tmp_path, monkeypatch,
        probabilities=[[0.2, 0.80000009], [0.8, 0.2], [0.4, 0.6], [0.7, 0.3]],
    )
    output = tmp_path / "validation"
    validation.run_plain_mlp_validation(output, **paths)
    frame = pd.read_csv(output / "validation_predictions.csv")
    assert tuple(frame.columns) == validation.PREDICTION_COLUMNS
    assert frame.loc[0, "p_denied"] == pytest.approx(0.2)
    assert frame.loc[0, "p_approved"] == pytest.approx(0.80000009)


def test_validation_rejects_probability_sum_above_locked_tolerance(tmp_path, monkeypatch):
    paths, _, _, _ = _setup(
        tmp_path, monkeypatch,
        probabilities=[[0.2, 0.80000011], [0.8, 0.2], [0.4, 0.6], [0.7, 0.3]],
    )
    with pytest.raises(validation.PlainMLPValidationError, match="probability"):
        validation.run_plain_mlp_validation(tmp_path / "validation", **paths)


def test_validation_accepts_feature_yaml_presentation_drift(tmp_path, monkeypatch):
    paths, _, _, _ = _setup(tmp_path, monkeypatch)
    output = tmp_path / "validation"
    validation.run_plain_mlp_validation(output, **paths)
    feature = paths["feature_contract_path"]
    contract = yaml.safe_load(feature.read_text(encoding="utf-8"))
    feature.write_text(
        "# presentation-only change\n" + yaml.safe_dump(contract, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )

    assert validation.validate_plain_mlp_validation(output, **paths)["status"] == "PASS"


def test_threshold_is_reproducible_and_does_not_predict_again(tmp_path, monkeypatch):
    paths, loaded, _, estimator = _setup(tmp_path, monkeypatch)
    output = tmp_path / "validation"
    validation.run_plain_mlp_validation(output, **paths)

    def forbidden(*args, **kwargs):
        raise AssertionError("Threshold selection must not load a bundle or protected features")

    monkeypatch.setattr(validation, "DatasetBSplitLoader", forbidden)
    monkeypatch.setattr(validation, "validate_plain_mlp_full_train", forbidden)
    monkeypatch.setattr(validation, "load_plain_mlp_bundle", forbidden)

    manifest = threshold.run_plain_mlp_validation_threshold(output)

    assert manifest["contract_id"] == "PMLP-F05-THRESHOLD"
    assert manifest["candidate_rows"] == 999
    assert estimator.predict_calls == 1
    assert loaded == ["validation"]
    selection = json.loads(
        (output / "threshold_selection/threshold_selection.json").read_text(encoding="utf-8")
    )
    assert selection["objective"] == "maximize_validation_balanced_accuracy"
    assert selection["tie_break"] == ["minimum_absolute_distance_to_0.5", "lower_threshold"]
    assert selection["data_boundary"]["model_fit_performed"] is False
    assert selection["data_boundary"]["predict_proba_calls"] == 0
    assert threshold.validate_plain_mlp_validation_threshold(output) == manifest


def test_threshold_validator_rejects_tampering(tmp_path, monkeypatch):
    paths, _, _, _ = _setup(tmp_path, monkeypatch)
    output = tmp_path / "validation"
    validation.run_plain_mlp_validation(output, **paths)
    threshold.run_plain_mlp_validation_threshold(output)
    with (output / "threshold_selection/threshold_candidates.csv").open("a", encoding="utf-8") as handle:
        handle.write("1,0.5,0.5,0.5,1,1,1,1\n")
    with pytest.raises(threshold.PlainMLPThresholdError, match="hash mismatch"):
        threshold.validate_plain_mlp_validation_threshold(output)


def test_threshold_tie_break_prefers_nearest_to_half_then_lower():
    rows = [
        {"threshold": 0.4, "balanced_accuracy": 0.8},
        {"threshold": 0.6, "balanced_accuracy": 0.8},
        {"threshold": 0.5, "balanced_accuracy": 0.7},
    ]
    selected, maximizing = threshold._select(rows)
    assert selected["threshold"] == 0.4
    assert len(maximizing) == 2
