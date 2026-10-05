"""Targeted source-closure tests for the final 15-feature RealMLP pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.dataset_b.models.realmlp import RealMLPClassifier, load_realmlp_v1_config
from src.dataset_b.preprocessing import DatasetBPartition
from src.dataset_b.preprocessing import DatasetBInputQualityGate
from src.dataset_b.preprocessing.categorical_encoder import (
    CATEGORICAL_COLUMNS,
    NUMERIC_VALUE_COLUMNS,
    TWO_CHANNEL_COLUMNS,
)
from src.dataset_b.preprocessing.raw_parser import FEATURE_COLUMNS
from src.dataset_b.runtime import realmlp_final_contract as contract
from src.dataset_b.runtime.realmlp_final_contract import (
    RealMLPFinalContractError,
    RealMLPFinalPaths,
    base_manifest,
    fit_native_preprocessing,
    load_quality_gated_partition,
    validate_manifest,
    validate_run_id,
    write_json,
)
from src.dataset_b.runtime.realmlp_final_training import lock_realmlp_fixed_epoch
from src.dataset_b.runtime import realmlp_final_training as training
from src.dataset_b.evaluation import realmlp_final_evaluation as evaluation
from src.dataset_b.reporting.realmlp_reporting import write_realmlp_reporting
from src.dataset_b.review import realmlp_publication as publication
from src.dataset_b.review.realmlp_final_review import review_realmlp_final
from src.dataset_b.review.realmlp_publication import publish_realmlp_final


RUN_ID = "realmlp_final_15f_20261004_01"


def _raw_row(**changes):
    row = {
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


class _Loader:
    feature_columns = FEATURE_COLUMNS

    def __init__(self, rows=20):
        raw = pd.DataFrame(
            [
                _raw_row(
                    reverse_mortgage="1111" if index % 5 == 0 else "2",
                    income=str(100 + index),
                    loan_amount=str(200 + index),
                )
                for index in range(rows)
            ],
            columns=FEATURE_COLUMNS,
            dtype=object,
        )
        labels = pd.Series([index % 2 for index in range(rows)], dtype="int8")
        actions = labels.map({1: "1", 0: "3"})
        self.partition = DatasetBPartition(
            name="train",
            X=raw,
            y=labels,
            record_ids=pd.Series([f"r{index:03d}" for index in range(rows)]),
            audit=pd.DataFrame({f"audit_{index}": ["x"] * rows for index in range(9)}),
            action_taken=actions,
        )

    def load(self, partition):
        assert partition == "train"
        return self.partition


class _FakeEstimator:
    classes_ = np.asarray([0, 1])

    def __init__(self):
        self.predict_calls = 0

    def predict_proba(self, frame):
        self.predict_calls += 1
        positive = np.linspace(0.2, 0.8, len(frame), dtype=np.float64)
        return np.column_stack((1.0 - positive, positive))


def test_final_config_and_wrapper_use_native_15_to_19_without_scaler_or_ohe():
    config = load_realmlp_v1_config()
    assert config["status"] == "final_locked_configuration"
    assert config["contracts"]["raw_feature_count"] == 15
    assert config["contracts"]["semantic_feature_count"] == 19
    assert config["input_contract"] == {
        "preprocessing_version": "dataset_b_neural_preprocessing_v1",
        "representation": "realmlp_native_19_columns",
        "raw_feature_count": 15,
        "semantic_feature_count": 19,
        "numerical_dtype": "float32",
        "categorical_dtype": "category",
        "external_scaler": False,
        "external_ohe": False,
    }
    assert RealMLPClassifier.from_yaml().device == "cuda"


def test_central_quality_gate_feeds_typed_native_19_column_frame(monkeypatch):
    monkeypatch.setitem(contract.EXPECTED_ROWS, "train", 20)
    gated = load_quality_gated_partition("train", loader=_Loader())
    pipeline, prepared = fit_native_preprocessing(gated)

    assert gated.quality_audit.status == "PASS"
    assert gated.quality_audit.raw_1111_counts["reverse_mortgage"] == 4
    assert gated.quality_audit.normalized_exempt_counts["reverse_mortgage"] == 4
    assert tuple(prepared.X.columns) == TWO_CHANNEL_COLUMNS
    assert len(prepared.X.columns) == 19
    assert all(prepared.X[column].dtype == np.float32 for column in NUMERIC_VALUE_COLUMNS)
    assert all(
        isinstance(prepared.X[column].dtype, pd.CategoricalDtype)
        for column in CATEGORICAL_COLUMNS
    )
    assert prepared.X["reverse_mortgage"].eq("Exempt").sum() == 4
    assert "scale" not in pipeline.named_steps
    assert "encode" not in pipeline.named_steps


def test_run_id_and_manifest_replace_sha_lineage(tmp_path):
    assert validate_run_id(RUN_ID) == RUN_ID
    with pytest.raises(RealMLPFinalContractError):
        validate_run_id("realmlp_v1")
    manifest = base_manifest(RUN_ID, "cv")
    assert validate_manifest(manifest, run_id=RUN_ID, stage="cv") == manifest
    assert "sha" not in json.dumps(manifest).lower()
    paths = RealMLPFinalPaths.for_run(RUN_ID, artifact_root=tmp_path)
    assert paths.working_root == tmp_path / "working" / RUN_ID
    assert paths.final_root == tmp_path / "final" / RUN_ID


def test_fixed_epoch_is_run_scoped_and_not_hard_coded(tmp_path):
    paths = RealMLPFinalPaths.for_run(RUN_ID, artifact_root=tmp_path)
    pilot = paths.stage("pilot")
    pilot.mkdir(parents=True)
    write_json(
        pilot / "manifest.json",
        {
            **base_manifest(RUN_ID, "pilot", status="AWAITING_DECISION"),
            "status": "AWAITING_DECISION",
            "maximum_epochs": 256,
            "completed_epochs": 256,
            "pilot_fit_rows": 1,
            "pilot_inner_stop_rows": 1,
        },
    )
    pd.DataFrame({"record_id": ["fit-1"]}).to_csv(
        pilot / "pilot_fit_ids.csv", index=False
    )
    pd.DataFrame({"record_id": ["stop-1"]}).to_csv(
        pilot / "pilot_inner_stop_ids.csv", index=False
    )
    pd.DataFrame(
        {
            "record_id": ["stop-1"],
            "y_approved": [1],
            "p_denied": [0.25],
            "p_approved": [0.75],
        }
    ).to_csv(pilot / "pilot_predictions.csv", index=False)
    decision = lock_realmlp_fixed_epoch(
        RUN_ID,
        37,
        artifact_root=tmp_path,
        rationale="minimum observed Train-only inner-stop loss",
    )
    assert decision["fixed_epoch"] == 37
    assert decision["run_id"] == RUN_ID
    assert "sha" not in json.dumps(decision).lower()


def test_pilot_is_train_only_and_persists_disjoint_ordered_ids(tmp_path, monkeypatch):
    gated = DatasetBInputQualityGate().apply(_Loader(rows=100).partition)
    monkeypatch.setitem(contract.EXPECTED_ROWS, "train", 100)
    monkeypatch.setattr(training, "load_quality_gated_partition", lambda partition: gated)

    def fake_fit(X, y, **kwargs):
        estimator = _FakeEstimator()
        probabilities = estimator.predict_proba(kwargs["X_predict"])
        history = [{"epoch": epoch} for epoch in range(1, kwargs["epochs"] + 1)]
        return estimator, probabilities, history, 0.01

    result = training.run_realmlp_pilot(
        RUN_ID,
        artifact_root=tmp_path,
        fit_model=fake_fit,
    )
    pilot = RealMLPFinalPaths.for_run(RUN_ID, artifact_root=tmp_path).stage("pilot")
    fit_ids = pd.read_csv(pilot / "pilot_fit_ids.csv", dtype=str)["record_id"].tolist()
    stop_ids = pd.read_csv(
        pilot / "pilot_inner_stop_ids.csv", dtype=str
    )["record_id"].tolist()
    prediction_ids = pd.read_csv(
        pilot / "pilot_predictions.csv", dtype=str
    )["record_id"].tolist()

    assert result["status"] == "AWAITING_DECISION"
    assert len(fit_ids) == result["pilot_fit_rows"]
    assert len(stop_ids) == result["pilot_inner_stop_rows"]
    assert set(fit_ids).isdisjoint(stop_ids)
    assert prediction_ids == stop_ids


def test_active_final_sources_exclude_historical_locks():
    root = Path(__file__).parents[1]
    active = [
        root / "config/dataset_b/realmlp_v1.yaml",
        root / "src/dataset_b/models/realmlp.py",
        root / "src/dataset_b/runtime/realmlp_final_contract.py",
        root / "src/dataset_b/runtime/realmlp_final_training.py",
        root / "src/dataset_b/evaluation/realmlp_final_evaluation.py",
        root / "src/dataset_b/reporting/realmlp_reporting.py",
        root / "src/dataset_b/review/realmlp_final_review.py",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in active)
    for forbidden in (
        "realmlp_native_16_columns",
        "temporary_training_lock_v0_2",
        "LOCKED_THRESHOLD = 0.771",
        "FIXED_EPOCHS = 51",
    ):
        assert forbidden not in combined


def test_cv_and_full_train_use_dynamic_epoch_and_fresh_state(tmp_path, monkeypatch):
    gated = DatasetBInputQualityGate().apply(_Loader(rows=100).partition)
    monkeypatch.setitem(contract.EXPECTED_ROWS, "train", 100)
    monkeypatch.setattr(training, "load_quality_gated_partition", lambda partition: gated)
    paths = RealMLPFinalPaths.for_run(RUN_ID, artifact_root=tmp_path)
    pilot = paths.stage("pilot")
    pilot.mkdir(parents=True)
    write_json(
        pilot / "decision.json",
        {
            **base_manifest(RUN_ID, "pilot_decision"),
            "decision": "FIXED_EPOCH",
            "fixed_epoch": 3,
            "rationale": "synthetic source-closure evidence",
        },
    )
    calls = []

    def fake_fit(X, y, **kwargs):
        estimator = _FakeEstimator()
        calls.append((estimator, kwargs["stage"], kwargs["epochs"], kwargs["X_predict"] is not None))
        probabilities = None
        if kwargs["X_predict"] is not None:
            probabilities = estimator.predict_proba(kwargs["X_predict"])
        return estimator, probabilities, [{"epoch": value} for value in range(1, kwargs["epochs"] + 1)], 0.01

    cv = training.run_realmlp_cv(
        RUN_ID, artifact_root=tmp_path, fit_model=fake_fit
    )
    full = training.run_realmlp_full_train(
        RUN_ID, artifact_root=tmp_path, fit_model=fake_fit
    )

    assert cv["fixed_epoch"] == full["fixed_epoch"] == 3
    assert cv["fold_count"] == 5
    assert full["execution"] == {"fit_calls": 1, "predict_proba_calls": 0}
    assert len(calls) == 6
    assert len({id(call[0]) for call in calls}) == 6
    assert all(call[2] == 3 for call in calls)
    assert [call[3] for call in calls] == [True, True, True, True, True, False]


def test_validation_is_zero_fit_and_single_predict(tmp_path, monkeypatch):
    gated = DatasetBInputQualityGate().apply(_Loader(rows=20).partition)
    gated.name = "validation"
    pipeline, _prepared = fit_native_preprocessing(gated)
    estimator = _FakeEstimator()
    paths = RealMLPFinalPaths.for_run(RUN_ID, artifact_root=tmp_path)
    monkeypatch.setitem(contract.EXPECTED_ROWS, "validation", 20)
    monkeypatch.setattr(evaluation, "validate_realmlp_full_train", lambda _paths: {"status": "PASS"})
    monkeypatch.setattr(evaluation, "load_quality_gated_partition", lambda partition: gated)
    monkeypatch.setattr(
        evaluation,
        "_load_bundle",
        lambda _paths: {"estimator": estimator, "preprocessing": pipeline, "scope": "all_immutable_train_rows"},
    )

    manifest = evaluation.run_realmlp_validation(RUN_ID, artifact_root=tmp_path)
    threshold = evaluation.select_realmlp_threshold(RUN_ID, artifact_root=tmp_path)

    assert estimator.predict_calls == 1
    assert manifest["execution"] == {
        "model_fit_calls": 0,
        "preprocessing_fit_calls": 0,
        "preprocessing_transform_calls": 1,
        "predict_proba_calls": 1,
    }
    assert 0.0 < threshold["selected_threshold"] < 1.0


def test_closure_interfaces_are_importable_without_execution():
    assert callable(write_realmlp_reporting)
    assert callable(review_realmlp_final)
    assert callable(publish_realmlp_final)


def _realmlp_publication_paths(artifact_root: Path) -> RealMLPFinalPaths:
    paths = RealMLPFinalPaths.for_run(RUN_ID, artifact_root=artifact_root)
    paths.stage("reporting").mkdir(parents=True)
    return paths


def test_realmlp_publication_validates_final_before_frozen(tmp_path, monkeypatch):
    paths = _realmlp_publication_paths(tmp_path)
    events = []

    def reviewer(run_id, *, artifact_root, published=False):
        assert run_id == RUN_ID
        assert Path(artifact_root) == tmp_path
        root = paths.final_root if published else paths.working_root
        assert root.is_dir()
        assert not (root / "FROZEN").exists()
        events.append(("review", "final" if published else "working"))
        return {"status": "PASS", "run_id": RUN_ID}

    def reporting_validator(path):
        path = Path(path)
        root = path.parent
        assert path.is_dir()
        assert not (root / "FROZEN").exists()
        events.append(("reporting", "final" if root == paths.final_root else "working"))
        return {"run_id": RUN_ID, "reporting_status": "COMPLETE"}

    monkeypatch.setattr(publication, "review_realmlp_final", reviewer)
    monkeypatch.setattr(publication, "validate_reporting_directory", reporting_validator)

    destination = publish_realmlp_final(RUN_ID, artifact_root=tmp_path)

    assert destination == paths.final_root
    assert events == [
        ("review", "working"),
        ("reporting", "working"),
        ("review", "final"),
        ("reporting", "final"),
    ]
    assert (paths.final_root / "FROZEN").read_text(encoding="utf-8") == (
        "REALMLP FINAL FROZEN\n"
    )
    assert not paths.working_root.exists()


def test_realmlp_publication_failure_paths_never_create_frozen(tmp_path, monkeypatch):
    def passed_review(run_id, *, artifact_root, published=False):
        return {"status": "PASS", "run_id": run_id}

    def passed_reporting(_path):
        return {"run_id": RUN_ID, "reporting_status": "COMPLETE"}

    rename_root = tmp_path / "rename_failure"
    rename_paths = _realmlp_publication_paths(rename_root)
    monkeypatch.setattr(publication, "review_realmlp_final", passed_review)
    monkeypatch.setattr(publication, "validate_reporting_directory", passed_reporting)
    monkeypatch.setattr(
        publication.os,
        "replace",
        lambda *_args: (_ for _ in ()).throw(OSError("atomic rename failure")),
    )
    with pytest.raises(OSError, match="atomic rename failure"):
        publish_realmlp_final(RUN_ID, artifact_root=rename_root)
    assert rename_paths.working_root.is_dir()
    assert not (rename_paths.working_root / "FROZEN").exists()
    assert not rename_paths.final_root.exists()

    monkeypatch.undo()
    post_review_root = tmp_path / "post_review_failure"
    post_review_paths = _realmlp_publication_paths(post_review_root)

    def failed_published_review(run_id, *, artifact_root, published=False):
        return {"status": "FAIL" if published else "PASS", "run_id": run_id}

    monkeypatch.setattr(publication, "review_realmlp_final", failed_published_review)
    monkeypatch.setattr(publication, "validate_reporting_directory", passed_reporting)
    with pytest.raises(RealMLPFinalContractError, match="Published RealMLP final review"):
        publish_realmlp_final(RUN_ID, artifact_root=post_review_root)
    assert post_review_paths.final_root.is_dir()
    assert not (post_review_paths.final_root / "FROZEN").exists()

    monkeypatch.undo()
    post_reporting_root = tmp_path / "post_reporting_failure"
    post_reporting_paths = _realmlp_publication_paths(post_reporting_root)

    def reporting_fails_after_rename(path):
        if Path(path).parent == post_reporting_paths.final_root:
            raise RuntimeError("post-move reporting failure")
        return {"run_id": RUN_ID, "reporting_status": "COMPLETE"}

    monkeypatch.setattr(publication, "review_realmlp_final", passed_review)
    monkeypatch.setattr(publication, "validate_reporting_directory", reporting_fails_after_rename)
    with pytest.raises(RuntimeError, match="post-move reporting failure"):
        publish_realmlp_final(RUN_ID, artifact_root=post_reporting_root)
    assert post_reporting_paths.final_root.is_dir()
    assert not (post_reporting_paths.final_root / "FROZEN").exists()


def test_realmlp_existing_final_fails_before_review(tmp_path, monkeypatch):
    paths = _realmlp_publication_paths(tmp_path)
    paths.final_root.mkdir(parents=True)
    monkeypatch.setattr(
        publication,
        "review_realmlp_final",
        lambda *_args, **_kwargs: pytest.fail("must not review an existing final"),
    )

    with pytest.raises(RealMLPFinalContractError, match="publication boundary"):
        publish_realmlp_final(RUN_ID, artifact_root=tmp_path)

    assert paths.working_root.is_dir()
    assert not (paths.working_root / "FROZEN").exists()
    assert not (paths.final_root / "FROZEN").exists()
