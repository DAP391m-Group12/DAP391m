"""Synthetic orchestration tests for PMLP-F04 CV and full-Train runners."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from sklearn.model_selection import StratifiedKFold

from src.dataset_b.runtime import plain_mlp_training as training
from src.dataset_b.runtime.plain_mlp_engine import PlainMLPEpochRecord


class _FakePreprocessor:
    def __init__(self, identity: int) -> None:
        self.identity = identity
        self.fit_rows = 0

    @staticmethod
    def _matrix(frame: pd.DataFrame) -> csr_matrix:
        values = np.zeros((len(frame), 62), dtype=np.float64)
        values[:, 0] = frame["value"].to_numpy(dtype=np.float64)
        values[:, 4] = 1.0
        return csr_matrix(values)

    def fit_transform(self, frame: pd.DataFrame, y: np.ndarray) -> csr_matrix:
        self.fit_rows = len(frame)
        return self._matrix(frame)

    def transform(self, frame: pd.DataFrame) -> csr_matrix:
        if self.fit_rows == 0:
            raise AssertionError("unfitted preprocessing state was reused")
        return self._matrix(frame)


class _FakeEstimator:
    def __init__(self, best_epoch: int, identity: int, *, restored: bool = False) -> None:
        self.best_epoch = best_epoch
        self.identity = identity
        self.restored = restored
        self.predict_calls = 0

    def fit(self, X: csr_matrix, y: np.ndarray):
        self.fit_rows = X.shape[0]
        self.pos_weight_ = float(np.count_nonzero(y == 0) / np.count_nonzero(y == 1))
        self.training_history_ = [0.5 / epoch for epoch in range(1, self.best_epoch + 1)]
        return self

    def predict_proba(self, X: csr_matrix) -> np.ndarray:
        self.predict_calls += 1
        approved = np.full(X.shape[0], 0.6, dtype=np.float64)
        return np.column_stack((np.ones_like(approved) - approved, approved))

    def checkpoint_payload(self) -> dict[str, object]:
        return {
            "best_epoch": self.best_epoch,
            "identity": self.identity,
            "pos_weight": self.pos_weight_,
            "history": list(self.training_history_),
        }

    @classmethod
    def restore(cls, payload: dict[str, object]):
        estimator = cls(
            int(payload["best_epoch"]), int(payload["identity"]), restored=True
        )
        estimator.pos_weight_ = float(payload["pos_weight"])
        estimator.training_history_ = [float(value) for value in payload["history"]]
        return estimator


def _population(rows: int = 50) -> training.PlainMLPTrainPopulation:
    return training.PlainMLPTrainPopulation(
        X=pd.DataFrame({"value": np.arange(rows, dtype=np.float64)}),
        y=np.asarray([0, 1] * (rows // 2), dtype=np.float64),
        record_ids=tuple(f"train_{index:03d}" for index in range(rows)),
    )


def _dependencies():
    population = _population()
    trace = {
        "load_train_calls": 0,
        "preprocessors": [],
        "selector_calls": [],
        "final_estimators": [],
    }

    def load_train():
        trace["load_train_calls"] += 1
        return population

    def outer_folds(X, y):
        return StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y)

    def preprocessor_factory():
        value = _FakePreprocessor(len(trace["preprocessors"]) + 1)
        trace["preprocessors"].append(value)
        return value

    def selector_runner(X_fit, y_fit, X_stop, y_stop, batch_size):
        trace["selector_calls"].append(
            {
                "fit_rows": X_fit.shape[0],
                "stop_rows": X_stop.shape[0],
                "batch_size": batch_size,
                "pos_weight": float(
                    np.count_nonzero(y_fit == 0) / np.count_nonzero(y_fit == 1)
                ),
            }
        )
        history = (
            PlainMLPEpochRecord(1, 0.8, True, 1, 0),
            PlainMLPEpochRecord(2, 0.7, True, 2, 0),
        )
        return training.PlainMLPSelectorEvidence(
            best_epoch=2,
            best_inner_stop_bce_loss=0.7,
            stopped_after_epoch=2,
            stop_reason="synthetic_complete",
            history=history,
            pos_weight=trace["selector_calls"][-1]["pos_weight"],
        )

    def final_estimator_factory(best_epoch, batch_size):
        assert batch_size == 256
        value = _FakeEstimator(best_epoch, len(trace["final_estimators"]) + 1)
        trace["final_estimators"].append(value)
        return value

    dependencies = training.PlainMLPRunnerDependencies(
        load_train=load_train,
        outer_folds=outer_folds,
        preprocessor_factory=preprocessor_factory,
        selector_runner=selector_runner,
        final_estimator_factory=final_estimator_factory,
        checkpoint_loader=_FakeEstimator.restore,
    )
    return dependencies, trace, population


def test_cv_and_full_train_accounting_state_isolation_and_oof(tmp_path):
    dependencies, trace, population = _dependencies()
    historical = tmp_path / "historical"
    historical.mkdir()
    (historical / "forbidden.joblib").write_bytes(b"must-not-be-read")
    cv_dir = tmp_path / "cv"
    full_dir = tmp_path / "full_train"

    cv_manifest = training.run_plain_mlp_cv(
        cv_dir,
        dependencies=dependencies,
        expected_train_rows=50,
        run_id="plain_mlp_test_run",
    )
    full_manifest = training.run_plain_mlp_full_train(
        full_dir,
        dependencies=dependencies,
        expected_train_rows=50,
        run_id="plain_mlp_test_run",
    )

    assert cv_manifest["status"] == full_manifest["status"] == "PASS"
    assert cv_manifest["run_id"] == full_manifest["run_id"] == "plain_mlp_test_run"
    assert trace["load_train_calls"] == 2
    assert len(trace["selector_calls"]) == 6
    assert len(trace["final_estimators"]) == 6
    assert len(trace["preprocessors"]) == 12
    assert len({id(value) for value in trace["preprocessors"]}) == 12
    assert len({id(value) for value in trace["final_estimators"]}) == 6
    assert all(value.fit_rows > 0 for value in trace["preprocessors"])
    assert [value.predict_calls for value in trace["final_estimators"][:5]] == [1] * 5
    assert trace["final_estimators"][5].predict_calls == 1

    with np.load(cv_dir / "oof_predictions.npz", allow_pickle=False) as oof:
        assert oof["record_ids"].astype(str).tolist() == list(population.record_ids)
        assert len(set(oof["record_ids"].astype(str).tolist())) == 50
        assert np.array_equal(oof["coverage"], np.ones(50, dtype=np.int8))
        assert oof["probabilities"].dtype == np.float64
    cv_result = json.loads((cv_dir / "cv_result.json").read_text(encoding="utf-8"))
    full_result = json.loads(
        (full_dir / "full_train_result.json").read_text(encoding="utf-8")
    )
    assert cv_result["initializations"] == {
        "temporary_selectors": 5,
        "final_estimators": 5,
    }
    assert cv_result["run_id"] == full_result["run_id"] == "plain_mlp_test_run"
    assert cv_result["fit_call_accounting"] == {
        "selector_preprocessing_fit": 5,
        "selector_model_fit": 5,
        "final_preprocessing_fit": 5,
        "final_model_fit": 5,
        "outer_fold_validation_predict_proba": 5,
    }
    assert full_result["initializations"] == {
        "temporary_selectors": 1,
        "final_estimators": 1,
    }
    assert full_result["fit_call_accounting"] == {
        "selector_preprocessing_fit": 1,
        "selector_model_fit": 1,
        "final_preprocessing_fit": 1,
        "final_model_fit": 1,
        "fixed_subset_predict_proba_before_reload": 1,
        "fixed_subset_predict_proba_after_reload": 1,
    }
    assert cv_result["final_evidence_estimators"] + full_result[
        "final_evidence_estimators"
    ] == 6
    assert full_result["checkpoint_reload"] == {
        "performed": True,
        "fixed_ordered_subset_rows": 50,
        "positions": list(range(50)),
        "predictions_exactly_equal": True,
        "scope_equal": True,
    }
    assert all(
        fold["final_estimator"]["predict_proba_calls"] == 1
        and fold["selector"]["temporary_state_published"] is False
        for fold in cv_result["folds"]
    )
    evidence_bundles = list(cv_dir.glob("fold_*/final_bundle.joblib")) + [
        full_dir / "checkpoints/fitted_bundle.joblib"
    ]
    assert len(evidence_bundles) == 6 and all(path.is_file() for path in evidence_bundles)
    assert not list(tmp_path.rglob("*selector*.joblib"))
    assert (historical / "forbidden.joblib").read_bytes() == b"must-not-be-read"
    assert not cv_dir.with_name("cv.working").exists()
    assert not full_dir.with_name("full_train.working").exists()


@pytest.mark.parametrize("stage", ["cv", "full_train"])
def test_incomplete_working_state_is_preserved_and_never_resumed(tmp_path, stage):
    dependencies, trace, _ = _dependencies()
    destination = tmp_path / stage
    working = destination.with_name(destination.name + ".working")
    working.mkdir()
    stale = working / "incomplete_fitted_state.joblib"
    stale.write_bytes(b"preserve")
    runner = (
        training.run_plain_mlp_cv
        if stage == "cv"
        else training.run_plain_mlp_full_train
    )

    with pytest.raises(training.PlainMLPTrainingError, match="preserved"):
        runner(
            destination,
            dependencies=dependencies,
            expected_train_rows=50,
            run_id="plain_mlp_test_run",
        )

    assert stale.read_bytes() == b"preserve"
    assert trace["load_train_calls"] == 0
    assert trace["selector_calls"] == []
    assert trace["final_estimators"] == []


@pytest.mark.parametrize("stage", ["cv", "full_train"])
def test_existing_destination_is_never_overwritten_or_used_as_history(tmp_path, stage):
    dependencies, trace, _ = _dependencies()
    destination = tmp_path / stage
    destination.mkdir()
    marker = destination / "historical.txt"
    marker.write_text("preserve\n", encoding="utf-8")
    runner = (
        training.run_plain_mlp_cv
        if stage == "cv"
        else training.run_plain_mlp_full_train
    )

    with pytest.raises(training.PlainMLPTrainingError, match="historical destination"):
        runner(
            destination,
            dependencies=dependencies,
            expected_train_rows=50,
            run_id="plain_mlp_test_run",
        )

    assert marker.read_text(encoding="utf-8") == "preserve\n"
    assert trace["load_train_calls"] == 0


def test_artifact_hash_tampering_is_rejected(tmp_path):
    dependencies, _, _ = _dependencies()
    destination = tmp_path / "cv"
    training.run_plain_mlp_cv(
        destination,
        dependencies=dependencies,
        expected_train_rows=50,
        run_id="plain_mlp_test_run",
    )
    with (destination / "cv_result.json").open("a", encoding="utf-8") as handle:
        handle.write("\n")

    with pytest.raises(training.PlainMLPTrainingError, match="hash mismatch"):
        training._validate_plain_mlp_cv(
            destination, 50, _FakeEstimator.restore
        )


def test_feature_contract_hash_remains_provenance_not_semantic_identity():
    expected = {
        "split_manifest_sha256": "a" * 64,
        "feature_contract_sha256": "b" * 64,
    }
    presentation_drift = {
        "split_manifest_sha256": "a" * 64,
        "feature_contract_sha256": "c" * 64,
    }

    assert training._lineage_matches_feature_provenance(
        presentation_drift, expected
    )
    assert not training._lineage_matches_feature_provenance(
        {**presentation_drift, "split_manifest_sha256": "d" * 64}, expected
    )


def test_default_loader_requests_train_only(monkeypatch):
    requested = []
    raw = object()
    gated = SimpleNamespace(
        X=pd.DataFrame({"value": [0.0, 1.0]}),
        y=pd.Series([0, 1], dtype="int8"),
        record_ids=pd.Series(["train_0", "train_1"]),
        quality_audit=SimpleNamespace(rejected_rows=0),
    )

    class Loader:
        def load(self, partition):
            requested.append(partition)
            return raw

    class Gate:
        def apply(self, partition):
            assert partition is raw
            return gated

    monkeypatch.setattr(training, "DatasetBSplitLoader", Loader)
    monkeypatch.setattr(training, "DatasetBInputQualityGate", Gate)

    population = training._default_load_train()

    assert requested == ["train"]
    assert population.record_ids == ("train_0", "train_1")
