"""Run-scoped atomic export checks for final PlainMLP preprocessing."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import load_npz
from sklearn.model_selection import StratifiedKFold

from src.dataset_b.preprocessing import export_neural_cv as export_module
from src.dataset_b.preprocessing.raw_parser import EXEMPTION_FLAG_COLUMNS, FEATURE_COLUMNS
from src.dataset_b.preprocessing.split_inputs import (
    DatasetBInputQualityGate,
    DatasetBPartition,
    InputQualityError,
)


def _row(**changes):
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
        "preapproval": "2",
        "interest_only_payment": "2",
        "balloon_payment": "2",
    }
    row.update(changes)
    return row


def _partition(rows):
    X = pd.DataFrame(rows, columns=FEATURE_COLUMNS, dtype=object)
    y = pd.Series([0, 1] * (len(rows) // 2), dtype="int8")
    ids = pd.Series([f"record_{index}" for index in range(len(rows))], dtype=object)
    actions = y.map({0: "3", 1: "1"})
    return DatasetBPartition(
        name="train",
        X=X,
        y=y,
        record_ids=ids,
        audit=pd.DataFrame(index=X.index),
        action_taken=actions,
    )


def _install_loader_and_splitter(monkeypatch, partition, accessed):
    class FakeLoader:
        feature_version = "feature_set_v1"

        def load(self, requested):
            accessed.append(requested)
            return partition

    class FakeSplitter:
        @staticmethod
        def from_yaml():
            return SimpleNamespace(
                cv_splitter=lambda: StratifiedKFold(
                    n_splits=5, shuffle=True, random_state=42
                )
            )

    monkeypatch.setattr(export_module, "DatasetBSplitLoader", FakeLoader)
    monkeypatch.setattr(export_module, "DatasetBSplitter", FakeSplitter)


def test_neural_export_is_final_train_only_run_scoped_and_non_overwriting(
    tmp_path, monkeypatch
):
    rows = []
    for index in range(10):
        changes = {"income": str(10 + index), "loan_amount": str(200 + index * 10)}
        if index < 5:
            changes.update({column: "1111" for column in EXEMPTION_FLAG_COLUMNS})
        rows.append(_row(**changes))
    partition = _partition(rows)
    accessed: list[str] = []
    _install_loader_and_splitter(monkeypatch, partition, accessed)

    state = {"gate_calls": 0, "pipeline_calls": 0}

    class CountingGate(DatasetBInputQualityGate):
        def apply(self, value):
            state["gate_calls"] += 1
            return super().apply(value)

    original_factory = export_module.make_plain_mlp_preprocessing_pipeline

    def checked_factory():
        assert state["gate_calls"] == 1
        state["pipeline_calls"] += 1
        return original_factory()

    monkeypatch.setattr(export_module, "DatasetBInputQualityGate", CountingGate)
    monkeypatch.setattr(
        export_module, "make_plain_mlp_preprocessing_pipeline", checked_factory
    )

    root = tmp_path / "cv_neural"
    run_id = "plain_mlp_final_15f_test_01"
    manifest_path = export_module.export_neural_cv_preprocessed(run_id, root)
    destination = root / run_id
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert accessed == ["train"]
    assert state == {"gate_calls": 1, "pipeline_calls": 5}
    assert manifest_path == destination / "manifest.json"
    assert manifest["run_id"] == run_id
    assert manifest["source_partition"] == "train"
    assert manifest["fit_scope"] == "fold_train_only"
    assert manifest["use"] == "cross_validation_only"
    assert (
        manifest["raw_feature_count"],
        manifest["semantic_feature_count"],
        manifest["encoded_feature_count"],
    ) == (15, 19, 62)
    assert manifest["raw_feature_columns"] == list(FEATURE_COLUMNS)
    assert len(manifest["semantic_columns"]) == 19
    assert len(manifest["output_columns"]) == 62
    assert manifest["matrix_contract"] == {
        "format": "csr",
        "dtype": "float64",
        "finite": True,
        "full_matrix_densification": False,
    }
    assert manifest["numeric_transforms"]["log1p_columns"] == [
        "income_value",
        "loan_amount_value",
    ]
    assert manifest["numeric_transforms"]["robust_scaled_columns"] == [
        "income_value",
        "loan_amount_value",
        "loan_to_value_ratio_value",
        "loan_term_value",
    ]
    assert manifest["numeric_transforms"]["ohe_scaled"] is False
    assert manifest["quality_gate"]["rejected_rows"] == 0
    assert manifest["quality_gate"]["semantic_1111_count"] == 0
    assert manifest["quality_gate"]["raw_1111_counts"] == {
        column: 5 for column in EXEMPTION_FLAG_COLUMNS
    }
    assert manifest["quality_gate"]["normalized_exempt_counts"] == {
        column: 5 for column in EXEMPTION_FLAG_COLUMNS
    }
    assert "MISSING" not in manifest["output_categories"]
    assert "Exempt" in manifest["output_categories"]
    assert manifest["fold_validation_coverage"] == {
        "rows": 10,
        "minimum_assignments": 1,
        "maximum_assignments": 1,
        "exactly_once": True,
    }

    centers = set()
    for fold in manifest["folds"]:
        fold_dir = destination / fold["directory"]
        train_matrix = load_npz(fold_dir / "plain_mlp_train.npz")
        validation_matrix = load_npz(fold_dir / "plain_mlp_fold_validation.npz")
        assert train_matrix.shape == (8, 62)
        assert validation_matrix.shape == (2, 62)
        assert train_matrix.dtype == validation_matrix.dtype == np.float64
        assert np.isfinite(train_matrix.data).all()
        assert np.isfinite(validation_matrix.data).all()
        assert set(np.unique(train_matrix[:, 4:].data)).issubset({1.0})
        assert set(np.unique(validation_matrix[:, 4:].data)).issubset({1.0})
        assert fold["selective_log1p_columns"] == [
            "income_value",
            "loan_amount_value",
        ]
        assert fold["unchanged_numeric_columns"] == [
            "loan_to_value_ratio_value",
            "loan_term_value",
        ]
        centers.add(tuple(fold["robust_center"]))
    assert len(centers) > 1

    with pytest.raises(FileExistsError, match="already exists"):
        export_module.export_neural_cv_preprocessed(run_id, root)
    assert accessed == ["train"]


@pytest.mark.parametrize("run_id", ["", "../escape", "bad/run", "bad run", ".hidden"])
def test_neural_export_rejects_invalid_run_id_before_loading(run_id, tmp_path, monkeypatch):
    class FailLoader:
        def __init__(self):
            raise AssertionError("loader must not be created for an invalid run_id")

    monkeypatch.setattr(export_module, "DatasetBSplitLoader", FailLoader)
    with pytest.raises(ValueError, match="run_id"):
        export_module.export_neural_cv_preprocessed(run_id, tmp_path)


def test_neural_export_stops_when_quality_gate_would_reject_a_record(
    tmp_path, monkeypatch
):
    rows = [_row(income="")]
    rows.extend(_row(income=str(index + 1)) for index in range(1, 10))
    partition = _partition(rows)
    accessed: list[str] = []
    _install_loader_and_splitter(monkeypatch, partition, accessed)

    with pytest.raises(InputQualityError, match="rejected one or more rows"):
        export_module.export_neural_cv_preprocessed("plain_mlp_bad_quality", tmp_path)
    assert accessed == ["train"]
    assert not (tmp_path / "plain_mlp_bad_quality").exists()


def test_absent_category_gate_fails_before_artifact_creation(tmp_path, monkeypatch):
    partition = _partition([_row(loan_type="4"), *[_row() for _ in range(9)]])
    accessed: list[str] = []

    class FakeLoader:
        feature_version = "feature_set_v1"

        def load(self, requested):
            accessed.append(requested)
            return partition

    folds = [(np.arange(1, 10), np.asarray([0]))] + [
        (
            np.asarray([index for index in range(10) if index != validation]),
            np.asarray([validation]),
        )
        for validation in range(1, 5)
    ]

    class FixedFiveFold:
        def split(self, X_value, y_value):
            yield from folds

    class FakeSplitter:
        @staticmethod
        def from_yaml():
            return SimpleNamespace(cv_splitter=lambda: FixedFiveFold())

    monkeypatch.setattr(export_module, "DatasetBSplitLoader", FakeLoader)
    monkeypatch.setattr(export_module, "DatasetBSplitter", FakeSplitter)
    root = tmp_path / "exports"
    with pytest.raises(ValueError, match="absent-category gate failed"):
        export_module.export_neural_cv_preprocessed("plain_mlp_absent", root)
    assert accessed == ["train"]
    assert not (root / "plain_mlp_absent").exists()
    assert not list(root.glob(".plain_mlp_absent.*"))
