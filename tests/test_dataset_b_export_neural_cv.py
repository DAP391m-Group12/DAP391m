"""Atomic export checks for Dataset B plain-MLP fold artifacts."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import load_npz
from sklearn.model_selection import StratifiedKFold

from src.dataset_b.preprocessing import export_neural_cv as export_module
from src.dataset_b.preprocessing.raw_parser import FEATURE_COLUMNS


def _row(**changes):
    row = {
        "loan_type": "1", "loan_purpose": "1", "lien_status": "1",
        "reverse_mortgage": "2", "open-end_line_of_credit": "2",
        "business_or_commercial_purpose": "2", "income": "100",
        "debt_to_income_ratio": "42", "loan_amount": "200",
        "loan_to_value_ratio": "75", "loan_term": "360", "occupancy_type": "1",
    }
    row.update(changes)
    return row


def _frame(*rows):
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS, dtype=object)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_neural_export_is_train_only_float32_hashed_and_non_overwriting(tmp_path, monkeypatch):
    X = _frame(*[_row(income=str(10 + i)) for i in range(10)])
    y = pd.Series([0, 1] * 5, dtype="int8")
    ids = pd.Series([f"record_{i}" for i in range(10)])
    accessed = []

    class FakeLoader:
        feature_version = "synthetic_test"

        def load(self, partition):
            accessed.append(partition)
            return SimpleNamespace(X=X, y=y, record_ids=ids)

    class FakeSplitter:
        @staticmethod
        def from_yaml():
            return SimpleNamespace(
                cv_splitter=lambda: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            )

    monkeypatch.setattr(export_module, "DatasetBSplitLoader", FakeLoader)
    monkeypatch.setattr(export_module, "DatasetBSplitter", FakeSplitter)
    destination = tmp_path / "cv_neural"
    manifest_path = export_module.export_neural_cv_preprocessed(destination)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert accessed == ["train"]
    assert manifest["preprocessing_version"] == "dataset_b_neural_preprocessing_v1"
    assert manifest["source_partition"] == "train"
    assert manifest["fit_scope"] == "fold_train_only"
    assert manifest["use"] == "cross_validation_only"
    assert manifest["fold_count"] == 5 and manifest["matrix_count"] == 10
    assert manifest["encoded_feature_count"] == 66
    assert manifest["matrix_format"] == "csr" and manifest["matrix_dtype"] == "float32"
    assert len(manifest["artifact_hashes"]) == 15
    for relative, expected_hash in manifest["artifact_hashes"].items():
        assert _digest(destination / relative) == expected_hash
    for fold in manifest["folds"]:
        fold_dir = destination / fold["directory"]
        assert load_npz(fold_dir / "plain_mlp_train.npz").dtype == np.float32
        assert load_npz(fold_dir / "plain_mlp_fold_validation.npz").dtype == np.float32
        assert fold["log_audit"]["fold_train"]["finite_after_log1p"] is True
        assert fold["category_support"]["loan_type"]["MISSING"]["outcome"] == "warning_absent_both"
    with pytest.raises(FileExistsError):
        export_module.export_neural_cv_preprocessed(destination)


def test_absent_category_gate_fails_before_artifact_creation(tmp_path, monkeypatch):
    X = _frame(_row(loan_type="4"), *[_row() for _ in range(9)])
    y = pd.Series([0, 1] * 5, dtype="int8")
    ids = pd.Series([f"record_{i}" for i in range(10)])

    class FakeLoader:
        feature_version = "synthetic_test"

        def load(self, partition):
            assert partition == "train"
            return SimpleNamespace(X=X, y=y, record_ids=ids)

    folds = [(np.arange(1, 10), np.asarray([0]))] + [
        (np.asarray([i for i in range(10) if i != validation]), np.asarray([validation]))
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
    destination = tmp_path / "blocked"
    with pytest.raises(ValueError, match="absent-category gate failed before artifact creation"):
        export_module.export_neural_cv_preprocessed(destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".blocked.staging-*"))
