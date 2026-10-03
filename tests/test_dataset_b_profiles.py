"""Fold-local profile and export checks for Dataset B preprocessing."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import load_npz
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.dataset_b.preprocessing import DatasetBRawParser, make_preprocessing_pipeline
from src.dataset_b.preprocessing.raw_parser import FEATURE_COLUMNS
from src.dataset_b.preprocessing import export_cv as export_module
from src.dataset_b.preprocessing.split_inputs import DatasetBPartition


def _row(**changes):
    row = {
        "loan_type": "1", "loan_purpose": "1", "lien_status": "1",
        "reverse_mortgage": "2", "open-end_line_of_credit": "2",
        "business_or_commercial_purpose": "2", "income": "100",
        "debt_to_income_ratio": "42", "loan_amount": "200",
        "loan_to_value_ratio": "75", "loan_term": "360",
        "occupancy_type": "1", "preapproval": "2",
        "interest_only_payment": "2", "balloon_payment": "2",
    }
    row.update(changes)
    return row


def _frame(*rows):
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS, dtype=object)


def _semantic_frame(*rows):
    return DatasetBRawParser().parse(_frame(*rows)).cleaned


def test_profiles_are_cloneable_and_scale_only_numeric_fold_train_values():
    train = _semantic_frame(
        _row(income="-5", loan_amount="100", loan_to_value_ratio="50", loan_term="120"),
        _row(income="100", loan_amount="200", loan_to_value_ratio="75", loan_term="240"),
        _row(income="200", loan_amount="300", loan_to_value_ratio="100", loan_term="360"),
        _row(income="NA", loan_amount="400", loan_to_value_ratio="Exempt", loan_term="NA"),
    )
    held_out = _semantic_frame(
        _row(income="100000000", loan_amount="500", loan_to_value_ratio="NA", loan_term="Exempt"),
        _row(income="NA", loan_amount="9999", loan_to_value_ratio="2000", loan_term="NA"),
    )
    original = held_out.copy(deep=True)

    lr = clone(make_preprocessing_pipeline("lr"))
    rf = clone(make_preprocessing_pipeline("rf"))
    xgb = clone(make_preprocessing_pipeline("xgb"))
    lr_train = lr.fit_transform(train)
    rf_train = rf.fit_transform(train)
    xgb_train = xgb.fit_transform(train)
    lr_hold = lr.transform(held_out)
    rf_hold = rf.transform(held_out)

    assert lr_train.shape == rf_train.shape == xgb_train.shape == (4, 62)
    assert lr_hold.shape == rf_hold.shape == (2, 62)
    assert (rf_train != xgb_train).nnz == 0
    assert (lr_train[:, 4:] != rf_train[:, 4:]).nnz == 0
    assert (lr_hold[:, 4:] != rf_hold[:, 4:]).nnz == 0
    assert len(lr.get_feature_names_out()) == 62
    assert lr.get_feature_names_out().tolist() == rf.get_feature_names_out().tolist()
    assert rf_train[0, 0] == 0.0  # Negative income is mapped to zero.
    assert rf_hold[0, 0] == 100000000.0
    assert rf_hold[1, 2] == 2000.0  # Extreme LTV stays in the cohort.
    assert rf_hold[1, 0] == 100.0  # NA uses fold-train median placeholder.
    assert rf_hold[0, 2] == 75.0  # NA uses median in value channel.
    assert rf_hold[0, 3] == 240.0  # Exempt uses median in value channel.
    assert lr.named_steps["two_channel"].imputer_.medians_["income"] == 100.0
    assert lr.named_steps["scale_numeric"].scaler_.center_[0] == pytest.approx(np.log1p(100.0))
    assert lr_hold[0, 0] == pytest.approx(
        (np.log1p(100000000.0) - np.log1p(100.0)) / lr.named_steps["scale_numeric"].scaler_.scale_[0]
    )
    assert lr_hold[1, 0] == 0.0
    pd.testing.assert_frame_equal(held_out, original)
    assert lr_train.dtype == np.float64
    assert np.isfinite(lr_train.data).all()
    logged = lr.named_steps["selective_log1p"].transform(
        lr.named_steps["two_channel"].transform(train)
    )
    assert logged.loc[0, "loan_to_value_ratio_value"] == 50.0
    assert logged.loc[0, "loan_term_value"] == 120.0


def test_invalid_profile_is_rejected():
    with pytest.raises(ValueError, match="lr, rf or xgb"):
        make_preprocessing_pipeline("other")


def test_cv_export_default_path_and_version_are_new():
    assert export_module.DEFAULT_OUTPUT_DIR.name == "cv_preprocessing_v2"
    assert export_module.PREPROCESSING_VERSION == "dataset_b_preprocessing_v2"


def test_profile_can_be_swapped_inside_sklearn_model_pipeline():
    X = _semantic_frame(_row(income="10"), _row(income="20"), _row(income="30"), _row(income="40"))
    y = [0, 1, 0, 1]
    model = Pipeline([
        ("preprocess", make_preprocessing_pipeline("lr")),
        ("estimator", DummyClassifier(strategy="prior")),
    ])
    assert clone(model).fit(X, y).predict_proba(X).shape == (4, 2)
    model.set_params(preprocess=make_preprocessing_pipeline("rf"))
    assert model.fit(X, y).predict_proba(X).shape == (4, 2)
    assert model.named_steps["preprocess"].named_steps["scale_numeric"] == "passthrough"
    assert model.named_steps["preprocess"].named_steps["selective_log1p"] == "passthrough"


def test_cv_export_uses_only_train_and_preserves_fold_rows(tmp_path, monkeypatch):
    X = _frame(*[_row(income=str(50 + i * 10)) for i in range(10)])
    y = pd.Series([0, 1] * 5, dtype="int8")
    ids = pd.Series([f"record_{i}" for i in range(10)])
    action_taken = y.map({1: "1", 0: "3"})
    accessed = []

    class FakeLoader:
        feature_version = "synthetic_test"

        def load(self, partition):
            accessed.append(partition)
            assert partition == "train"
            return DatasetBPartition(
                name="train",
                X=X,
                y=y,
                record_ids=ids,
                audit=pd.DataFrame(index=X.index),
                action_taken=action_taken,
            )

    class FakeSplitter:
        @staticmethod
        def from_yaml():
            return SimpleNamespace(cv_splitter=lambda: StratifiedKFold(n_splits=5, shuffle=True, random_state=42))

    monkeypatch.setattr(export_module, "DatasetBSplitLoader", FakeLoader)
    monkeypatch.setattr(export_module, "DatasetBSplitter", FakeSplitter)
    manifest_path = export_module.export_cv_preprocessed("run_test_001", tmp_path / "cv")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert accessed == ["train"]
    assert manifest_path.parent == tmp_path / "cv" / "run_test_001"
    assert manifest["run_id"] == "run_test_001"
    assert manifest["preprocessing_version"] == "dataset_b_preprocessing_v2"
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
    assert manifest["status_reference_categories"] == {
        "income_status": "OBSERVED",
        "loan_amount_status": "OBSERVED",
        "loan_to_value_ratio_status": "OBSERVED",
        "loan_term_status": "OBSERVED",
    }
    assert manifest["tree_profile_shared_by"] == ["rf", "xgb"]
    assert manifest["sample_weights_exported"] is False
    assert set(manifest["library_versions"]) == {
        "numpy", "pandas", "scikit-learn", "scipy", "xgboost",
    }
    assert len(manifest["output_columns"]) == 62
    assert len(manifest["output_raw_sources"]) == 62
    assert len(manifest["output_categories"]) == 62
    assert "OBSERVED" not in manifest["output_categories"]
    assert "MISSING" not in manifest["output_categories"]
    assert "1111" not in manifest["output_categories"]
    assert manifest["blank_row_audit"]["rejected_rows"] == 0
    assert set(manifest["blank_row_audit"]["blank_counts"]) == set(FEATURE_COLUMNS)
    assert manifest["normalization_audit"]["raw_equals_normalized"] is True
    assert manifest["normalization_audit"]["semantic_1111_count"] == 0
    assert manifest["numeric_transforms"]["lr"] == {
        "log1p_columns": ["income_value", "loan_amount_value"],
        "robust_scaled_columns": [
            "income_value", "loan_amount_value",
            "loan_to_value_ratio_value", "loan_term_value",
        ],
    }
    assert manifest["matrix_contract"] == {
        "format": "csr", "dtype": "float64", "finite": True,
    }
    assert manifest["fold_validation_coverage"] == {
        "rows": 10,
        "minimum_assignments": 1,
        "maximum_assignments": 1,
        "exactly_once": True,
    }
    assert len(manifest["folds"]) == 5
    seen_validation = []
    for fold in manifest["folds"]:
        fold_dir = manifest_path.parent / fold["directory"]
        with np.load(fold_dir / "rows.npz", allow_pickle=False) as rows:
            train_positions = rows["train_positions"]
            validation_positions = rows["fold_validation_positions"]
            assert not set(train_positions) & set(validation_positions)
            assert rows["train_record_ids"].tolist() == ids.iloc[train_positions].tolist()
            assert rows["fold_validation_record_ids"].tolist() == ids.iloc[validation_positions].tolist()
            assert rows["train_y"].tolist() == y.iloc[train_positions].tolist()
            assert rows["fold_validation_y"].tolist() == y.iloc[validation_positions].tolist()
            assert fold["profiles"]["lr"]["numeric_medians"]["income"] == float(np.median([50 + i * 10 for i in train_positions]))
            assert fold["profiles"]["lr"]["selective_log1p_columns"] == [
                "income_value", "loan_amount_value",
            ]
            assert len(fold["profiles"]["lr"]["robust_center"]) == 4
            assert len(fold["profiles"]["lr"]["robust_scale"]) == 4
            seen_validation.extend(validation_positions.tolist())
        for profile in ("lr", "tree"):
            train_matrix = load_npz(fold_dir / f"{profile}_train.npz")
            validation_matrix = load_npz(fold_dir / f"{profile}_fold_validation.npz")
            assert train_matrix.shape == (8, 62)
            assert validation_matrix.shape == (2, 62)
            assert train_matrix.format == validation_matrix.format == "csr"
            assert train_matrix.dtype == validation_matrix.dtype == np.float64
            assert np.isfinite(train_matrix.data).all()
            assert np.isfinite(validation_matrix.data).all()
    assert sorted(seen_validation) == list(range(10))
    with pytest.raises(FileExistsError):
        export_module.export_cv_preprocessed("run_test_001", tmp_path / "cv")


@pytest.mark.parametrize("run_id", ["", "../escape", "bad/run", "bad run", ".hidden"])
def test_cv_export_rejects_invalid_run_id(run_id, tmp_path):
    with pytest.raises(ValueError, match="run_id"):
        export_module.export_cv_preprocessed(run_id, tmp_path / "cv")
