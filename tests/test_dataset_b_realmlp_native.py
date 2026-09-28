"""Contract tests for Dataset B RealMLP native preprocessing."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src.dataset_b.preprocessing import (
    DatasetBRealMLPNativeAdapter,
    RealMLPCategorySupportError,
    make_realmlp_native_preprocessing_pipeline,
    validate_realmlp_category_support,
)
from src.dataset_b.preprocessing.categorical_encoder import (
    CATEGORICAL_COLUMNS,
    NUMERIC_VALUE_COLUMNS,
    TWO_CHANNEL_COLUMNS,
)
from src.dataset_b.preprocessing.raw_parser import FEATURE_COLUMNS


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
    }
    row.update(changes)
    return row


def _frame(*rows):
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS, dtype=object)


def test_realmlp_native_profile_is_fold_local_typed_and_non_mutating():
    train = _frame(
        _row(income="0", loan_amount="100", loan_to_value_ratio="NA"),
        _row(
            income="100",
            loan_amount="300",
            loan_to_value_ratio="Exempt",
            reverse_mortgage="1111",
        ),
        _row(income="", loan_amount="", loan_term="NA"),
    )
    held_out = _frame(
        _row(income="", loan_amount="400", loan_to_value_ratio="NA", loan_term="NA")
    )
    train_before = train.copy(deep=True)
    held_out_before = held_out.copy(deep=True)

    profile = clone(make_realmlp_native_preprocessing_pipeline())
    train_native = profile.fit_transform(train)
    held_out_native = profile.transform(held_out)

    assert tuple(train_native.columns) == tuple(held_out_native.columns) == TWO_CHANNEL_COLUMNS
    assert all(train_native[column].dtype == np.float32 for column in NUMERIC_VALUE_COLUMNS)
    assert all(held_out_native[column].dtype == np.float32 for column in NUMERIC_VALUE_COLUMNS)
    assert all(isinstance(train_native[column].dtype, pd.CategoricalDtype) for column in CATEGORICAL_COLUMNS)
    assert all(isinstance(held_out_native[column].dtype, pd.CategoricalDtype) for column in CATEGORICAL_COLUMNS)
    assert np.isfinite(train_native.loc[:, NUMERIC_VALUE_COLUMNS].to_numpy()).all()
    assert np.isfinite(held_out_native.loc[:, NUMERIC_VALUE_COLUMNS].to_numpy()).all()
    assert held_out_native.loc[0, "income_value"] == np.float32(np.log1p(50.0))
    assert held_out_native.loc[0, "income_status"] == "MISSING"
    assert held_out_native.loc[0, "loan_to_value_ratio_status"] == "NA"
    assert held_out_native.loc[0, "loan_term_status"] == "NA"
    assert train_native.loc[1, "loan_to_value_ratio_status"] == "Exempt"
    assert train_native.loc[1, "reverse_mortgage"] == "1111"
    assert profile.named_steps["native"].last_audit_["numeric_dtype"] == "float32"
    pd.testing.assert_frame_equal(train, train_before)
    pd.testing.assert_frame_equal(held_out, held_out_before)


def test_realmlp_category_support_reports_absent_both_and_rejects_unseen_held_out():
    profile = make_realmlp_native_preprocessing_pipeline()
    train_native = profile.fit_transform(_frame(_row(), _row(income="NA")))

    report = validate_realmlp_category_support(train_native, train_native.iloc[[0]])
    assert report["loan_type"]["4"] == {
        "fold_train": 0,
        "fold_validation": 0,
        "outcome": "warning_absent_both",
    }

    held_out_channels = train_native.iloc[[0]].copy(deep=True)
    held_out_channels["loan_type"] = held_out_channels["loan_type"].cat.set_categories(
        train_native["loan_type"].cat.categories
    )
    held_out_channels.loc[:, "loan_type"] = "4"
    with pytest.raises(RealMLPCategorySupportError, match="loan_type"):
        validate_realmlp_category_support(train_native, held_out_channels)
    with pytest.raises(RealMLPCategorySupportError, match="loan_type"):
        profile.named_steps["native"].transform(held_out_channels)


def test_realmlp_native_adapter_is_cloneable_and_rejects_invalid_boundary():
    profile = make_realmlp_native_preprocessing_pipeline()
    channels = profile[:-1].fit_transform(_frame(_row(), _row(income="NA")))
    template = DatasetBRealMLPNativeAdapter()
    first = clone(template).fit(channels)
    second = clone(template).fit(channels)

    assert not hasattr(template, "fold_train_category_counts_")
    assert first.fold_train_category_counts_ == second.fold_train_category_counts_
    assert first.get_feature_names_out().tolist() == list(TWO_CHANNEL_COLUMNS)

    nonfinite = channels.copy(deep=True)
    nonfinite.loc[0, "income_value"] = np.inf
    with pytest.raises(ValueError, match="finite numbers"):
        first.transform(nonfinite)

    unknown = channels.copy(deep=True)
    unknown.loc[0, "loan_type"] = "9"
    with pytest.raises(ValueError, match="Unexpected Dataset B category"):
        first.transform(unknown)

    with pytest.raises(ValueError, match="16 two-channel columns"):
        first.transform(channels.drop(columns="loan_type"))
