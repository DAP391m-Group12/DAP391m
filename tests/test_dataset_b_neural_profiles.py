"""Tests for the final Dataset B PlainMLP preprocessing profile."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import clone

from src.dataset_b.data.numeric_value_scaler import DatasetBNumericValueRobustScaler
from src.dataset_b.preprocessing import make_plain_mlp_preprocessing_pipeline
from src.dataset_b.preprocessing.categorical_encoder import (
    ENCODED_FEATURE_COUNT,
    NUMERIC_VALUE_COLUMNS,
)
from src.dataset_b.preprocessing.numeric_two_channel import TWO_CHANNEL_COLUMNS
from src.dataset_b.preprocessing.raw_parser import FEATURE_COLUMNS


def _row(**changes):
    row = {
        "loan_type": "1",
        "loan_purpose": "1",
        "lien_status": "1",
        "reverse_mortgage": "2",
        "open-end_line_of_credit": "2",
        "business_or_commercial_purpose": "2",
        "income": 100.0,
        "debt_to_income_ratio": "42",
        "loan_amount": 200.0,
        "loan_to_value_ratio": 75.0,
        "loan_term": 360.0,
        "occupancy_type": "1",
        "preapproval": "2",
        "interest_only_payment": "2",
        "balloon_payment": "2",
    }
    row.update(changes)
    return row


def _frame(*rows):
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS, dtype=object)


def test_plain_mlp_profile_is_normalized_15_to_19_to_62_float64_and_fit_local():
    train = _frame(
        _row(income=0.0, loan_amount=100.0),
        _row(income=100.0, loan_amount=200.0),
        _row(
            income="NA",
            loan_amount=300.0,
            loan_to_value_ratio="Exempt",
            reverse_mortgage="Exempt",
        ),
    )
    held_out = _frame(
        _row(income="NA", loan_amount=400.0, loan_to_value_ratio=90.0, loan_term="NA")
    )
    profile = clone(make_plain_mlp_preprocessing_pipeline())

    assert isinstance(profile.named_steps["scale_numeric"], DatasetBNumericValueRobustScaler)
    assert profile.named_steps["scale_numeric"].__class__.__module__ == (
        "src.dataset_b.data.numeric_value_scaler"
    )

    train_matrix = profile.fit_transform(train)
    held_matrix = profile.transform(held_out)

    assert "parse" not in profile.named_steps
    assert tuple(train.columns) == FEATURE_COLUMNS
    assert profile.named_steps["two_channel"].get_feature_names_out().tolist() == list(
        TWO_CHANNEL_COLUMNS
    )
    assert profile.get_feature_names_out().tolist() == profile.named_steps[
        "encode"
    ].get_feature_names_out().tolist()
    assert len(profile.get_feature_names_out()) == ENCODED_FEATURE_COUNT == 62
    assert isinstance(train_matrix, csr_matrix)
    assert train_matrix.shape == (3, 62) and held_matrix.shape == (1, 62)
    assert train_matrix.dtype == held_matrix.dtype == np.float64
    assert np.isfinite(train_matrix.data).all() and np.isfinite(held_matrix.data).all()
    assert profile.named_steps["two_channel"].imputer_.medians_["income"] == 50.0
    assert "MISSING" not in profile.named_steps["encode"].feature_categories_
    assert "Exempt" in profile.named_steps["encode"].feature_categories_

    channels = profile.named_steps["two_channel"].transform(held_out)
    logged = profile.named_steps["log1p_numeric"].transform(channels)
    encoded = profile.named_steps["encode"].transform(logged)
    assert logged.loc[0, "income_value"] == np.log1p(50.0)
    assert logged.loc[0, "loan_amount_value"] == np.log1p(400.0)
    assert logged.loc[0, "loan_to_value_ratio_value"] == 90.0
    assert logged.loc[0, "loan_term_value"] == 360.0
    np.testing.assert_array_equal(
        held_matrix[:, len(NUMERIC_VALUE_COLUMNS) :].data,
        encoded[:, len(NUMERIC_VALUE_COLUMNS) :].data,
    )


def test_plain_mlp_profile_never_densifies_more_than_four_numeric_columns(monkeypatch):
    train = _frame(*[_row(income=float(index + 1)) for index in range(6)])
    original = csr_matrix.toarray
    widths: list[int] = []

    def guarded_toarray(self, *args, **kwargs):
        widths.append(self.shape[1])
        if self.shape[1] > len(NUMERIC_VALUE_COLUMNS):
            raise AssertionError("full matrix densification is forbidden")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(csr_matrix, "toarray", guarded_toarray)
    result = make_plain_mlp_preprocessing_pipeline().fit_transform(train)

    assert result.shape == (6, 62)
    assert widths and max(widths) == 4
