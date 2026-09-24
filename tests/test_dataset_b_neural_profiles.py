"""Tests for the separate Dataset B plain-MLP preprocessing profile."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import clone

from src.dataset_b.preprocessing import (
    make_plain_mlp_preprocessing_pipeline, make_preprocessing_pipeline,
)
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


def test_plain_mlp_profile_is_fold_local_float32_and_does_not_change_classical_profile():
    train = _frame(
        _row(income="0", loan_amount="100"),
        _row(income="100", loan_amount="200"),
        _row(income="NA", loan_amount="300", loan_to_value_ratio="Exempt"),
    )
    held_out = _frame(_row(income="", loan_amount="400", loan_term="NA"))
    neural = clone(make_plain_mlp_preprocessing_pipeline())
    classical = clone(make_preprocessing_pipeline("rf"))

    neural_train = neural.fit_transform(train)
    neural_hold = neural.transform(held_out)
    classical_train = classical.fit_transform(train)

    assert isinstance(neural_train, csr_matrix)
    assert neural_train.shape == (3, 66)
    assert neural_hold.shape == (1, 66)
    assert neural_train.dtype == neural_hold.dtype == np.float32
    assert classical_train.dtype == np.float64
    assert np.isfinite(neural_train.data).all() and np.isfinite(neural_hold.data).all()
    assert neural.named_steps["two_channel"].imputer_.medians_["income"] == 50.0
    assert neural_hold[0, 0] == np.float32(np.log1p(50.0))
    np.testing.assert_allclose(neural_train[:, 4:].toarray(), classical_train[:, 4:].toarray())
    assert neural.get_feature_names_out().tolist() == classical.get_feature_names_out().tolist()
