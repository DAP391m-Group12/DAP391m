"""Fixed-category one-hot checks for Dataset B's three tabular models."""

import logging
import re

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from src.dataset_b.preprocessing import (
    DatasetBOneHotEncoder, DatasetBNumericTwoChannel, DatasetBRawParser,
)
from src.dataset_b.preprocessing.raw_parser import FEATURE_COLUMNS


def _row(**changes):
    row = {
        "loan_type": "1", "loan_purpose": "1", "lien_status": "1",
        "reverse_mortgage": "2", "open-end_line_of_credit": "2",
        "business_or_commercial_purpose": "2", "income": "100",
        "debt_to_income_ratio": "42", "loan_amount": "200",
        "loan_to_value_ratio": "75", "loan_term": "360",
        "occupancy_type": "1",
    }
    row.update(changes)
    return row


def _parsed(*rows):
    raw = pd.DataFrame(rows, columns=FEATURE_COLUMNS, dtype=object)
    return DatasetBRawParser().parse(raw).cleaned


def _column_position(encoder, input_column, category):
    safe_prefix = input_column.replace("-", "_") + "__"
    return next(
        i for i, (name, value) in enumerate(
            zip(encoder.get_feature_names_out(), encoder.feature_categories_)
        ) if name.startswith(safe_prefix) and value == category
    )


def test_ohe_preserves_special_categories_unseen_in_fold_train(caplog):
    train = _parsed(_row(), _row(income="NA", loan_to_value_ratio="NA", loan_term="Exempt"))
    held_out = _parsed(
        _row(loan_type="", reverse_mortgage="1111", income="", loan_amount=" ",
             loan_to_value_ratio="Exempt", loan_term="NA", debt_to_income_ratio="Exempt"),
        _row(loan_type="4", reverse_mortgage="", income="1111", loan_to_value_ratio="NaN",
             loan_term="", debt_to_income_ratio="NA"),
    )
    two_channel = DatasetBNumericTwoChannel().fit(train)
    train_channels = two_channel.transform(train)
    held_out_channels = two_channel.transform(held_out)
    before = held_out_channels.copy(deep=True)

    with caplog.at_level(logging.INFO, logger="src.dataset_b.preprocessing.categorical_encoder"):
        encoder = DatasetBOneHotEncoder().fit(train_channels)
        result = encoder.transform(held_out_channels)

    assert isinstance(result, csr_matrix)
    assert result.shape == (2, 66)
    assert np.isfinite(result.data).all()
    assert len(encoder.feature_names_out_) == len(encoder.feature_sources_) == len(encoder.feature_categories_) == 66
    assert all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) for name in encoder.get_feature_names_out())
    assert len(set(encoder.get_feature_names_out())) == 66
    assert "OBSERVED" not in encoder.feature_categories_
    assert encoder.feature_categories_.count("MISSING") == 12
    assert encoder.feature_sources_[_column_position(encoder, "income_status", "MISSING")] == "income"
    assert encoder.feature_sources_[_column_position(encoder, "loan_to_value_ratio_status", "Exempt")] == "loan_to_value_ratio"
    assert encoder.feature_sources_[_column_position(encoder, "lien_status", "1")] == "lien_status"
    assert set(encoder.feature_sources_) == set(FEATURE_COLUMNS)
    assert encoder.feature_categories_[:4] == (None, None, None, None)
    assert encoder.transform(train_channels.iloc[[0]])[:, 4:].sum() == 8.0
    assert result[:, 4:].sum(axis=1).A1.tolist() == [12.0, 10.0]
    for row, column, category in (
        (0, "loan_type", "MISSING"),
        (0, "reverse_mortgage", "1111"),
        (0, "income_status", "MISSING"),
        (0, "loan_amount_status", "MISSING"),
        (0, "loan_to_value_ratio_status", "Exempt"),
        (0, "loan_term_status", "NA"),
        (0, "debt_to_income_ratio", "Exempt"),
        (1, "loan_type", "4"),
        (1, "reverse_mortgage", "MISSING"),
        (1, "loan_to_value_ratio_status", "MISSING"),
        (1, "debt_to_income_ratio", "NA"),
    ):
        assert result[row, _column_position(encoder, column, category)] == 1.0
    assert result[1, 0] == 1111.0
    pd.testing.assert_frame_equal(held_out_channels, before)
    assert "Fitted Dataset B OHE rows=2" in caplog.text
    assert "Transformed Dataset B OHE rows=2 columns=66" in caplog.text


def test_ohe_columns_remain_identical_across_fold_fits():
    first = DatasetBNumericTwoChannel().fit_transform(_parsed(_row(loan_type="1")))
    second = DatasetBNumericTwoChannel().fit_transform(_parsed(_row(loan_type="4")))
    template = DatasetBOneHotEncoder()
    with pytest.raises(NotFittedError):
        template.transform(first)
    first_encoder = clone(template).fit(first)
    second_encoder = clone(template).fit(second)

    assert first_encoder.get_feature_names_out().tolist() == second_encoder.get_feature_names_out().tolist()
    assert first_encoder.transform(first).shape == second_encoder.transform(second).shape == (1, 66)
    assert not hasattr(template, "encoder_")


def test_ohe_dtype_is_configurable_without_changing_classical_default():
    channels = DatasetBNumericTwoChannel().fit_transform(_parsed(_row()))
    classical = DatasetBOneHotEncoder().fit_transform(channels)
    explicit_classical = DatasetBOneHotEncoder(dtype=np.float64).fit_transform(channels)
    neural = DatasetBOneHotEncoder(dtype=np.float32).fit_transform(channels)

    assert classical.dtype == explicit_classical.dtype == np.float64
    assert neural.dtype == np.float32
    assert (classical != explicit_classical).nnz == 0
    np.testing.assert_array_equal(classical.toarray(), neural.toarray())

    with pytest.raises(ValueError, match="float32 or float64"):
        DatasetBOneHotEncoder(dtype=np.int32).fit(channels)


def test_ohe_rejects_unknown_codes_nonfinite_values_and_wrong_schema():
    channels = DatasetBNumericTwoChannel().fit_transform(_parsed(_row()))
    encoder = DatasetBOneHotEncoder().fit(channels)

    unknown = channels.copy(deep=True)
    unknown.loc[0, "loan_type"] = "9"
    with pytest.raises(ValueError, match="Unexpected Dataset B category in loan_type"):
        encoder.transform(unknown)

    nonfinite = channels.copy(deep=True)
    nonfinite.loc[0, "income_value"] = np.nan
    with pytest.raises(ValueError, match="finite numbers"):
        encoder.transform(nonfinite)

    with pytest.raises(ValueError, match="16 two-channel columns"):
        encoder.transform(channels.drop(columns="loan_type"))
