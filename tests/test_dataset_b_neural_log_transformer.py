"""Tests for the plain-MLP numeric log boundary."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src.dataset_b.preprocessing import (
    DatasetBNeuralLog1pTransformer, DatasetBNumericTwoChannel, DatasetBRawParser,
)
from src.dataset_b.preprocessing.categorical_encoder import NUMERIC_VALUE_COLUMNS
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


def _channels(*rows):
    raw = pd.DataFrame(rows, columns=FEATURE_COLUMNS, dtype=object)
    parsed = DatasetBRawParser().parse(raw).cleaned
    return DatasetBNumericTwoChannel().fit_transform(parsed)


def test_log1p_changes_only_numeric_values_and_is_cloneable():
    channels = _channels(
        _row(income="0", loan_amount="99"),
        _row(income="NA", loan_amount="199", loan_to_value_ratio="Exempt"),
    )
    before = channels.copy(deep=True)
    transformer = clone(DatasetBNeuralLog1pTransformer())
    result = transformer.fit_transform(channels)

    np.testing.assert_allclose(
        result.loc[:, NUMERIC_VALUE_COLUMNS].to_numpy(dtype=float),
        np.log1p(before.loc[:, NUMERIC_VALUE_COLUMNS].to_numpy(dtype=float)),
    )
    pd.testing.assert_frame_equal(
        result.drop(columns=list(NUMERIC_VALUE_COLUMNS)),
        before.drop(columns=list(NUMERIC_VALUE_COLUMNS)),
    )
    pd.testing.assert_frame_equal(channels, before)
    pd.testing.assert_frame_equal(transformer.transform(channels), result)
    assert transformer.last_audit_["minimum_before_log1p"] == 0.0
    assert transformer.last_audit_["finite_after_log1p"] is True


def test_log1p_rejects_negative_nonfinite_and_wrong_schema():
    channels = _channels(_row())
    negative = channels.copy(deep=True)
    negative.loc[0, "income_value"] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        DatasetBNeuralLog1pTransformer().fit(negative)

    nonfinite = channels.copy(deep=True)
    nonfinite.loc[0, "loan_amount_value"] = np.inf
    with pytest.raises(ValueError, match="finite numbers"):
        DatasetBNeuralLog1pTransformer().fit(nonfinite)

    with pytest.raises(ValueError, match="16 two-channel columns"):
        DatasetBNeuralLog1pTransformer().fit(channels.drop(columns="loan_type"))
