"""Selective LR log1p contract checks."""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src.dataset_b.preprocessing import (
    DatasetBLRSelectiveLog1pTransformer,
    DatasetBNumericTwoChannel,
    DatasetBRawParser,
)
from src.dataset_b.preprocessing.raw_parser import FEATURE_COLUMNS


def _semantic(**changes):
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
    raw = pd.DataFrame([row], columns=FEATURE_COLUMNS, dtype=object)
    return DatasetBRawParser().parse(raw).cleaned


def test_lr_log1p_changes_only_income_and_loan_amount():
    channels = DatasetBNumericTwoChannel().fit_transform(_semantic())
    before = channels.copy(deep=True)
    transformer = DatasetBLRSelectiveLog1pTransformer().fit(channels)
    result = transformer.transform(channels)

    assert result.loc[0, "income_value"] == pytest.approx(np.log1p(100.0))
    assert result.loc[0, "loan_amount_value"] == pytest.approx(np.log1p(200.0))
    assert result.loc[0, "loan_to_value_ratio_value"] == 75.0
    assert result.loc[0, "loan_term_value"] == 360.0
    pd.testing.assert_frame_equal(channels, before)
    assert transformer.last_audit_["unchanged_numeric_columns"] == (
        "loan_to_value_ratio_value", "loan_term_value",
    )


def test_lr_log1p_rejects_unfitted_nonfinite_and_negative_selected_values():
    channels = DatasetBNumericTwoChannel().fit_transform(_semantic())
    with pytest.raises(NotFittedError):
        DatasetBLRSelectiveLog1pTransformer().transform(channels)
    transformer = DatasetBLRSelectiveLog1pTransformer().fit(channels)
    nonfinite = channels.copy(deep=True)
    nonfinite.loc[0, "loan_term_value"] = np.inf
    with pytest.raises(ValueError, match="finite numbers"):
        transformer.transform(nonfinite)
    negative = channels.copy(deep=True)
    negative.loc[0, "income_value"] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        transformer.transform(negative)
