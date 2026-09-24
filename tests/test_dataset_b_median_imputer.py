"""Fold-only median statistics and held-out transform checks."""

import logging

import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from src.dataset_b.preprocessing import DatasetBNumericMedianImputer, DatasetBRawParser
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
    return DatasetBRawParser().parse(pd.DataFrame(rows, columns=FEATURE_COLUMNS, dtype=object)).cleaned


def test_medians_fit_only_on_fold_train_and_do_not_change_on_held_out_transform(caplog):
    fold_train = _parsed(
        _row(income="-5", loan_amount="100", loan_to_value_ratio="50", loan_term="120"),
        _row(income="100", loan_amount="200", loan_to_value_ratio="NA", loan_term="Exempt"),
        _row(income="NA", loan_amount="", loan_to_value_ratio="100", loan_term="240", loan_type=" "),
        _row(income=" ", loan_amount="300", loan_to_value_ratio="Exempt", loan_term="NA"),
    )
    validation = _parsed(_row(income="999999", loan_amount="999999", loan_to_value_ratio="999999", loan_term="999999"))
    test = _parsed(_row(income="", loan_amount=" ", loan_to_value_ratio="NaN", loan_term=None,
                        debt_to_income_ratio=" "))
    train_before = fold_train.copy(deep=True)
    test_before = test.copy(deep=True)

    with caplog.at_level(logging.INFO, logger="src.dataset_b.preprocessing.median_imputer"):
        imputer = DatasetBNumericMedianImputer().fit(fold_train)
        train_result = imputer.transform(fold_train)
        validation_result = imputer.transform(validation)
        test_result = imputer.transform(test)

    assert imputer.medians_ == {
        "income": 50.0, "loan_amount": 200.0,
        "loan_to_value_ratio": 75.0, "loan_term": 180.0,
    }
    assert imputer.observed_counts_ == {
        "income": 2, "loan_amount": 3, "loan_to_value_ratio": 2, "loan_term": 2,
    }
    assert test_result.loc[0, ["income", "loan_amount", "loan_to_value_ratio", "loan_term"]].tolist() == [
        50.0, 200.0, 75.0, 180.0,
    ]
    assert train_result.loc[2, "income"] == "NA"
    assert train_result.loc[1, "loan_to_value_ratio"] == "NA"
    assert train_result.loc[3, "loan_to_value_ratio"] == "Exempt"
    assert train_result.loc[1, "loan_term"] == "Exempt"
    assert train_result.loc[2, "loan_type"] == "MISSING"
    assert test_result.loc[0, "debt_to_income_ratio"] == "MISSING"
    assert validation_result.loc[0, "income"] == 999999.0
    assert imputer.medians_["income"] == 50.0
    pd.testing.assert_frame_equal(fold_train, train_before)
    pd.testing.assert_frame_equal(test, test_before)
    assert "Fitted Dataset B fold-train numeric medians" in caplog.text
    assert "Transformed Dataset B rows=1" in caplog.text


def test_each_fold_uses_fresh_fitted_state():
    first_fold = _parsed(_row(income="10"), _row(income="20"))
    second_fold = _parsed(_row(income="1000"), _row(income="2000"))
    template = DatasetBNumericMedianImputer()
    first = clone(template).fit(first_fold)
    second = clone(template).fit(second_fold)

    assert first.medians_["income"] == 15.0
    assert second.medians_["income"] == 1500.0
    assert not hasattr(template, "medians_")


def test_rejects_unfitted_transform_and_fold_without_numeric_observations():
    missing = _parsed(_row(income="NA"), _row(income=" "))
    with pytest.raises(NotFittedError):
        DatasetBNumericMedianImputer().transform(missing)
    with pytest.raises(ValueError, match="no observed numeric values"):
        DatasetBNumericMedianImputer().fit(missing)
