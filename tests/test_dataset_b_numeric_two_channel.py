"""Numeric value/status channels preserve Dataset B special states."""

import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from src.dataset_b.preprocessing import DatasetBNumericTwoChannel, DatasetBRawParser
from src.dataset_b.preprocessing.raw_parser import FEATURE_COLUMNS


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


def _parsed(*rows):
    raw = pd.DataFrame(rows, columns=FEATURE_COLUMNS, dtype=object)
    return DatasetBRawParser().parse(raw).cleaned


def test_two_channel_preserves_states_and_uses_only_fold_train_medians(caplog):
    fold_train = _parsed(
        _row(income="-5", loan_amount="100", loan_to_value_ratio="50", loan_term="120"),
        _row(income="100", loan_amount="200", loan_to_value_ratio="NA", loan_term="Exempt"),
        _row(income="NA", loan_amount="250", loan_to_value_ratio="100", loan_term="240"),
        _row(income="NA", loan_amount="300", loan_to_value_ratio="Exempt", loan_term="NA"),
    )
    validation = _parsed(_row(income="1111", loan_amount="999999", loan_to_value_ratio="2000000"))
    test = _parsed(_row(income="NA", loan_amount="400", loan_to_value_ratio="NA", loan_term="Exempt"))
    before_train = fold_train.copy(deep=True)
    before_test = test.copy(deep=True)

    with caplog.at_level(logging.INFO, logger="src.dataset_b.preprocessing.numeric_two_channel"):
        transformer = DatasetBNumericTwoChannel().fit(fold_train)
        medians = dict(transformer.imputer_.medians_)
        train_result = transformer.transform(fold_train)
        validation_result = transformer.transform(validation)
        test_result = transformer.transform(test)

    assert medians == {
        "income": 50.0, "loan_amount": 225.0,
        "loan_to_value_ratio": 75.0, "loan_term": 180.0,
    }
    assert len(train_result.columns) == 19
    assert list(train_result.columns[:3]) == ["loan_type", "loan_purpose", "lien_status"]
    assert list(train_result.columns[6:8]) == ["income_value", "income_status"]
    assert train_result.loc[0, ["income_value", "income_status"]].tolist() == [0.0, "OBSERVED"]
    assert train_result.loc[2, ["income_value", "income_status"]].tolist() == [50.0, "NA"]
    assert train_result.loc[3, ["income_value", "income_status"]].tolist() == [50.0, "NA"]
    assert train_result.loc[2, ["loan_amount_value", "loan_amount_status"]].tolist() == [250.0, "OBSERVED"]
    assert train_result.loc[1, ["loan_to_value_ratio_value", "loan_to_value_ratio_status"]].tolist() == [75.0, "NA"]
    assert train_result.loc[3, ["loan_to_value_ratio_value", "loan_to_value_ratio_status"]].tolist() == [75.0, "Exempt"]
    assert train_result.loc[1, ["loan_term_value", "loan_term_status"]].tolist() == [180.0, "Exempt"]
    assert train_result.loc[2, "preapproval"] == "2"
    assert validation_result.loc[0, ["income_value", "income_status"]].tolist() == [1111.0, "OBSERVED"]
    assert validation_result.loc[0, "loan_to_value_ratio_value"] == 2000000.0
    for column in ("income", "loan_to_value_ratio", "loan_term"):
        assert test_result.loc[0, f"{column}_value"] == medians[column]
        assert np.isfinite(test_result[f"{column}_value"]).all()
        assert pd.api.types.is_float_dtype(test_result[f"{column}_value"])
    assert test_result.loc[0, "income_status"] == "NA"
    assert test_result.loc[0, "loan_to_value_ratio_status"] == "NA"
    assert test_result.loc[0, "loan_term_status"] == "Exempt"
    assert transformer.imputer_.medians_ == medians
    pd.testing.assert_frame_equal(fold_train, before_train)
    pd.testing.assert_frame_equal(test, before_test)
    assert "Fitted Dataset B numeric two-channel rows=4" in caplog.text
    assert "Transformed Dataset B numeric two-channel rows=1" in caplog.text


def test_two_channel_is_unfitted_until_fit_and_isolated_between_folds():
    first_fold = _parsed(_row(income="10"), _row(income="20"))
    second_fold = _parsed(_row(income="1000"), _row(income="2000"))
    template = DatasetBNumericTwoChannel()
    with pytest.raises(NotFittedError):
        template.transform(first_fold)

    first = clone(template).fit(first_fold)
    second = clone(template).fit(second_fold)
    assert first.imputer_.medians_["income"] == 15.0
    assert second.imputer_.medians_["income"] == 1500.0
    assert not hasattr(template, "imputer_")


def test_two_channel_requires_observed_fold_train_value_and_parsed_input():
    all_special = _parsed(_row(income="NA"), _row(income="NA"))
    with pytest.raises(ValueError, match="no observed numeric values"):
        DatasetBNumericTwoChannel().fit(all_special)

    valid = DatasetBNumericTwoChannel().fit(_parsed(_row()))
    raw = pd.DataFrame([_row(income="100")], columns=FEATURE_COLUMNS, dtype=object)
    with pytest.raises(ValueError, match="Invalid parsed numeric income"):
        valid.transform(raw)
