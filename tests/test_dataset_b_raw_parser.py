"""Raw-token and deterministic cleaning checks for the 12-feature contract."""

import logging

import pandas as pd
import pytest

from src.dataset_b.preprocessing import DatasetBRawParser, RawFeatureError
from src.dataset_b.preprocessing.raw_parser import FEATURE_COLUMNS


def _row():
    return {
        "loan_type": "1", "loan_purpose": "1", "lien_status": "1",
        "reverse_mortgage": "2", "open-end_line_of_credit": "2",
        "business_or_commercial_purpose": "2", "income": "100",
        "debt_to_income_ratio": "42", "loan_amount": "215000.0",
        "loan_to_value_ratio": "75.0", "loan_term": "360",
        "occupancy_type": "1",
    }


def _frame(*rows):
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS, dtype=object)


def test_parser_preserves_special_labels_and_maps_negative_income_only(caplog):
    negative = _row()
    negative.update(income="-5", loan_to_value_ratio="1001", reverse_mortgage="1111")
    zero = _row()
    zero.update(income="0", loan_to_value_ratio="Exempt", loan_term="NA",
                debt_to_income_ratio="Exempt")
    special = _row()
    special.update(income="NA", loan_to_value_ratio="NA", loan_term="Exempt",
                   debt_to_income_ratio="<20%", loan_purpose="5")
    missing = _row()
    missing.update(income=float("nan"), loan_to_value_ratio=pd.NA,
                   loan_type=None, debt_to_income_ratio="NaN")
    X = _frame(negative, zero, special, missing)
    original = X.copy(deep=True)

    with caplog.at_level(logging.INFO, logger="src.dataset_b.preprocessing.raw_parser"):
        parsed = DatasetBRawParser().parse(X)

    pd.testing.assert_frame_equal(X, original)
    assert parsed.raw.loc[0, "income"] == "-5"
    assert parsed.cleaned.loc[0, "income"] == 0.0
    assert parsed.cleaned.loc[1, "income"] == 0.0
    assert parsed.cleaned.loc[2, "income"] == "NA"
    assert parsed.cleaned.loc[3, "income"] == "MISSING"
    assert parsed.cleaned.loc[0, "loan_to_value_ratio"] == 1001.0
    assert parsed.cleaned.loc[1, "loan_to_value_ratio"] == "Exempt"
    assert parsed.cleaned.loc[2, "loan_to_value_ratio"] == "NA"
    assert parsed.cleaned.loc[3, "loan_to_value_ratio"] == "MISSING"
    assert parsed.cleaned.loc[0, "reverse_mortgage"] == "1111"
    assert parsed.cleaned.loc[1, "debt_to_income_ratio"] == "Exempt"
    assert parsed.cleaned.loc[3, "debt_to_income_ratio"] == "MISSING"
    assert parsed.cleaned.loc[3, "loan_type"] == "MISSING"
    assert parsed.diagnostics.income_negative_to_zero == 1
    assert parsed.diagnostics.ltv_gt_1000 == 1
    assert parsed.diagnostics.nan_to_missing["income"] == 1
    assert "income_negative_to_zero=1" in caplog.text
    assert "LTV >1000% retained" in caplog.text


@pytest.mark.parametrize(
    ("column", "value", "error"),
    [
        ("loan_type", "9", "Invalid loan_type"),
        ("loan_amount", "NA", "Invalid numeric loan_amount"),
        ("loan_term", "-1", "Nonpositive numeric loan_term"),
        ("income", "not-a-number", "Invalid numeric income"),
        ("debt_to_income_ratio", "35", "Invalid DTI"),
    ],
)
def test_parser_rejects_out_of_domain(column, value, error):
    row = _row()
    row[column] = value
    with pytest.raises(RawFeatureError, match=error):
        DatasetBRawParser().parse(_frame(row))


def test_parser_maps_blank_and_whitespace_to_missing_in_every_column():
    blank = _row()
    whitespace = _row()
    for column in FEATURE_COLUMNS:
        blank[column] = ""
        whitespace[column] = " \t "

    parsed = DatasetBRawParser().parse(_frame(blank, whitespace))

    assert all(parsed.cleaned[column].tolist() == ["MISSING", "MISSING"] for column in FEATURE_COLUMNS)
    assert all(parsed.diagnostics.values_to_missing[column] == 2 for column in FEATURE_COLUMNS)
    assert parsed.diagnostics.nan_to_missing == parsed.diagnostics.values_to_missing


def test_parser_rejects_columns_outside_locked_twelve():
    X = _frame(_row())
    X["action_taken"] = "1"
    with pytest.raises(RawFeatureError, match="12 locked features"):
        DatasetBRawParser().parse(X)
