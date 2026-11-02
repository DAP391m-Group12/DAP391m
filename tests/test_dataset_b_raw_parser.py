"""Raw-token validation and exemption normalization for the final 15 features."""

import logging

import pandas as pd
import pytest

from src.dataset_b.data import (
    EXEMPTION_FLAG_COLUMNS,
    FEATURE_COLUMNS,
    DatasetBRawParser,
    RawFeatureError,
)


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
        "loan_amount": "215000.0",
        "loan_to_value_ratio": "75.0",
        "loan_term": "360",
        "occupancy_type": "1",
        "preapproval": "2",
        "interest_only_payment": "2",
        "balloon_payment": "2",
    }
    row.update(changes)
    return row


def _frame(*rows):
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS, dtype=object)


def test_parser_normalizes_five_exemption_flags_and_preserves_numeric_1111(caplog):
    exempt = _row(
        reverse_mortgage="1111",
        **{
            "open-end_line_of_credit": "1111",
            "business_or_commercial_purpose": "1111",
        },
        income="1111",
        interest_only_payment="1111",
        balloon_payment="1111",
    )
    X = _frame(exempt)
    original = X.copy(deep=True)

    with caplog.at_level(logging.INFO, logger="src.dataset_b.data.raw_parser"):
        parsed = DatasetBRawParser().parse(X)

    pd.testing.assert_frame_equal(X, original)
    pd.testing.assert_frame_equal(parsed.raw, original)
    assert parsed.cleaned.loc[0, "income"] == 1111.0
    assert all(parsed.cleaned.loc[0, column] == "Exempt" for column in EXEMPTION_FLAG_COLUMNS)
    assert parsed.diagnostics.raw_1111_counts == {
        column: 1 for column in EXEMPTION_FLAG_COLUMNS
    }
    assert parsed.diagnostics.normalized_exempt_counts == {
        column: 1 for column in EXEMPTION_FLAG_COLUMNS
    }
    assert parsed.diagnostics.semantic_1111_count == 0
    assert "raw_1111=5" in caplog.text


def test_parser_preserves_special_labels_and_maps_negative_income_only(caplog):
    negative = _row(income="-5", loan_to_value_ratio="1001")
    zero = _row(
        income="0",
        loan_to_value_ratio="Exempt",
        loan_term="NA",
        debt_to_income_ratio="Exempt",
    )
    special = _row(
        income="NA",
        loan_to_value_ratio="NA",
        loan_term="Exempt",
        debt_to_income_ratio="<20%",
        loan_purpose="5",
    )
    X = _frame(negative, zero, special)
    original = X.copy(deep=True)

    with caplog.at_level(logging.INFO, logger="src.dataset_b.data.raw_parser"):
        parsed = DatasetBRawParser().parse(X)

    pd.testing.assert_frame_equal(X, original)
    assert parsed.cleaned.loc[0, "income"] == 0.0
    assert parsed.cleaned.loc[1, "income"] == 0.0
    assert parsed.cleaned.loc[2, "income"] == "NA"
    assert parsed.cleaned.loc[0, "loan_to_value_ratio"] == 1001.0
    assert parsed.cleaned.loc[1, "loan_to_value_ratio"] == "Exempt"
    assert parsed.cleaned.loc[2, "loan_to_value_ratio"] == "NA"
    assert parsed.cleaned.loc[1, "debt_to_income_ratio"] == "Exempt"
    assert parsed.diagnostics.income_negative_to_zero == 1
    assert parsed.diagnostics.ltv_gt_1000 == 1
    assert "income_negative_to_zero=1" in caplog.text
    assert "LTV >1000% retained" in caplog.text


@pytest.mark.parametrize(
    ("column", "value", "error"),
    [
        ("loan_type", "9", "Invalid loan_type"),
        ("preapproval", "3", "Invalid preapproval"),
        ("interest_only_payment", "Exempt", "Invalid interest_only_payment"),
        ("balloon_payment", "NA", "Invalid balloon_payment"),
        ("loan_amount", "NA", "Invalid numeric loan_amount"),
        ("loan_term", "-1", "Nonpositive numeric loan_term"),
        ("income", "not-a-number", "Invalid numeric income"),
        ("debt_to_income_ratio", "35", "Invalid DTI"),
    ],
)
def test_parser_rejects_out_of_domain(column, value, error):
    with pytest.raises(RawFeatureError, match=error):
        DatasetBRawParser().parse(_frame(_row(**{column: value})))


@pytest.mark.parametrize("blank", [None, "", " \t ", "NaN", " nAn ", float("nan"), pd.NA])
def test_parser_rejects_blank_instead_of_creating_missing(blank):
    with pytest.raises(RawFeatureError, match="run the input-quality gate first"):
        DatasetBRawParser().parse(_frame(_row(income=blank)))


def test_parser_rejects_columns_outside_final_fifteen():
    X = _frame(_row())
    X["action_taken"] = "1"
    with pytest.raises(RawFeatureError, match="15 final features"):
        DatasetBRawParser().parse(X)
