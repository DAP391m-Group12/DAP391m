"""Final Dataset B blank, alignment, and exemption quality-gate checks."""

import pandas as pd
import pytest

from src.dataset_b.preprocessing import (
    EXEMPTION_FLAG_COLUMNS,
    DatasetBInputQualityGate,
    DatasetBPartition,
    InputQualityError,
)
from src.dataset_b.preprocessing.raw_parser import FEATURE_COLUMNS


AUDIT_COLUMNS = (
    "lei",
    "county_code",
    "applicant_race-1",
    "applicant_race-2",
    "applicant_race-3",
    "applicant_race-4",
    "applicant_race-5",
    "applicant_sex",
    "applicant_age",
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
        "loan_amount": "215000",
        "loan_to_value_ratio": "75",
        "loan_term": "360",
        "occupancy_type": "1",
        "preapproval": "2",
        "interest_only_payment": "2",
        "balloon_payment": "2",
    }
    row.update(changes)
    return row


def _partition(rows, *, record_ids=None, actions=None, y=None):
    size = len(rows)
    record_ids = record_ids or [f"r{index + 1}" for index in range(size)]
    actions = actions or ["1"] * size
    y = y or [1 if action in {"1", "2"} else 0 for action in actions]
    return DatasetBPartition(
        name="train",
        X=pd.DataFrame(rows, columns=FEATURE_COLUMNS, dtype=object),
        y=pd.Series(y, dtype="int8"),
        record_ids=pd.Series(record_ids, dtype="object"),
        audit=pd.DataFrame(
            [{column: f"{column}-{index}" for column in AUDIT_COLUMNS} for index in range(size)],
            columns=AUDIT_COLUMNS,
            dtype=object,
        ),
        action_taken=pd.Series(actions, dtype="object"),
    )


def test_gate_rejects_blank_rows_and_preserves_aligned_retained_payload():
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
    blank = _row(income=" nAn ", preapproval="  ")
    ordinary = _row(income="250", preapproval="1")
    partition = _partition(
        [exempt, blank, ordinary],
        record_ids=["r1", "r2", "r3"],
        actions=["1", "3", "2"],
        y=[1, 0, 1],
    )
    original_X = partition.X.copy(deep=True)
    original_ids = partition.record_ids.copy(deep=True)

    result = DatasetBInputQualityGate().apply(partition)

    pd.testing.assert_frame_equal(partition.X, original_X)
    pd.testing.assert_series_equal(partition.record_ids, original_ids)
    assert result.record_ids.tolist() == ["r1", "r3"]
    assert result.y.tolist() == [1, 1]
    assert result.action_taken.tolist() == ["1", "2"]
    assert result.quality_audit.status == "PASS"
    assert result.quality_audit.input_rows == 3
    assert result.quality_audit.retained_rows == 2
    assert result.quality_audit.rejected_rows == 1
    assert result.quality_audit.rejected_record_ids == ("r2",)
    assert result.quality_audit.blank_counts["income"] == 1
    assert result.quality_audit.blank_counts["preapproval"] == 1
    assert sum(result.quality_audit.blank_counts.values()) == 2
    assert result.quality_audit.raw_1111_counts == {
        column: 1 for column in EXEMPTION_FLAG_COLUMNS
    }
    assert result.quality_audit.normalized_exempt_counts == {
        column: 1 for column in EXEMPTION_FLAG_COLUMNS
    }
    assert result.quality_audit.semantic_1111_count == 0
    assert result.quality_audit.row_count_preserved_after_normalization
    assert result.quality_audit.row_order_preserved
    assert result.quality_audit.target_alignment_preserved
    assert result.quality_audit.record_id_alignment_preserved
    assert result.X.loc[0, "income"] == 1111.0
    assert all(result.X.loc[0, column] == "Exempt" for column in EXEMPTION_FLAG_COLUMNS)
    assert not result.X.astype(str).eq("MISSING").to_numpy().any()


def test_gate_rejects_invalid_domain_instead_of_coercing():
    partition = _partition([_row(preapproval="9")])

    with pytest.raises(InputQualityError, match="Invalid preapproval"):
        DatasetBInputQualityGate().apply(partition)


@pytest.mark.parametrize(
    ("record_ids", "actions", "y", "error"),
    [
        (["r1", "r1"], ["1", "3"], [1, 0], "nonblank and unique"),
        (["r1", "r2"], ["1", "3"], [0, 0], "target alignment"),
    ],
)
def test_gate_rejects_identifier_or_target_alignment_failure(record_ids, actions, y, error):
    partition = _partition(
        [_row(), _row()],
        record_ids=record_ids,
        actions=actions,
        y=y,
    )

    with pytest.raises(InputQualityError, match=error):
        DatasetBInputQualityGate().apply(partition)


def test_gate_rejects_wrong_schema_and_all_blank_population():
    wrong = _partition([_row()])
    wrong.X = wrong.X.drop(columns="balloon_payment")
    with pytest.raises(InputQualityError, match="exact final 15-field schema"):
        DatasetBInputQualityGate().apply(wrong)

    all_blank = _partition([_row(income="")])
    with pytest.raises(InputQualityError, match="rejected every row"):
        DatasetBInputQualityGate().apply(all_blank)
