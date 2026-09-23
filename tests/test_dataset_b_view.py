import csv

import pytest

from src.dataset_b.preparation.dataset_view import DatasetBViewBuilder, DatasetBViewError


def _contract() -> dict:
    return {
        "dataset": "dataset_b",
        "target": {
            "source_column": "action_taken",
            "eligible_values": [1, 2, 3],
            "positive_values": [1, 2],
            "negative_values": [3],
        },
        "model_input_raw_columns": ["loan_type", "income"],
        "audit_only_columns": ["lei", "applicant_sex"],
    }


def _write_csv(path, rows, fieldnames=None):
    fieldnames = fieldnames or ["action_taken", "loan_type", "income", "lei", "applicant_sex"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_build_filters_cohort_maps_target_and_preserves_raw_tokens(tmp_path):
    raw_csv = tmp_path / "raw.csv"
    _write_csv(
        raw_csv,
        [
            {"action_taken": "1", "loan_type": "1", "income": "NA", "lei": "a", "applicant_sex": "1"},
            {"action_taken": "2", "loan_type": "2", "income": "", "lei": "b", "applicant_sex": "2"},
            {"action_taken": "3", "loan_type": "3", "income": "Exempt", "lei": "c", "applicant_sex": "3"},
            {"action_taken": "4", "loan_type": "4", "income": "100", "lei": "d", "applicant_sex": "1"},
        ],
    )

    view = DatasetBViewBuilder(_contract()).build(raw_csv)

    assert [record["record_id"] for record in view.records] == [
        "dataset_b-000000001",
        "dataset_b-000000002",
        "dataset_b-000000003",
    ]
    assert [record["y_approved"] for record in view.records] == [1, 1, 0]
    assert [record["action_taken"] for record in view.records] == ["1", "2", "3"]
    assert [record["income"] for record in view.records] == ["NA", "", "Exempt"]
    assert view.model_input_columns == ("loan_type", "income")
    assert view.audit_only_columns == ("lei", "applicant_sex")


def test_build_rejects_missing_required_source_column(tmp_path):
    raw_csv = tmp_path / "raw.csv"
    _write_csv(
        raw_csv,
        [{"action_taken": "1", "loan_type": "1", "income": "100", "lei": "a"}],
        fieldnames=["action_taken", "loan_type", "income", "lei"],
    )

    with pytest.raises(DatasetBViewError, match="applicant_sex"):
        DatasetBViewBuilder(_contract()).build(raw_csv)


def test_contract_rejects_audit_column_as_model_input():
    contract = _contract()
    contract["model_input_raw_columns"].append("applicant_sex")

    with pytest.raises(DatasetBViewError, match="overlap"):
        DatasetBViewBuilder(contract)
