"""Boundary checks for the fixed Dataset B split loader."""

import csv
import json
import logging

import pytest
import yaml

from src.dataset_b.preprocessing import DatasetBSplitLoader, SplitInputError


FEATURES = [
    "loan_type", "loan_purpose", "lien_status", "reverse_mortgage",
    "open-end_line_of_credit", "business_or_commercial_purpose", "income",
    "debt_to_income_ratio", "loan_amount", "loan_to_value_ratio", "loan_term",
    "occupancy_type", "preapproval", "interest_only_payment", "balloon_payment",
]
AUDIT = [
    "lei", "county_code", "applicant_race-1", "applicant_race-2",
    "applicant_race-3", "applicant_race-4", "applicant_race-5",
    "applicant_sex", "applicant_age",
]
HEADER = ["record_id", "action_taken", "y_approved", *FEATURES, *AUDIT]


def _row(record_id, action):
    row = dict.fromkeys(HEADER, "")
    row.update(
        record_id=record_id,
        action_taken=action,
        y_approved="0" if action == "3" else "1",
        loan_type="1",
        loan_purpose="1",
        lien_status="1",
        reverse_mortgage="2",
        **{"open-end_line_of_credit": "2", "business_or_commercial_purpose": "2"},
        income="NA",
        debt_to_income_ratio="42",
        loan_amount="215000.0",
        loan_to_value_ratio="Exempt",
        loan_term="360",
        occupancy_type="1",
        preapproval="2",
        interest_only_payment="2",
        balloon_payment="2",
        lei="example-lei",
    )
    return row


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def _inputs(tmp_path):
    split_dir = tmp_path / "split_v1"
    split_dir.mkdir()
    rows = {
        "train": [_row("r1", "1"), _row("r2", "3")],
        "validation": [_row("r3", "2")],
        "test": [_row("r4", "3")],
    }
    for partition, records in rows.items():
        _write_csv(split_dir / f"{partition}.csv", records)
    manifest_path = tmp_path / "split_v1.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": "dataset_b", "split_version": "split_v1",
                "split_unit": "record_id", "grouping_constraint": "none",
                "random_state": 42,
                "train_record_ids": ["r1", "r2"],
                "validation_record_ids": ["r3"],
                "test_record_ids": ["r4"],
            }
        ),
        encoding="utf-8",
    )
    feature_path = tmp_path / "feature_set_v1.yaml"
    feature_path.write_text(
        yaml.safe_dump(
            {
                "dataset": "dataset_b", "feature_version": "feature_set_v1",
                "target": {
                    "name": "y_approved", "source_column": "action_taken",
                    "eligible_values": [1, 2, 3], "positive_values": [1, 2],
                    "negative_values": [3],
                },
                "model_input_raw_columns": FEATURES,
                "audit_only_columns": AUDIT,
            }
        ),
        encoding="utf-8",
    )
    return manifest_path, split_dir, feature_path, rows


def test_load_train_preserves_tokens_and_keeps_other_partitions_unread(tmp_path, caplog):
    manifest, split_dir, feature_contract, _ = _inputs(tmp_path)
    (split_dir / "test.csv").unlink()
    loader = DatasetBSplitLoader(manifest, split_dir, feature_contract)

    with caplog.at_level(logging.INFO, logger="src.dataset_b.preprocessing.split_inputs"):
        train = loader.load("train")

    assert train.name == "train"
    assert list(train.record_ids) == ["r1", "r2"]
    assert list(train.y) == [1, 0]
    assert list(train.X.columns) == FEATURES
    assert list(train.audit.columns) == AUDIT
    assert train.X.loc[0, "income"] == "NA"
    assert train.X.loc[0, "loan_to_value_ratio"] == "Exempt"
    assert "action_taken" not in train.X and "record_id" not in train.X
    assert "lei" not in train.X
    assert list(train.action_taken) == ["1", "3"]
    assert "partition=train rows=2" in caplog.text


def test_loader_rejects_manifest_order_or_target_corruption(tmp_path):
    manifest, split_dir, feature_contract, rows = _inputs(tmp_path)
    loader = DatasetBSplitLoader(manifest, split_dir, feature_contract)
    _write_csv(split_dir / "train.csv", list(reversed(rows["train"])))
    with pytest.raises(SplitInputError, match="record order"):
        loader.load("train")

    rows["train"][0]["y_approved"] = "0"
    _write_csv(split_dir / "train.csv", rows["train"])
    with pytest.raises(SplitInputError, match="target mapping"):
        loader.load("train")


def test_loader_preserves_textual_nan_for_downstream_quality_gate(tmp_path):
    manifest, split_dir, feature_contract, rows = _inputs(tmp_path)
    rows["validation"][0]["income"] = "NaN"
    _write_csv(split_dir / "validation.csv", rows["validation"])

    validation = DatasetBSplitLoader(manifest, split_dir, feature_contract).load("validation")

    assert validation.X.loc[0, "income"] == "NaN"


def test_loader_rejects_unexpected_schema_and_partition(tmp_path):
    manifest, split_dir, feature_contract, _ = _inputs(tmp_path)
    loader = DatasetBSplitLoader(manifest, split_dir, feature_contract)
    with pytest.raises(SplitInputError, match="Partition"):
        loader.load("development")

    with (split_dir / "train.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER[:-1])
    with pytest.raises(SplitInputError, match="schema"):
        loader.load("train")
