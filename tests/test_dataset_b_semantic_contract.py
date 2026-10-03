"""Tests for shared Dataset B semantic YAML and split-quality validation."""

from __future__ import annotations

import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.dataset_b.contracts import (
    DatasetBSemanticValidationError,
    validate_feature_contract,
    validate_split_data_quality,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_CONTRACT = PROJECT_ROOT / "config/dataset_b/feature_set_v1.yaml"


def _contract() -> dict:
    return yaml.safe_load(PRODUCTION_CONTRACT.read_text(encoding="utf-8"))


def _write_contract(path: Path, value: dict, *, crlf: bool = False) -> None:
    text = "# presentation-only comment\n" + yaml.safe_dump(
        value,
        sort_keys=True,
        allow_unicode=True,
    )
    if crlf:
        text = text.replace("\n", "\r\n")
    path.write_bytes(text.encode("utf-8"))


def test_production_contract_is_final_15_feature_lock():
    observed = validate_feature_contract(PRODUCTION_CONTRACT)
    contract = observed.semantic_identity

    assert contract["feature_version"] == "feature_set_v1"
    assert contract["status"] == "final_feature_lock"
    assert contract["purpose"] == "final_model_retraining"
    assert contract["source_of_truth"]["section"] == 11
    assert len(contract["model_input_raw_columns"]) == 15
    assert contract["model_input_raw_columns"][-3:] == [
        "preapproval",
        "interest_only_payment",
        "balloon_payment",
    ]
    assert contract["constraints"] == {
        "audit_columns_are_model_inputs": False,
        "final_model_features": True,
        "raw_feature_count": 15,
        "semantic_feature_count": 19,
        "preprocessing_profile": "dataset_b_preprocessing_v2",
        "encoded_feature_count": 62,
        "numeric_status_reference_category": "OBSERVED",
        "missing_category_retained": False,
    }


@pytest.mark.parametrize("crlf", [False, True], ids=["lf", "crlf"])
def test_semantic_identity_ignores_yaml_presentation(tmp_path, crlf):
    baseline = validate_feature_contract(PRODUCTION_CONTRACT)
    rendered = tmp_path / "feature_set_v1.yaml"
    _write_contract(rendered, _contract(), crlf=crlf)

    observed = validate_feature_contract(rendered)

    assert observed.semantic_identity == baseline.semantic_identity
    assert observed.semantic_sha256 == baseline.semantic_sha256
    assert observed.raw_sha256 != baseline.raw_sha256


@pytest.mark.parametrize(
    ("mutation", "expected_field"),
    [
        (lambda value: value.update(schema_version=True), "schema_version"),
        (lambda value: value.update(dataset="dataset_a"), "dataset"),
        (lambda value: value.update(feature_version="other"), "feature_version"),
        (lambda value: value.update(status="final"), "status"),
        (lambda value: value["target"].update(positive_values=[1]), "target"),
        (
            lambda value: value.update(
                model_input_raw_columns=list(reversed(value["model_input_raw_columns"]))
            ),
            "model_input_raw_columns",
        ),
        (
            lambda value: value.update(
                model_input_raw_columns=value["model_input_raw_columns"][:-1]
            ),
            "model_input_raw_columns",
        ),
        (
            lambda value: value.update(
                model_input_raw_columns=[*value["model_input_raw_columns"], "extra_feature"]
            ),
            "model_input_raw_columns",
        ),
        (
            lambda value: value.update(
                model_input_raw_columns=[
                    *value["model_input_raw_columns"][:-1],
                    value["model_input_raw_columns"][0],
                ]
            ),
            "model_input_raw_columns",
        ),
        (
            lambda value: value.update(
                audit_only_columns=[*value["audit_only_columns"][:-1], "loan_type"]
            ),
            "audit_only_columns",
        ),
        (
            lambda value: value["protected_attributes"]["sex"].update(
                source_column="loan_type"
            ),
            "protected_attributes",
        ),
        (
            lambda value: value["constraints"].update(encoded_feature_count=65),
            "constraints",
        ),
        (
            lambda value: value["constraints"].update(raw_feature_count=14),
            "constraints",
        ),
        (
            lambda value: value["constraints"].update(semantic_feature_count=18),
            "constraints",
        ),
        (
            lambda value: value["constraints"].update(missing_category_retained=True),
            "constraints",
        ),
    ],
)
def test_semantic_contract_rejects_changed_meaning(
    tmp_path,
    mutation,
    expected_field,
):
    value = deepcopy(_contract())
    mutation(value)
    path = tmp_path / "feature_set_v1.yaml"
    _write_contract(path, value)

    with pytest.raises(DatasetBSemanticValidationError, match=expected_field):
        validate_feature_contract(path)


def test_semantic_contract_rejects_non_mapping_and_duplicate_keys(tmp_path):
    non_mapping = tmp_path / "list.yaml"
    non_mapping.write_text("- dataset_b\n", encoding="utf-8")
    with pytest.raises(DatasetBSemanticValidationError, match="root must be a mapping"):
        validate_feature_contract(non_mapping)

    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text("dataset: dataset_b\ndataset: dataset_b\n", encoding="utf-8")
    with pytest.raises(DatasetBSemanticValidationError, match="valid Dataset B"):
        validate_feature_contract(duplicate)


def test_semantic_contract_rejects_unknown_root_field(tmp_path):
    value = _contract()
    value["unreviewed_policy"] = True
    path = tmp_path / "feature_set_v1.yaml"
    _write_contract(path, value)

    with pytest.raises(DatasetBSemanticValidationError, match="root fields"):
        validate_feature_contract(path)


def test_split_quality_checks_schema_order_targets_and_locked_manifest(tmp_path):
    feature = tmp_path / "feature_set_v1.yaml"
    _write_contract(feature, _contract())
    split_dir = tmp_path / "split_v1"
    split_dir.mkdir()
    model_columns = _contract()["model_input_raw_columns"]
    audit_columns = _contract()["audit_only_columns"]
    header = ["record_id", "action_taken", "y_approved", *model_columns, *audit_columns]
    rows = {
        "train": [("r1", "1", "1"), ("r2", "3", "0")],
        "validation": [("r3", "2", "1")],
        "test": [("r4", "3", "0")],
    }
    for name, records in rows.items():
        with (split_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writeheader()
            for record_id, action, target in records:
                row = dict.fromkeys(header, "")
                row.update(record_id=record_id, action_taken=action, y_approved=target)
                writer.writerow(row)

    manifest = tmp_path / "split_v1.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "dataset_b",
                "split_version": "split_v1",
                "split_unit": "record_id",
                "grouping_constraint": "none",
                "random_state": 42,
                "train_record_ids": ["r1", "r2"],
                "validation_record_ids": ["r3"],
                "test_record_ids": ["r4"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()

    result = validate_split_data_quality(
        feature,
        manifest,
        split_dir,
        expected_partition_rows={"train": 2, "validation": 1, "test": 1},
        expected_manifest_sha256=manifest_hash,
    )

    assert result.partitions["train"].rows == 2
    assert result.partitions["validation"].unique_record_ids == 1
    assert result.partitions["test"].model_input_columns == 15
    assert result.partitions["test"].audit_columns == 9

    manifest.write_text(manifest.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(DatasetBSemanticValidationError, match="locked split_v1"):
        validate_split_data_quality(
            feature,
            manifest,
            split_dir,
            expected_partition_rows={"train": 2, "validation": 1, "test": 1},
            expected_manifest_sha256=manifest_hash,
        )


def test_split_quality_rejects_target_corruption(tmp_path):
    feature = tmp_path / "feature_set_v1.yaml"
    _write_contract(feature, _contract())
    split_dir = tmp_path / "split_v1"
    split_dir.mkdir()
    value = _contract()
    header = [
        "record_id",
        "action_taken",
        "y_approved",
        *value["model_input_raw_columns"],
        *value["audit_only_columns"],
    ]
    ids = {"train": ["r1"], "validation": ["r2"], "test": ["r3"]}
    for name, record_ids in ids.items():
        with (split_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writeheader()
            row = dict.fromkeys(header, "")
            row.update(record_id=record_ids[0], action_taken="1", y_approved="0")
            writer.writerow(row)
    manifest = tmp_path / "split_v1.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "dataset_b",
                "split_version": "split_v1",
                "split_unit": "record_id",
                "grouping_constraint": "none",
                "random_state": 42,
                "train_record_ids": ids["train"],
                "validation_record_ids": ids["validation"],
                "test_record_ids": ids["test"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(DatasetBSemanticValidationError, match="data quality"):
        validate_split_data_quality(
            feature,
            manifest,
            split_dir,
            expected_partition_rows={"train": 1, "validation": 1, "test": 1},
            expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        )
