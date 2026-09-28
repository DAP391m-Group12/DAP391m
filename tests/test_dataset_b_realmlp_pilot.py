"""Contract tests for immutable Dataset B RealMLP pilot artifacts."""

import csv
import json

import pytest
import sklearn
import yaml

from src.dataset_b.splits import (
    DatasetBRealMLPPilotSplitter,
    RealMLPPilotError,
    materialize_realmlp_pilot_artifacts,
    validate_realmlp_pilot_artifacts,
)


FEATURES = [
    "loan_type",
    "loan_purpose",
    "lien_status",
    "reverse_mortgage",
    "open-end_line_of_credit",
    "business_or_commercial_purpose",
    "income",
    "debt_to_income_ratio",
    "loan_amount",
    "loan_to_value_ratio",
    "loan_term",
    "occupancy_type",
]
AUDIT = [
    "lei",
    "county_code",
    "applicant_race-1",
    "applicant_race-2",
    "applicant_race-3",
    "applicant_race-4",
    "applicant_race-5",
    "applicant_sex",
    "applicant_age",
]
HEADER = ["record_id", "action_taken", "y_approved", *FEATURES, *AUDIT]


def _write_partition(path, record_ids, labels):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER, lineterminator="\n")
        writer.writeheader()
        for record_id, label in zip(record_ids, labels, strict=True):
            row = dict.fromkeys(HEADER, "")
            row.update(
                record_id=record_id,
                action_taken="1" if label else "3",
                y_approved=str(label),
                loan_type="1",
                loan_purpose="1",
                lien_status="1",
                reverse_mortgage="2",
                **{
                    "open-end_line_of_credit": "2",
                    "business_or_commercial_purpose": "2",
                },
                income="100",
                debt_to_income_ratio="42",
                loan_amount="200000",
                loan_to_value_ratio="75",
                loan_term="360",
                occupancy_type="1",
            )
            writer.writerow(row)


def _source(tmp_path):
    split_dir = tmp_path / "split_v1"
    split_dir.mkdir()
    train_ids = [f"train-{index:03d}" for index in range(100)]
    train_labels = [index % 2 for index in range(100)]
    validation_ids = [f"validation-{index:03d}" for index in range(10)]
    test_ids = [f"test-{index:03d}" for index in range(10)]
    _write_partition(split_dir / "train.csv", train_ids, train_labels)
    _write_partition(split_dir / "validation.csv", validation_ids, [0, 1] * 5)
    _write_partition(split_dir / "test.csv", test_ids, [0, 1] * 5)

    manifest_path = tmp_path / "split_v1.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "dataset_b",
                "split_version": "split_v1",
                "split_unit": "record_id",
                "grouping_constraint": "none",
                "random_state": 42,
                "train_record_ids": train_ids,
                "validation_record_ids": validation_ids,
                "test_record_ids": test_ids,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    feature_path = tmp_path / "feature_set_v1.yaml"
    feature_path.write_text(
        yaml.safe_dump(
            {
                "dataset": "dataset_b",
                "feature_version": "temporary_training_lock_v0_2",
                "target": {
                    "name": "y_approved",
                    "source_column": "action_taken",
                    "eligible_values": [1, 2, 3],
                    "positive_values": [1, 2],
                    "negative_values": [3],
                },
                "model_input_raw_columns": FEATURES,
                "audit_only_columns": AUDIT,
            }
        ),
        encoding="utf-8",
    )
    return manifest_path, split_dir, feature_path, train_ids, train_labels


def test_realmlp_pilot_split_is_deterministic_stratified_and_disjoint():
    record_ids = [f"record-{index:03d}" for index in range(100)]
    labels = [index % 2 for index in range(100)]
    splitter = DatasetBRealMLPPilotSplitter()

    first = splitter.build(record_ids, labels)
    second = splitter.build(record_ids, labels)

    assert first == second
    assert len(first.pilot_record_ids) == 20
    assert len(first.pilot_fit_record_ids) == 16
    assert len(first.pilot_inner_stop_record_ids) == 4
    assert not set(first.pilot_fit_record_ids) & set(first.pilot_inner_stop_record_ids)
    assert set(first.pilot_fit_record_ids) | set(first.pilot_inner_stop_record_ids) == set(
        first.pilot_record_ids
    )
    labels_by_id = dict(zip(record_ids, labels, strict=True))
    assert sum(labels_by_id[record_id] for record_id in first.pilot_record_ids) == 10
    assert sum(labels_by_id[record_id] for record_id in first.pilot_fit_record_ids) == 8
    assert sum(
        labels_by_id[record_id] for record_id in first.pilot_inner_stop_record_ids
    ) == 2
    with pytest.raises(RealMLPPilotError, match="binary"):
        splitter.build(record_ids, [0.5, *labels[1:]])


def test_realmlp_pilot_artifacts_are_valid_hashed_and_immutable(tmp_path):
    manifest, split_dir, feature_contract, _, _ = _source(tmp_path)
    output_dir = tmp_path / "realmlp_v1" / "pilot"

    published = materialize_realmlp_pilot_artifacts(
        output_dir,
        split_manifest_path=manifest,
        split_dir=split_dir,
        feature_contract_path=feature_contract,
    )
    payload = validate_realmlp_pilot_artifacts(
        published,
        split_manifest_path=manifest,
        split_dir=split_dir,
        feature_contract_path=feature_contract,
    )

    assert {path.name for path in published.iterdir()} == {
        "pilot_split_manifest.json",
        "pilot_fit_ids.csv",
        "pilot_inner_stop_ids.csv",
    }
    assert payload["pilot"]["rows"] == 20
    assert payload["outputs"]["pilot_fit"]["rows"] == 16
    assert payload["outputs"]["pilot_inner_stop"]["rows"] == 4
    assert payload["validation"] == {
        "zero_overlap": True,
        "all_ids_in_train": True,
        "no_validation_ids": True,
        "no_test_ids": True,
        "internal_union_matches_pilot": True,
    }
    with pytest.raises(RealMLPPilotError, match="immutable"):
        materialize_realmlp_pilot_artifacts(
            output_dir,
            split_manifest_path=manifest,
            split_dir=split_dir,
            feature_contract_path=feature_contract,
        )


def test_realmlp_pilot_validation_rejects_tampered_id_artifact(tmp_path):
    manifest, split_dir, feature_contract, _, _ = _source(tmp_path)
    output_dir = materialize_realmlp_pilot_artifacts(
        tmp_path / "realmlp_v1" / "pilot",
        split_manifest_path=manifest,
        split_dir=split_dir,
        feature_contract_path=feature_contract,
    )
    fit_path = output_dir / "pilot_fit_ids.csv"
    fit_path.write_text(fit_path.read_text(encoding="utf-8") + "train-099\n", encoding="utf-8")

    with pytest.raises(RealMLPPilotError, match="locked split|overlap"):
        validate_realmlp_pilot_artifacts(
            output_dir,
            split_manifest_path=manifest,
            split_dir=split_dir,
            feature_contract_path=feature_contract,
        )


def test_realmlp_pilot_preserves_sampling_version_as_provenance(tmp_path, monkeypatch):
    manifest, split_dir, feature_contract, _, _ = _source(tmp_path)
    original_version = sklearn.__version__
    monkeypatch.setattr(sklearn, "__version__", "historical-sklearn")
    output_dir = materialize_realmlp_pilot_artifacts(
        tmp_path / "realmlp_v1" / "pilot",
        split_manifest_path=manifest,
        split_dir=split_dir,
        feature_contract_path=feature_contract,
    )
    monkeypatch.setattr(sklearn, "__version__", original_version)

    payload = validate_realmlp_pilot_artifacts(
        output_dir,
        split_manifest_path=manifest,
        split_dir=split_dir,
        feature_contract_path=feature_contract,
    )

    assert payload["sampling"]["sklearn_version"] == "historical-sklearn"
