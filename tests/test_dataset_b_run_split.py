import csv

import yaml

from src.dataset_b.data import DatasetBSplitter
from src.dataset_b.splits.run_split import main


def _write_raw_csv(path):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["action_taken", "loan_type", "lei"])
        writer.writeheader()
        for index in range(20):
            writer.writerow(
                {
                    "action_taken": "1" if index % 2 == 0 else "3",
                    "loan_type": "1",
                    "lei": f"lei-{index}",
                }
            )


def _write_contracts(tmp_path):
    feature_contract = tmp_path / "feature_set.yaml"
    split_contract = tmp_path / "split.yaml"
    feature_contract.write_text(
        yaml.safe_dump(
            {
                "dataset": "dataset_b",
                "target": {
                    "source_column": "action_taken",
                    "eligible_values": [1, 2, 3],
                    "positive_values": [1, 2],
                    "negative_values": [3],
                },
                "model_input_raw_columns": ["loan_type"],
                "audit_only_columns": ["lei"],
            }
        ),
        encoding="utf-8",
    )
    split_contract.write_text(
        yaml.safe_dump(
            {
                "dataset": "dataset_b",
                "split_version": "split_v1",
                "split_unit": "record_id",
                "grouping_constraint": "none",
                "random_state": 42,
                "holdout_split": {
                    "train_fraction": 0.70,
                    "validation_fraction": 0.15,
                    "test_fraction": 0.15,
                    "stratify_by": "y_approved",
                },
                "cross_validation": {
                    "splitter": "StratifiedKFold",
                    "n_splits": 5,
                    "shuffle": True,
                    "random_state": 42,
                    "allowed_population": "train_only",
                },
            }
        ),
        encoding="utf-8",
    )
    return feature_contract, split_contract


def test_cli_creates_and_reloads_manifest(tmp_path, capsys):
    raw_csv = tmp_path / "raw.csv"
    output_path = tmp_path / "artifacts" / "split_v1.json"
    split_data_dir = tmp_path / "data" / "dataset_b" / "split_v1"
    _write_raw_csv(raw_csv)
    feature_contract, split_contract = _write_contracts(tmp_path)

    result = main(
        [
            "--raw-csv",
            str(raw_csv),
            "--feature-contract",
            str(feature_contract),
            "--split-contract",
            str(split_contract),
            "--output",
            str(output_path),
            "--split-data-dir",
            str(split_data_dir),
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "Train records: 14" in output
    assert f"Train CSV: {split_data_dir / 'train.csv'}" in output
    manifest = DatasetBSplitter.load_manifest(output_path)
    assert len(manifest.train_record_ids) == 14
    assert len(manifest.validation_record_ids) == 3
    assert len(manifest.test_record_ids) == 3
    assert len(set(manifest.all_record_ids)) == 20

    expected_ids = {
        "train.csv": manifest.train_record_ids,
        "validation.csv": manifest.validation_record_ids,
        "test.csv": manifest.test_record_ids,
    }
    for filename, record_ids in expected_ids.items():
        with (split_data_dir / filename).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        assert reader.fieldnames == ["record_id", "action_taken", "y_approved", "loan_type", "lei"]
        assert [row["record_id"] for row in rows] == list(record_ids)
        assert all(row["action_taken"] in {"1", "3"} for row in rows)


def test_cli_refuses_to_overwrite_without_force(tmp_path, capsys):
    raw_csv = tmp_path / "raw.csv"
    output_path = tmp_path / "split_v1.json"
    _write_raw_csv(raw_csv)
    feature_contract, split_contract = _write_contracts(tmp_path)
    output_path.write_text("existing manifest", encoding="utf-8")

    result = main(
        [
            "--raw-csv",
            str(raw_csv),
            "--feature-contract",
            str(feature_contract),
            "--split-contract",
            str(split_contract),
            "--output",
            str(output_path),
        ]
    )

    assert result == 2
    assert output_path.read_text(encoding="utf-8") == "existing manifest"
    assert "Refusing to overwrite" in capsys.readouterr().err


def test_cli_refuses_to_overwrite_an_existing_split_csv_without_force(tmp_path, capsys):
    raw_csv = tmp_path / "raw.csv"
    output_path = tmp_path / "split_v1.json"
    split_data_dir = tmp_path / "data" / "dataset_b" / "split_v1"
    _write_raw_csv(raw_csv)
    feature_contract, split_contract = _write_contracts(tmp_path)
    split_data_dir.mkdir(parents=True)
    train_csv = split_data_dir / "train.csv"
    train_csv.write_text("existing split", encoding="utf-8")

    result = main(
        [
            "--raw-csv",
            str(raw_csv),
            "--feature-contract",
            str(feature_contract),
            "--split-contract",
            str(split_contract),
            "--output",
            str(output_path),
            "--split-data-dir",
            str(split_data_dir),
        ]
    )

    assert result == 2
    assert train_csv.read_text(encoding="utf-8") == "existing split"
    assert "Refusing to overwrite" in capsys.readouterr().err
