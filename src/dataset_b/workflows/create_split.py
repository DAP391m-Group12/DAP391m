"""Create the immutable Dataset B split manifest and materialized CSVs."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

from src.dataset_b.data import (
    DatasetBSplitter,
    DatasetBView,
    DatasetBViewBuilder,
    DatasetBViewError,
    SplitManifest,
    SplitManifestError,
)
from src.utils.path import DATASET_B_DIR, PROJECT_ROOT, RAW_STATE_NY_PATH


DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "dataset_b" / "split_v1.json"
DEFAULT_SPLIT_DATA_DIR = DATASET_B_DIR / "split_v1"
SPLIT_FILENAMES = {
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv",
}


class SplitDataExportError(ValueError):
    """Raised when a materialized Dataset B split cannot be written or verified."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the explicit inputs for one Dataset B split-manifest run."""
    parser = argparse.ArgumentParser(
        description="Create the fixed Dataset B 70/15/15 split manifest from the raw CSV."
    )
    parser.add_argument(
        "--raw-csv",
        type=Path,
        default=RAW_STATE_NY_PATH,
        help=f"Raw HMDA CSV to read (default: {RAW_STATE_NY_PATH}).",
    )
    parser.add_argument(
        "--feature-contract",
        type=Path,
        default=DatasetBViewBuilder.DEFAULT_CONTRACT_PATH,
        help="Versioned Dataset B feature contract.",
    )
    parser.add_argument(
        "--split-contract",
        type=Path,
        default=DatasetBSplitter.DEFAULT_CONTRACT_PATH,
        help="Versioned Dataset B split contract.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Manifest path to create (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--split-data-dir",
        type=Path,
        default=DEFAULT_SPLIT_DATA_DIR,
        help=(
            "Directory for the materialized raw Dataset B split CSVs "
            f"(default: {DEFAULT_SPLIT_DATA_DIR})."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow replacement of an existing manifest at --output.",
    )
    return parser.parse_args(argv)


def split_csv_paths(output_dir: Path) -> dict[str, Path]:
    """Return the three fixed materialized-split paths for one Dataset B version."""
    return {name: output_dir / filename for name, filename in SPLIT_FILENAMES.items()}


def write_split_csvs(
    view: DatasetBView, manifest: SplitManifest, output_dir: Path
) -> dict[str, Path]:
    """Materialize the manifest partitions without transforming controlled raw values."""
    records_by_id = _records_by_id(view)
    partition_ids = {
        "train": manifest.train_record_ids,
        "validation": manifest.validation_record_ids,
        "test": manifest.test_record_ids,
    }
    manifest_ids = set(manifest.all_record_ids)
    if manifest_ids != set(records_by_id):
        raise SplitDataExportError("Manifest record IDs do not match the Dataset B view.")

    paths = split_csv_paths(output_dir)
    for partition_name, record_ids in partition_ids.items():
        records = [records_by_id[record_id] for record_id in record_ids]
        _write_csv_atomically(paths[partition_name], view.columns, records)
        _verify_split_csv(
            paths[partition_name],
            view.columns,
            records_by_id,
            record_ids,
        )
    return paths


def _records_by_id(view: DatasetBView) -> dict[str, dict[str, str | int]]:
    records_by_id = {str(record["record_id"]): record for record in view.records}
    if len(records_by_id) != len(view.records):
        raise SplitDataExportError("Dataset B view has duplicate record IDs.")
    return records_by_id


def _write_csv_atomically(
    path: Path,
    fieldnames: Sequence[str],
    records: Sequence[Mapping[str, str | int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    except OSError as exc:
        raise SplitDataExportError(f"Cannot write Dataset B split CSV: {path}") from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _verify_split_csv(
    path: Path,
    fieldnames: Sequence[str],
    records_by_id: Mapping[str, Mapping[str, str | int]],
    expected_record_ids: Sequence[str],
) -> None:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(fieldnames):
                raise SplitDataExportError(f"Dataset B split CSV has an invalid schema: {path}")

            actual_record_ids: list[str] = []
            for row in reader:
                record_id = row.get("record_id")
                if record_id not in records_by_id:
                    raise SplitDataExportError(f"Dataset B split CSV has an unknown record ID: {path}")
                expected_record = records_by_id[record_id]
                if any(row[column] != str(expected_record[column]) for column in fieldnames):
                    raise SplitDataExportError(f"Dataset B split CSV altered a controlled raw value: {path}")
                actual_record_ids.append(record_id)
    except OSError as exc:
        raise SplitDataExportError(f"Cannot read Dataset B split CSV: {path}") from exc

    if tuple(actual_record_ids) != tuple(expected_record_ids):
        raise SplitDataExportError(f"Dataset B split CSV does not match its manifest partition: {path}")


def main(argv: Sequence[str] | None = None) -> int:
    """Build, persist, reload, and report one fixed Dataset B split manifest."""
    args = parse_args(argv)
    output_path = args.output
    data_paths = split_csv_paths(args.split_data_dir)
    existing_paths = [path for path in (output_path, *data_paths.values()) if path.exists()]

    if existing_paths and not args.force:
        print(
            "Refusing to overwrite existing Dataset B split output(s): "
            f"{', '.join(str(path) for path in existing_paths)}. "
            "Use --force only after reviewing it.",
            file=sys.stderr,
        )
        return 2

    try:
        view = DatasetBViewBuilder.from_yaml(args.feature_contract).build(args.raw_csv)
        splitter = DatasetBSplitter.from_yaml(args.split_contract)
        manifest = splitter.build(view)
        written_path = splitter.write_manifest(manifest, output_path)
        persisted_manifest = DatasetBSplitter.load_manifest(written_path)
        written_csv_paths = write_split_csvs(view, persisted_manifest, args.split_data_dir)
    except (DatasetBViewError, SplitManifestError, SplitDataExportError) as exc:
        print(f"Dataset B split failed: {exc}", file=sys.stderr)
        return 1

    if persisted_manifest != manifest:
        print("Dataset B split failed: persisted manifest does not match the generated manifest.", file=sys.stderr)
        return 1

    print(f"Manifest: {written_path}")
    print(f"Train records: {len(manifest.train_record_ids)}")
    print(f"Validation records: {len(manifest.validation_record_ids)}")
    print(f"Test records: {len(manifest.test_record_ids)}")
    print(f"Train CSV: {written_csv_paths['train']}")
    print(f"Validation CSV: {written_csv_paths['validation']}")
    print(f"Test CSV: {written_csv_paths['test']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
