"""Create and validate the immutable Train-only Dataset B RealMLP pilot split."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit

from src.utils.path import CONFIG_DIR, DATASET_B_DIR, PROJECT_ROOT

from .split_manifest import DatasetBSplitter, SplitManifestError


PILOT_FRACTION = 0.20
PILOT_FIT_FRACTION = 0.80
PILOT_INNER_STOP_FRACTION = 0.20
PILOT_SEED = 42
INNER_SPLIT_SEED = 42
SAMPLING_IMPLEMENTATION_VERSION = "dataset_b_realmlp_pilot_v1"
ID_HASH_FORMAT = "sha256_utf8_lf_record_ids_v1"


class RealMLPPilotError(ValueError):
    """Raised when the RealMLP pilot split or its artifacts violate the contract."""


@dataclass(frozen=True)
class RealMLPPilotSplit:
    """Immutable Train-only membership for the RealMLP pilot."""

    pilot_record_ids: tuple[str, ...]
    pilot_fit_record_ids: tuple[str, ...]
    pilot_inner_stop_record_ids: tuple[str, ...]


class DatasetBRealMLPPilotSplitter:
    """Build the locked 20% pilot and its internal 80/20 split."""

    def build(
        self,
        record_ids: Sequence[str],
        labels: Sequence[int],
    ) -> RealMLPPilotSplit:
        ids = tuple(record_ids)
        raw_target = tuple(labels)
        if any(value not in (0, 1) for value in raw_target):
            raise RealMLPPilotError("Source Train labels must contain only binary values.")
        target = tuple(int(value) for value in raw_target)
        self._validate_source(ids, target)

        positions = np.arange(len(ids))
        try:
            _, pilot_positions = next(
                StratifiedShuffleSplit(
                    n_splits=1,
                    test_size=PILOT_FRACTION,
                    random_state=PILOT_SEED,
                ).split(positions, target)
            )
            pilot_positions = np.sort(pilot_positions)
            pilot_labels = np.asarray(target)[pilot_positions]
            fit_relative, inner_stop_relative = next(
                StratifiedShuffleSplit(
                    n_splits=1,
                    test_size=PILOT_INNER_STOP_FRACTION,
                    random_state=INNER_SPLIT_SEED,
                ).split(pilot_positions, pilot_labels)
            )
        except ValueError as exc:
            raise RealMLPPilotError(
                "Dataset B Train cannot satisfy the locked stratified RealMLP pilot split."
            ) from exc

        fit_positions = np.sort(pilot_positions[fit_relative])
        inner_stop_positions = np.sort(pilot_positions[inner_stop_relative])
        result = RealMLPPilotSplit(
            pilot_record_ids=tuple(ids[position] for position in pilot_positions),
            pilot_fit_record_ids=tuple(ids[position] for position in fit_positions),
            pilot_inner_stop_record_ids=tuple(
                ids[position] for position in inner_stop_positions
            ),
        )
        self.validate(result, ids)
        return result

    @staticmethod
    def validate(split: RealMLPPilotSplit, train_record_ids: Sequence[str]) -> None:
        """Validate uniqueness, Train membership, and exact internal coverage."""
        train_ids = tuple(train_record_ids)
        train_set = set(train_ids)
        pilot_set = set(split.pilot_record_ids)
        fit_set = set(split.pilot_fit_record_ids)
        stop_set = set(split.pilot_inner_stop_record_ids)

        if len(train_set) != len(train_ids):
            raise RealMLPPilotError("Source Train record IDs must be unique.")
        if any(
            len(values) != len(set(values))
            for values in (
                split.pilot_record_ids,
                split.pilot_fit_record_ids,
                split.pilot_inner_stop_record_ids,
            )
        ):
            raise RealMLPPilotError("RealMLP pilot record IDs must be unique.")
        if not pilot_set.issubset(train_set):
            raise RealMLPPilotError("Every RealMLP pilot record ID must belong to Train.")
        if fit_set & stop_set:
            raise RealMLPPilotError("Pilot-fit and pilot-inner-stop IDs must not overlap.")
        if fit_set | stop_set != pilot_set:
            raise RealMLPPilotError(
                "Pilot-fit and pilot-inner-stop IDs must cover the complete pilot."
            )

        train_order = {record_id: position for position, record_id in enumerate(train_ids)}
        for name, values in (
            ("pilot", split.pilot_record_ids),
            ("pilot-fit", split.pilot_fit_record_ids),
            ("pilot-inner-stop", split.pilot_inner_stop_record_ids),
        ):
            positions = [train_order[record_id] for record_id in values]
            if positions != sorted(positions):
                raise RealMLPPilotError(
                    f"RealMLP {name} IDs must preserve source Train order."
                )

    @staticmethod
    def _validate_source(record_ids: tuple[str, ...], labels: tuple[int, ...]) -> None:
        if not record_ids or len(record_ids) != len(labels):
            raise RealMLPPilotError(
                "Source Train record IDs and labels must be non-empty and aligned."
            )
        if any(
            not isinstance(record_id, str)
            or not record_id
            or "\n" in record_id
            or "\r" in record_id
            for record_id in record_ids
        ):
            raise RealMLPPilotError(
                "Source Train record IDs must be non-empty strings without line breaks."
            )
        if len(set(record_ids)) != len(record_ids):
            raise RealMLPPilotError("Source Train record IDs must be unique.")
        if set(labels) != {0, 1}:
            raise RealMLPPilotError("Source Train labels must contain both binary classes.")


DEFAULT_SPLIT_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "dataset_b" / "split_v1.json"
DEFAULT_SPLIT_DIR = DATASET_B_DIR / "split_v1"
DEFAULT_FEATURE_CONTRACT_PATH = CONFIG_DIR / "dataset_b" / "feature_set_v1.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "dataset_b" / "realmlp_v1" / "pilot"
_FIT_FILENAME = "pilot_fit_ids.csv"
_STOP_FILENAME = "pilot_inner_stop_ids.csv"
_MANIFEST_FILENAME = "pilot_split_manifest.json"
_EXPECTED_FILENAMES = {_FIT_FILENAME, _STOP_FILENAME, _MANIFEST_FILENAME}


@dataclass(frozen=True)
class _PilotSource:
    train_record_ids: tuple[str, ...]
    labels: tuple[int, ...]
    validation_record_ids: frozenset[str]
    test_record_ids: frozenset[str]
    split_version: str
    split_manifest_sha256: str


def materialize_realmlp_pilot_artifacts(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    split_manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST_PATH,
    split_dir: str | Path = DEFAULT_SPLIT_DIR,
    feature_contract_path: str | Path = DEFAULT_FEATURE_CONTRACT_PATH,
) -> Path:
    """Build, validate, and atomically publish the immutable pilot artifacts."""
    destination = Path(output_dir)
    if destination.exists():
        raise RealMLPPilotError(
            f"RealMLP pilot artifacts are immutable and already exist: {destination}"
        )

    source = _load_source(split_manifest_path, split_dir, feature_contract_path)
    split = DatasetBRealMLPPilotSplitter().build(source.train_record_ids, source.labels)
    _validate_boundaries(split, source)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = _make_staging_directory(destination)
    try:
        _write_id_csv(staging / _FIT_FILENAME, split.pilot_fit_record_ids)
        _write_id_csv(staging / _STOP_FILENAME, split.pilot_inner_stop_record_ids)
        payload = _manifest_payload(staging, split, source)
        with (staging / _MANIFEST_FILENAME).open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        _validate_artifact_set(staging, source)
        os.replace(staging, destination)
    except (OSError, RealMLPPilotError) as exc:
        if isinstance(exc, RealMLPPilotError):
            raise
        raise RealMLPPilotError(
            f"Cannot publish Dataset B RealMLP pilot artifacts: {destination}"
        ) from exc
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    validate_realmlp_pilot_artifacts(
        destination,
        split_manifest_path=split_manifest_path,
        split_dir=split_dir,
        feature_contract_path=feature_contract_path,
    )
    return destination


def validate_realmlp_pilot_artifacts(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    split_manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST_PATH,
    split_dir: str | Path = DEFAULT_SPLIT_DIR,
    feature_contract_path: str | Path = DEFAULT_FEATURE_CONTRACT_PATH,
) -> dict:
    """Validate a published pilot set against the current immutable Train source."""
    source = _load_source(split_manifest_path, split_dir, feature_contract_path)
    return _validate_artifact_set(Path(output_dir), source)


def _load_source(
    split_manifest_path: str | Path,
    split_dir: str | Path,
    feature_contract_path: str | Path,
) -> _PilotSource:
    # Lazy import avoids a package cycle: split_inputs imports src.dataset_b.splits.
    from src.dataset_b.preprocessing.split_inputs import (
        DatasetBSplitLoader,
        SplitInputError,
    )

    manifest_path = Path(split_manifest_path)
    try:
        manifest = DatasetBSplitter.load_manifest(manifest_path)
        train = DatasetBSplitLoader(
            manifest_path=manifest_path,
            split_dir=split_dir,
            feature_contract_path=feature_contract_path,
        ).load("train")
        manifest_hash = _sha256_file(manifest_path)
    except (OSError, SplitManifestError, SplitInputError) as exc:
        raise RealMLPPilotError("Cannot load the immutable Dataset B Train source.") from exc

    return _PilotSource(
        train_record_ids=tuple(train.record_ids),
        labels=tuple(int(value) for value in train.y),
        validation_record_ids=frozenset(manifest.validation_record_ids),
        test_record_ids=frozenset(manifest.test_record_ids),
        split_version=manifest.split_version,
        split_manifest_sha256=manifest_hash,
    )


def _validate_boundaries(split: RealMLPPilotSplit, source: _PilotSource) -> None:
    DatasetBRealMLPPilotSplitter.validate(split, source.train_record_ids)
    pilot_set = set(split.pilot_record_ids)
    if pilot_set & source.validation_record_ids:
        raise RealMLPPilotError("RealMLP pilot must not contain Validation IDs.")
    if pilot_set & source.test_record_ids:
        raise RealMLPPilotError("RealMLP pilot must not contain Test IDs.")


def _manifest_payload(
    directory: Path,
    split: RealMLPPilotSplit,
    source: _PilotSource,
    *,
    sampling_sklearn_version: str | None = None,
) -> dict:
    labels_by_id = dict(zip(source.train_record_ids, source.labels, strict=True))
    return {
        "schema_version": 1,
        "dataset": "dataset_b",
        "model": "realmlp",
        "artifact_version": "realmlp_v1",
        "contract_id": "PILOT-SPLIT-01",
        "source": {
            "partition": "train",
            "split_version": source.split_version,
            "split_manifest_sha256": source.split_manifest_sha256,
            "train_id_hash_format": ID_HASH_FORMAT,
            "train_id_sha256": _sha256_ids(source.train_record_ids),
            "train_rows": len(source.train_record_ids),
            "target_distribution": _target_distribution(
                source.train_record_ids, labels_by_id
            ),
        },
        "sampling": {
            "algorithm": "sklearn.model_selection.StratifiedShuffleSplit",
            "implementation_version": SAMPLING_IMPLEMENTATION_VERSION,
            "sklearn_version": sampling_sklearn_version or sklearn.__version__,
            "stratify_by": "y_approved",
            "output_order": "source_train_order",
            "pilot_fraction": PILOT_FRACTION,
            "pilot_seed": PILOT_SEED,
            "pilot_fit_fraction": PILOT_FIT_FRACTION,
            "pilot_inner_stop_fraction": PILOT_INNER_STOP_FRACTION,
            "inner_split_seed": INNER_SPLIT_SEED,
        },
        "pilot": {
            "rows": len(split.pilot_record_ids),
            "target_distribution": _target_distribution(
                split.pilot_record_ids, labels_by_id
            ),
        },
        "outputs": {
            "pilot_fit": {
                "path": _FIT_FILENAME,
                "rows": len(split.pilot_fit_record_ids),
                "sha256": _sha256_file(directory / _FIT_FILENAME),
                "target_distribution": _target_distribution(
                    split.pilot_fit_record_ids, labels_by_id
                ),
            },
            "pilot_inner_stop": {
                "path": _STOP_FILENAME,
                "rows": len(split.pilot_inner_stop_record_ids),
                "sha256": _sha256_file(directory / _STOP_FILENAME),
                "target_distribution": _target_distribution(
                    split.pilot_inner_stop_record_ids, labels_by_id
                ),
            },
        },
        "validation": {
            "zero_overlap": True,
            "all_ids_in_train": True,
            "no_validation_ids": True,
            "no_test_ids": True,
            "internal_union_matches_pilot": True,
        },
    }


def _validate_artifact_set(directory: Path, source: _PilotSource) -> dict:
    if not directory.is_dir():
        raise RealMLPPilotError(f"RealMLP pilot artifact directory is missing: {directory}")
    actual_filenames = {path.name for path in directory.iterdir()}
    if actual_filenames != _EXPECTED_FILENAMES:
        raise RealMLPPilotError(
            "RealMLP pilot artifact directory must contain exactly the three locked files."
        )

    try:
        with (directory / _MANIFEST_FILENAME).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RealMLPPilotError("Cannot read the RealMLP pilot manifest.") from exc
    if not isinstance(payload, dict):
        raise RealMLPPilotError("RealMLP pilot manifest must be a JSON object.")

    fit_ids = _read_id_csv(directory / _FIT_FILENAME)
    stop_ids = _read_id_csv(directory / _STOP_FILENAME)
    selected_ids = set(fit_ids) | set(stop_ids)
    expected_split = DatasetBRealMLPPilotSplitter().build(
        source.train_record_ids, source.labels
    )
    observed_split = RealMLPPilotSplit(
        pilot_record_ids=tuple(
            record_id
            for record_id in source.train_record_ids
            if record_id in selected_ids
        ),
        pilot_fit_record_ids=fit_ids,
        pilot_inner_stop_record_ids=stop_ids,
    )
    _validate_boundaries(observed_split, source)
    if observed_split != expected_split:
        raise RealMLPPilotError(
            "Published RealMLP pilot IDs differ from the deterministic locked split."
        )

    try:
        stored_fit_hash = payload["outputs"]["pilot_fit"]["sha256"]
        stored_stop_hash = payload["outputs"]["pilot_inner_stop"]["sha256"]
    except (KeyError, TypeError) as exc:
        raise RealMLPPilotError("RealMLP pilot manifest is missing output hashes.") from exc
    if stored_fit_hash != _sha256_file(directory / _FIT_FILENAME):
        raise RealMLPPilotError("RealMLP pilot-fit artifact hash is invalid.")
    if stored_stop_hash != _sha256_file(directory / _STOP_FILENAME):
        raise RealMLPPilotError("RealMLP pilot-inner-stop artifact hash is invalid.")

    try:
        sampling_sklearn_version = payload["sampling"]["sklearn_version"]
    except (KeyError, TypeError) as exc:
        raise RealMLPPilotError(
            "RealMLP pilot manifest is missing sklearn provenance."
        ) from exc
    if not isinstance(sampling_sklearn_version, str) or not sampling_sklearn_version:
        raise RealMLPPilotError(
            "RealMLP pilot manifest has invalid sklearn provenance."
        )
    expected_payload = _manifest_payload(
        directory,
        expected_split,
        source,
        sampling_sklearn_version=sampling_sklearn_version,
    )
    if payload != expected_payload:
        raise RealMLPPilotError(
            "RealMLP pilot manifest differs from the current source and locked contract."
        )
    return payload


def _write_id_csv(path: Path, record_ids: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["record_id"])
        writer.writerows((record_id,) for record_id in record_ids)


def _read_id_csv(path: Path) -> tuple[str, ...]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
    except OSError as exc:
        raise RealMLPPilotError(f"Cannot read RealMLP pilot ID artifact: {path}") from exc
    if not rows or rows[0] != ["record_id"]:
        raise RealMLPPilotError("RealMLP pilot ID artifacts require only a record_id header.")
    if any(len(row) != 1 or not row[0] for row in rows[1:]):
        raise RealMLPPilotError("RealMLP pilot ID artifacts contain an invalid row.")
    values = tuple(row[0] for row in rows[1:])
    if len(values) != len(set(values)):
        raise RealMLPPilotError("RealMLP pilot ID artifacts contain duplicate IDs.")
    return values


def _target_distribution(
    record_ids: Sequence[str], labels_by_id: dict[str, int]
) -> dict[str, int]:
    result = {"0": 0, "1": 0}
    for record_id in record_ids:
        result[str(labels_by_id[record_id])] += 1
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_ids(record_ids: Sequence[str]) -> str:
    payload = "".join(f"{record_id}\n" for record_id in record_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _make_staging_directory(destination: Path) -> Path:
    """Create an atomic staging directory that inherits parent access on Windows."""
    for _ in range(10):
        candidate = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
        try:
            candidate.mkdir(mode=0o777)
        except FileExistsError:
            continue
        return candidate
    raise RealMLPPilotError(
        f"Cannot allocate RealMLP pilot staging directory: {destination.parent}"
    )
