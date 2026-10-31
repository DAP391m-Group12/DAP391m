"""Create the deterministic, record-level Dataset B split manifest."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from .dataset_view import DatasetBView
from src.utils.path import CONFIG_DIR


class SplitManifestError(ValueError):
    """Raised when a Dataset B split contract or view violates split rules."""


@dataclass(frozen=True)
class SplitManifest:
    """Record identifiers assigned to one immutable Dataset B split version."""

    train_record_ids: tuple[str, ...]
    validation_record_ids: tuple[str, ...]
    test_record_ids: tuple[str, ...]
    random_state: int
    split_version: str

    @property
    def all_record_ids(self) -> tuple[str, ...]:
        return (*self.train_record_ids, *self.validation_record_ids, *self.test_record_ids)


class DatasetBSplitter:
    """Create 70/15/15 stratified holdouts and train-only cross-validation folds."""

    DEFAULT_CONTRACT_PATH = CONFIG_DIR / "dataset_b" / "split_v1.yaml"

    def __init__(self, contract: Mapping[str, Any]) -> None:
        self._contract = dict(contract)
        self._validate_contract()

        holdout = self._contract["holdout_split"]
        cv = self._contract["cross_validation"]
        self._train_fraction = float(holdout["train_fraction"])
        self._validation_fraction = float(holdout["validation_fraction"])
        self._test_fraction = float(holdout["test_fraction"])
        self._random_state = int(self._contract["random_state"])
        self._split_version = str(self._contract["split_version"])
        self._cv_splits = int(cv["n_splits"])
        self._cv_shuffle = bool(cv["shuffle"])
        self._cv_random_state = int(cv["random_state"])

    @classmethod
    def from_yaml(cls, contract_path: str | Path | None = None) -> "DatasetBSplitter":
        """Load the versioned Dataset B split contract from YAML."""
        path = Path(contract_path) if contract_path is not None else cls.DEFAULT_CONTRACT_PATH
        try:
            with path.open("r", encoding="utf-8") as handle:
                contract = yaml.safe_load(handle)
        except OSError as exc:
            raise SplitManifestError(f"Cannot read Dataset B split contract: {path}") from exc

        if not isinstance(contract, Mapping):
            raise SplitManifestError("Dataset B split contract must be a YAML mapping.")
        return cls(contract)

    def build(self, view: DatasetBView) -> SplitManifest:
        """Split a Dataset B view by record ID without using audit attributes as groups."""
        record_ids, labels = self._record_ids_and_labels(view)
        positions = list(range(len(record_ids)))

        first_split = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self._validation_fraction + self._test_fraction,
            random_state=self._random_state,
        )
        train_positions, holdout_positions = next(first_split.split(positions, labels))

        holdout_labels = [labels[position] for position in holdout_positions]
        second_split = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self._test_fraction / (self._validation_fraction + self._test_fraction),
            random_state=self._random_state,
        )
        validation_relative, test_relative = next(
            second_split.split(holdout_positions, holdout_labels)
        )

        manifest = SplitManifest(
            train_record_ids=tuple(record_ids[position] for position in train_positions),
            validation_record_ids=tuple(
                record_ids[holdout_positions[position]] for position in validation_relative
            ),
            test_record_ids=tuple(record_ids[holdout_positions[position]] for position in test_relative),
            random_state=self._random_state,
            split_version=self._split_version,
        )
        self._validate_manifest(manifest, record_ids)
        return manifest

    def cv_splitter(self) -> StratifiedKFold:
        """Return the mandated splitter for use only with manifest train records."""
        return StratifiedKFold(
            n_splits=self._cv_splits,
            shuffle=self._cv_shuffle,
            random_state=self._cv_random_state,
        )

    def write_manifest(self, manifest: SplitManifest, output_path: str | Path) -> Path:
        """Persist a split manifest atomically without writing processed data."""
        path = Path(output_path)
        if manifest.random_state != self._random_state or manifest.split_version != self._split_version:
            raise SplitManifestError("Manifest metadata must match the active split contract.")
        self._validate_manifest(manifest, manifest.all_record_ids)
        payload = {
            "schema_version": 1,
            "dataset": "dataset_b",
            "split_version": manifest.split_version,
            "split_unit": "record_id",
            "grouping_constraint": "none",
            "random_state": manifest.random_state,
            "train_record_ids": list(manifest.train_record_ids),
            "validation_record_ids": list(manifest.validation_record_ids),
            "test_record_ids": list(manifest.test_record_ids),
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                temporary_path = Path(handle.name)
            os.replace(temporary_path, path)
        except OSError as exc:
            raise SplitManifestError(f"Cannot write Dataset B split manifest: {path}") from exc
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
        return path

    @staticmethod
    def load_manifest(manifest_path: str | Path) -> SplitManifest:
        """Load a persisted Dataset B manifest for downstream read-only use."""
        path = Path(manifest_path)
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise SplitManifestError(f"Cannot read Dataset B split manifest: {path}") from exc

        if not isinstance(payload, Mapping) or payload.get("dataset") != "dataset_b":
            raise SplitManifestError("Dataset B split manifest has an invalid dataset identifier.")
        if payload.get("split_unit") != "record_id" or payload.get("grouping_constraint") != "none":
            raise SplitManifestError("Dataset B split manifest has invalid split boundaries.")

        try:
            manifest = SplitManifest(
                train_record_ids=tuple(payload["train_record_ids"]),
                validation_record_ids=tuple(payload["validation_record_ids"]),
                test_record_ids=tuple(payload["test_record_ids"]),
                random_state=int(payload["random_state"]),
                split_version=str(payload["split_version"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SplitManifestError("Dataset B split manifest is missing required values.") from exc

        DatasetBSplitter._validate_manifest(manifest, manifest.all_record_ids)
        if manifest.random_state != 42:
            raise SplitManifestError("Dataset B split manifest requires random_state=42.")
        return manifest

    def _record_ids_and_labels(self, view: DatasetBView) -> tuple[list[str], list[int]]:
        if not view.records:
            raise SplitManifestError("Dataset B view cannot be empty.")

        record_ids: list[str] = []
        labels: list[int] = []
        for record in view.records:
            record_id = record.get("record_id")
            label = record.get("y_approved")
            if not isinstance(record_id, str) or not record_id:
                raise SplitManifestError("Every Dataset B record requires a non-empty record_id.")
            if label not in (0, 1):
                raise SplitManifestError("Every Dataset B record requires binary y_approved.")
            record_ids.append(record_id)
            labels.append(label)

        if len(set(record_ids)) != len(record_ids):
            raise SplitManifestError("Dataset B record_id values must be unique.")
        if len(set(labels)) != 2:
            raise SplitManifestError("Stratified Dataset B splitting requires both target classes.")
        return record_ids, labels

    def _validate_contract(self) -> None:
        if self._contract.get("dataset") != "dataset_b":
            raise SplitManifestError("Split contract must be for dataset_b.")
        if self._contract.get("split_unit") != "record_id":
            raise SplitManifestError("Dataset B split unit must be record_id.")
        if self._contract.get("grouping_constraint") != "none":
            raise SplitManifestError("Dataset B does not use a grouping constraint.")

        holdout = self._contract.get("holdout_split")
        cv = self._contract.get("cross_validation")
        if not isinstance(holdout, Mapping) or not isinstance(cv, Mapping):
            raise SplitManifestError("Dataset B split contract requires holdout and CV mappings.")
        if holdout.get("stratify_by") != "y_approved":
            raise SplitManifestError("Dataset B holdout split must stratify by y_approved.")
        fractions = [
            holdout.get("train_fraction"),
            holdout.get("validation_fraction"),
            holdout.get("test_fraction"),
        ]
        if fractions != [0.70, 0.15, 0.15]:
            raise SplitManifestError("Dataset B holdout split must be 70/15/15.")
        if cv.get("splitter") != "StratifiedKFold" or cv.get("n_splits") != 5:
            raise SplitManifestError("Dataset B CV must use StratifiedKFold with k=5.")
        if cv.get("shuffle") is not True or cv.get("random_state") != 42:
            raise SplitManifestError("Dataset B CV requires shuffle=True and random_state=42.")
        if cv.get("allowed_population") != "train_only":
            raise SplitManifestError("Dataset B CV must be restricted to train_only.")
        if self._contract.get("random_state") != 42:
            raise SplitManifestError("Dataset B holdout split requires random_state=42.")

    @staticmethod
    def _validate_manifest(manifest: SplitManifest, expected_record_ids: Sequence[str]) -> None:
        actual = manifest.all_record_ids
        if len(actual) != len(expected_record_ids) or set(actual) != set(expected_record_ids):
            raise SplitManifestError("Split manifest must assign every record exactly once.")
        if len(set(actual)) != len(actual):
            raise SplitManifestError("Split manifest record assignments must not overlap.")
