"""Read the existing Dataset B split without creating another split."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml

from src.dataset_b.splits import DatasetBSplitter, SplitManifestError
from src.utils.path import CONFIG_DIR, DATASET_B_DIR, PROJECT_ROOT


PartitionName = Literal["train", "validation", "test"]


class SplitInputError(ValueError):
    """Raised when an existing split does not match its manifest or feature contract."""


@dataclass
class DatasetBPartition:
    """Raw model inputs and non-input fields from one materialized partition."""

    name: PartitionName
    X: pd.DataFrame
    y: pd.Series
    record_ids: pd.Series
    audit: pd.DataFrame
    action_taken: pd.Series


class DatasetBSplitLoader:
    """Load one requested partition while preserving literal CSV tokens."""

    DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "dataset_b" / "split_v1.json"
    DEFAULT_SPLIT_DIR = DATASET_B_DIR / "split_v1"
    DEFAULT_FEATURE_CONTRACT_PATH = CONFIG_DIR / "dataset_b" / "feature_set_v1.yaml"
    _FILENAMES = {"train": "train.csv", "validation": "validation.csv", "test": "test.csv"}

    def __init__(
        self,
        manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
        split_dir: str | Path = DEFAULT_SPLIT_DIR,
        feature_contract_path: str | Path = DEFAULT_FEATURE_CONTRACT_PATH,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._split_dir = Path(split_dir)
        try:
            self._manifest = DatasetBSplitter.load_manifest(manifest_path)
        except SplitManifestError as exc:
            raise SplitInputError(str(exc)) from exc

        try:
            with Path(feature_contract_path).open("r", encoding="utf-8") as handle:
                contract = yaml.safe_load(handle)
        except (OSError, yaml.YAMLError) as exc:
            raise SplitInputError("Cannot read Dataset B feature contract.") from exc

        if not isinstance(contract, dict) or contract.get("dataset") != "dataset_b":
            raise SplitInputError("Invalid Dataset B feature contract.")
        self.feature_version = contract.get("feature_version")
        self.feature_columns = self._columns(contract, "model_input_raw_columns", 12)
        self.audit_columns = self._columns(contract, "audit_only_columns", 9)
        if set(self.feature_columns) & set(self.audit_columns):
            raise SplitInputError("Model and audit columns must not overlap.")
        if any(name in self.feature_columns for name in ("record_id", "action_taken", "y_approved")):
            raise SplitInputError("Identifier or target data cannot enter X.")

        target = contract.get("target")
        if not isinstance(target, dict) or target.get("name") != "y_approved":
            raise SplitInputError("Dataset B target contract is missing y_approved.")
        if target.get("source_column") != "action_taken":
            raise SplitInputError("Dataset B target must come from action_taken.")
        self._positive = {str(value) for value in target.get("positive_values", ())}
        self._negative = {str(value) for value in target.get("negative_values", ())}
        eligible = {str(value) for value in target.get("eligible_values", ())}
        if self._positive != {"1", "2"} or self._negative != {"3"} or eligible != {"1", "2", "3"}:
            raise SplitInputError("Dataset B target mapping does not match the locked cohort.")

        self._expected_columns = (
            "record_id", "action_taken", "y_approved", *self.feature_columns, *self.audit_columns
        )
        if self._manifest.split_version != "split_v1":
            raise SplitInputError("Expected the existing split_v1 manifest.")

    @staticmethod
    def _columns(contract: dict, key: str, expected_count: int) -> tuple[str, ...]:
        values = contract.get(key)
        if (
            not isinstance(values, list)
            or len(values) != expected_count
            or not all(isinstance(value, str) and value for value in values)
            or len(set(values)) != len(values)
        ):
            raise SplitInputError(f"Dataset B requires {expected_count} unique {key} columns.")
        return tuple(values)

    def load(self, partition: PartitionName) -> DatasetBPartition:
        """Read only the named CSV and verify its ID order and target mapping."""
        if partition not in self._FILENAMES:
            raise SplitInputError("Partition must be train, validation, or test.")
        path = self._split_dir / self._FILENAMES[partition]
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                header = next(csv.reader(handle), None)
            if header != list(self._expected_columns):
                raise SplitInputError(f"Dataset B {partition} CSV schema differs from the feature contract.")
            frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
        except OSError as exc:
            raise SplitInputError(f"Cannot read Dataset B {partition} CSV: {path}") from exc
        except pd.errors.ParserError as exc:
            raise SplitInputError(f"Cannot parse Dataset B {partition} CSV: {path}") from exc

        expected_ids = getattr(self._manifest, f"{partition}_record_ids")
        if tuple(frame["record_id"]) != expected_ids:
            raise SplitInputError(f"Dataset B {partition} record order differs from the manifest.")
        action = frame["action_taken"]
        expected_y = action.map({"1": "1", "2": "1", "3": "0"})
        if expected_y.isna().any() or not frame["y_approved"].eq(expected_y).all():
            raise SplitInputError(f"Dataset B {partition} target mapping is invalid.")

        result = DatasetBPartition(
            name=partition,
            X=frame.loc[:, self.feature_columns].copy(deep=True),
            y=frame["y_approved"].astype("int8").copy(),
            record_ids=frame["record_id"].copy(),
            audit=frame.loc[:, self.audit_columns].copy(deep=True),
            action_taken=action.copy(),
        )
        self._logger.info(
            "Loaded Dataset B partition=%s rows=%d raw_features=%d feature_version=%s",
            partition,
            len(frame),
            len(self.feature_columns),
            self.feature_version,
        )
        return result
