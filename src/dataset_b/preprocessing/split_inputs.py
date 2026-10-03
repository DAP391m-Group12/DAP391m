"""Read the existing Dataset B split without creating another split."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

import pandas as pd
import yaml

from src.dataset_b.splits import DatasetBSplitter, SplitManifestError
from src.utils.path import CONFIG_DIR, DATASET_B_DIR, PROJECT_ROOT

from .raw_parser import DatasetBRawParser, FEATURE_COLUMNS, RawFeatureError


PartitionName = Literal["train", "validation", "test"]
FINAL_MODEL_INPUT_COUNT = 15
AUDIT_COLUMN_COUNT = 9


class SplitInputError(ValueError):
    """Raised when an existing split does not match its manifest or feature contract."""


class InputQualityError(ValueError):
    """Raised when a loaded partition fails the final input-quality contract."""


@dataclass
class DatasetBPartition:
    """Raw model inputs and non-input fields from one materialized partition."""

    name: PartitionName
    X: pd.DataFrame
    y: pd.Series
    record_ids: pd.Series
    audit: pd.DataFrame
    action_taken: pd.Series


@dataclass(frozen=True)
class DatasetBInputQualityAudit:
    """Direct evidence that one partition passed the final input-quality gate."""

    status: str
    input_rows: int
    retained_rows: int
    rejected_rows: int
    rejected_record_ids: tuple[str, ...]
    blank_counts: Mapping[str, int]
    raw_1111_counts: Mapping[str, int]
    normalized_exempt_counts: Mapping[str, int]
    semantic_1111_count: int
    row_count_preserved_after_normalization: bool
    row_order_preserved: bool
    target_alignment_preserved: bool
    record_id_alignment_preserved: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "input_rows": self.input_rows,
            "retained_rows": self.retained_rows,
            "rejected_rows": self.rejected_rows,
            "rejected_record_ids": list(self.rejected_record_ids),
            "blank_counts": dict(self.blank_counts),
            "raw_1111_counts": dict(self.raw_1111_counts),
            "normalized_exempt_counts": dict(self.normalized_exempt_counts),
            "semantic_1111_count": self.semantic_1111_count,
            "row_count_preserved_after_normalization": (
                self.row_count_preserved_after_normalization
            ),
            "row_order_preserved": self.row_order_preserved,
            "target_alignment_preserved": self.target_alignment_preserved,
            "record_id_alignment_preserved": self.record_id_alignment_preserved,
        }


@dataclass
class DatasetBQualityGatedPartition:
    """Retained, normalized model inputs with aligned partition metadata."""

    name: PartitionName
    X: pd.DataFrame
    y: pd.Series
    record_ids: pd.Series
    audit: pd.DataFrame
    action_taken: pd.Series
    quality_audit: DatasetBInputQualityAudit


class DatasetBInputQualityGate:
    """Reject blank rows, validate domains, and normalize exemption flags."""

    def __init__(self, *, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger(__name__)

    def apply(self, partition: DatasetBPartition) -> DatasetBQualityGatedPartition:
        self._validate_partition_alignment(partition)
        raw_X = partition.X.copy(deep=True)
        blank_frame = raw_X.apply(lambda column: column.map(DatasetBRawParser.is_blank))
        blank_counts = {
            column: int(blank_frame[column].sum()) for column in FEATURE_COLUMNS
        }
        rejected_mask = blank_frame.any(axis=1).to_numpy(dtype=bool)
        retained_positions = tuple(
            position for position, rejected in enumerate(rejected_mask) if not rejected
        )
        rejected_positions = tuple(
            position for position, rejected in enumerate(rejected_mask) if rejected
        )
        if rejected_positions:
            rejected_ids = tuple(
                partition.record_ids.iloc[list(rejected_positions)].astype(str)
            )
            raise InputQualityError(
                "Dataset B input-quality gate rejected one or more rows; "
                f"rejected_record_ids={rejected_ids}."
            )

        retained_X = raw_X.iloc[list(retained_positions)].copy(deep=True)
        retained_y = partition.y.iloc[list(retained_positions)].copy()
        retained_ids = partition.record_ids.iloc[list(retained_positions)].copy()
        retained_audit = partition.audit.iloc[list(retained_positions)].copy(deep=True)
        retained_action = partition.action_taken.iloc[list(retained_positions)].copy()
        try:
            parsed = DatasetBRawParser(logger=self._logger).parse(retained_X)
        except RawFeatureError as exc:
            raise InputQualityError(f"Dataset B input-quality gate failed: {exc}") from exc

        if parsed.cleaned.astype(str).eq("MISSING").to_numpy().any():
            raise InputQualityError("Dataset B semantic inputs must not contain MISSING.")
        diagnostics = parsed.diagnostics
        if diagnostics.raw_1111_counts != diagnostics.normalized_exempt_counts:
            raise InputQualityError("Dataset B raw and normalized exemption counts differ.")
        if diagnostics.semantic_1111_count != 0:
            raise InputQualityError("Dataset B semantic exemption flags retain raw 1111.")

        row_order_preserved = parsed.cleaned.index.equals(retained_X.index)
        record_id_alignment_preserved = retained_ids.index.equals(retained_X.index)
        target_alignment_preserved = (
            retained_y.index.equals(retained_X.index)
            and retained_action.index.equals(retained_X.index)
        )
        if not (
            row_order_preserved
            and record_id_alignment_preserved
            and target_alignment_preserved
        ):
            raise InputQualityError("Dataset B retained row alignment changed during gating.")

        audit = DatasetBInputQualityAudit(
            status="PASS",
            input_rows=len(partition.X),
            retained_rows=len(parsed.cleaned),
            rejected_rows=len(rejected_positions),
            rejected_record_ids=(),
            blank_counts=blank_counts,
            raw_1111_counts=dict(diagnostics.raw_1111_counts),
            normalized_exempt_counts=dict(diagnostics.normalized_exempt_counts),
            semantic_1111_count=diagnostics.semantic_1111_count,
            row_count_preserved_after_normalization=(
                len(parsed.raw) == len(parsed.cleaned) == len(retained_X)
            ),
            row_order_preserved=row_order_preserved,
            target_alignment_preserved=target_alignment_preserved,
            record_id_alignment_preserved=record_id_alignment_preserved,
        )
        self._logger.info(
            "Dataset B input-quality gate status=PASS partition=%s input_rows=%d retained_rows=%d rejected_rows=%d",
            partition.name,
            audit.input_rows,
            audit.retained_rows,
            audit.rejected_rows,
        )
        return DatasetBQualityGatedPartition(
            name=partition.name,
            X=parsed.cleaned,
            y=retained_y,
            record_ids=retained_ids,
            audit=retained_audit,
            action_taken=retained_action,
            quality_audit=audit,
        )

    @staticmethod
    def _validate_partition_alignment(partition: DatasetBPartition) -> None:
        if not isinstance(partition.X, pd.DataFrame) or tuple(partition.X.columns) != FEATURE_COLUMNS:
            raise InputQualityError("Dataset B quality gate requires the exact final 15-field schema.")
        rows = len(partition.X)
        aligned = (
            len(partition.y) == rows
            and len(partition.record_ids) == rows
            and len(partition.audit) == rows
            and len(partition.action_taken) == rows
            and partition.y.index.equals(partition.X.index)
            and partition.record_ids.index.equals(partition.X.index)
            and partition.audit.index.equals(partition.X.index)
            and partition.action_taken.index.equals(partition.X.index)
        )
        if not aligned:
            raise InputQualityError("Dataset B partition rows or indexes are misaligned.")
        record_ids = partition.record_ids.astype(str)
        if record_ids.map(DatasetBRawParser.is_blank).any() or not record_ids.is_unique:
            raise InputQualityError("Dataset B record_id values must be nonblank and unique.")
        expected_y = partition.action_taken.astype(str).map({"1": 1, "2": 1, "3": 0})
        if expected_y.isna().any() or not partition.y.astype(int).eq(expected_y.astype(int)).all():
            raise InputQualityError("Dataset B target alignment or mapping is invalid.")


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
        self.feature_columns = self._columns(
            contract,
            "model_input_raw_columns",
            FINAL_MODEL_INPUT_COUNT,
        )
        self.audit_columns = self._columns(
            contract,
            "audit_only_columns",
            AUDIT_COLUMN_COUNT,
        )
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
