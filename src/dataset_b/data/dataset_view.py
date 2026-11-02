"""Build the immutable-input Dataset B view before preprocessing."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.utils.path import CONFIG_DIR


class DatasetBViewError(ValueError):
    """Raised when the source CSV or feature contract violates Dataset B rules."""


@dataclass(frozen=True)
class DatasetBView:
    """In-memory Dataset B cohort with raw tokens preserved for preprocessing."""

    records: tuple[dict[str, str | int], ...]
    model_input_columns: tuple[str, ...]
    audit_only_columns: tuple[str, ...]

    @property
    def columns(self) -> tuple[str, ...]:
        """Return the controlled Dataset B view column order."""
        return (
            "record_id",
            "action_taken",
            "y_approved",
            *self.model_input_columns,
            *self.audit_only_columns,
        )


class DatasetBViewBuilder:
    """Create a Dataset B view without applying preprocessing transformations."""

    DEFAULT_CONTRACT_PATH = CONFIG_DIR / "dataset_b" / "feature_set_v1.yaml"

    def __init__(self, contract: Mapping[str, Any]) -> None:
        self._contract = dict(contract)
        self._model_input_columns = self._required_columns("model_input_raw_columns")
        self._audit_only_columns = self._required_columns("audit_only_columns")
        self._validate_contract()

        target = self._contract.get("target")
        if not isinstance(target, Mapping):
            raise DatasetBViewError("Dataset B contract requires a target mapping.")

        self._target_source = self._required_string(target, "source_column")
        self._eligible_values = frozenset(str(value) for value in target.get("eligible_values", []))
        self._positive_values = frozenset(str(value) for value in target.get("positive_values", []))
        self._negative_values = frozenset(str(value) for value in target.get("negative_values", []))

        if not self._eligible_values or not self._positive_values or not self._negative_values:
            raise DatasetBViewError("Dataset B target values must be non-empty.")
        if self._positive_values & self._negative_values:
            raise DatasetBViewError("Positive and negative target values must not overlap.")
        if self._positive_values | self._negative_values != self._eligible_values:
            raise DatasetBViewError(
                "Eligible target values must be exactly the positive and negative values."
            )

    @classmethod
    def from_yaml(cls, contract_path: str | Path | None = None) -> "DatasetBViewBuilder":
        """Load the versioned Dataset B feature contract from YAML."""
        path = Path(contract_path) if contract_path is not None else cls.DEFAULT_CONTRACT_PATH
        try:
            with path.open("r", encoding="utf-8") as handle:
                contract = yaml.safe_load(handle)
        except OSError as exc:
            raise DatasetBViewError(f"Cannot read Dataset B feature contract: {path}") from exc

        if not isinstance(contract, Mapping):
            raise DatasetBViewError("Dataset B feature contract must be a YAML mapping.")
        return cls(contract)

    def build(self, raw_csv_path: str | Path) -> DatasetBView:
        """Return the cohort view while preserving every selected raw CSV token."""
        path = Path(raw_csv_path)
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self._validate_source_header(reader.fieldnames)
                records = tuple(self._build_records(reader))
        except OSError as exc:
            raise DatasetBViewError(f"Cannot read Dataset B raw CSV: {path}") from exc

        return DatasetBView(
            records=records,
            model_input_columns=self._model_input_columns,
            audit_only_columns=self._audit_only_columns,
        )

    def _build_records(self, reader: csv.DictReader) -> list[dict[str, str | int]]:
        records: list[dict[str, str | int]] = []
        selected_columns = (*self._model_input_columns, *self._audit_only_columns)

        for source_row_number, row in enumerate(reader, start=1):
            action_taken = row[self._target_source]
            if action_taken not in self._eligible_values:
                continue

            y_approved = 1 if action_taken in self._positive_values else 0
            record: dict[str, str | int] = {
                "record_id": f"dataset_b-{source_row_number:09d}",
                self._target_source: action_taken,
                "y_approved": y_approved,
            }
            record.update({column: row[column] for column in selected_columns})
            records.append(record)

        return records

    def _validate_contract(self) -> None:
        if self._contract.get("dataset") != "dataset_b":
            raise DatasetBViewError("Feature contract must be for dataset_b.")
        if len(set(self._model_input_columns)) != len(self._model_input_columns):
            raise DatasetBViewError("Dataset B model input columns must be unique.")
        if len(set(self._audit_only_columns)) != len(self._audit_only_columns):
            raise DatasetBViewError("Dataset B audit-only columns must be unique.")

        overlap = set(self._model_input_columns) & set(self._audit_only_columns)
        if overlap:
            raise DatasetBViewError(
                f"Dataset B model and audit columns overlap: {sorted(overlap)}"
            )

        target = self._contract.get("target")
        if isinstance(target, Mapping):
            source_column = target.get("source_column")
            if source_column in self._model_input_columns:
                raise DatasetBViewError("The target source cannot be a model input.")

    def _validate_source_header(self, fieldnames: list[str] | None) -> None:
        if fieldnames is None:
            raise DatasetBViewError("Dataset B raw CSV requires a header row.")

        required = {
            self._target_source,
            *self._model_input_columns,
            *self._audit_only_columns,
        }
        missing = sorted(required - set(fieldnames))
        if missing:
            raise DatasetBViewError(f"Dataset B raw CSV is missing columns: {missing}")

    def _required_columns(self, field_name: str) -> tuple[str, ...]:
        values = self._contract.get(field_name)
        if not isinstance(values, list) or not values or not all(
            isinstance(value, str) and value for value in values
        ):
            raise DatasetBViewError(f"Dataset B contract requires {field_name} as non-empty strings.")
        return tuple(values)

    @staticmethod
    def _required_string(mapping: Mapping[str, Any], field_name: str) -> str:
        value = mapping.get(field_name)
        if not isinstance(value, str) or not value:
            raise DatasetBViewError(f"Dataset B contract requires target.{field_name}.")
        return value
