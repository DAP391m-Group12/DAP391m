"""Validate Dataset B raw feature tokens without fitting model preprocessing."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Mapping

import pandas as pd


class RawFeatureError(ValueError):
    """Raised when a raw feature value falls outside its agreed domain."""


_CATEGORICAL_DOMAINS = {
    "loan_type": frozenset({"1", "2", "3", "4"}),
    "loan_purpose": frozenset({"1", "2", "31", "32", "4", "5"}),
    "lien_status": frozenset({"1", "2"}),
    "reverse_mortgage": frozenset({"1", "2", "1111"}),
    "open-end_line_of_credit": frozenset({"1", "2", "1111"}),
    "business_or_commercial_purpose": frozenset({"1", "2", "1111"}),
    "occupancy_type": frozenset({"1", "2", "3"}),
}
_DTI_DOMAIN = frozenset(
    {"<20%", "20%-<30%", "30%-<36%", "50%-60%", ">60%", "NA", "Exempt"}
    | {str(value) for value in range(36, 50)}
)
_NUMERIC_SPECIAL = {
    "income": frozenset({"NA"}),
    "loan_amount": frozenset(),
    "loan_to_value_ratio": frozenset({"NA", "Exempt"}),
    "loan_term": frozenset({"NA", "Exempt"}),
}
FEATURE_COLUMNS = (
    "loan_type", "loan_purpose", "lien_status", "reverse_mortgage",
    "open-end_line_of_credit", "business_or_commercial_purpose", "income",
    "debt_to_income_ratio", "loan_amount", "loan_to_value_ratio", "loan_term",
    "occupancy_type",
)


@dataclass(frozen=True)
class ParseDiagnostics:
    rows: int
    values_to_missing: Mapping[str, int]
    income_negative_to_zero: int
    ltv_gt_1000: int

    @property
    def nan_to_missing(self) -> Mapping[str, int]:
        """Compatibility alias; counts now include blanks and whitespace."""
        return self.values_to_missing


@dataclass
class ParsedFeatures:
    """Copies of raw and validated values; no encoder or imputer has been fit."""

    raw: pd.DataFrame
    cleaned: pd.DataFrame
    diagnostics: ParseDiagnostics


class DatasetBRawParser:
    """Apply only agreed, deterministic Dataset B raw-value rules."""

    def __init__(self, *, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger(__name__)

    def parse(self, X: pd.DataFrame) -> ParsedFeatures:
        if not isinstance(X, pd.DataFrame) or tuple(X.columns) != FEATURE_COLUMNS:
            raise RawFeatureError("Raw Dataset B input must contain the 12 locked features in order.")

        raw = X.copy(deep=True)
        cleaned = raw.copy(deep=True)
        missing_counts: dict[str, int] = {}
        income_negative = 0
        ltv_extreme = 0
        for column in FEATURE_COLUMNS:
            output: list[str | float] = []
            missing_count = 0
            for row_number, value in enumerate(raw[column].tolist(), start=1):
                if self._is_missing(value):
                    output.append("MISSING")
                    missing_count += 1
                    continue
                if not isinstance(value, str):
                    raise RawFeatureError(f"Invalid type in {column} at row {row_number}.")
                if column in _CATEGORICAL_DOMAINS:
                    if value not in _CATEGORICAL_DOMAINS[column]:
                        raise RawFeatureError(f"Invalid {column} code at row {row_number}: {value!r}.")
                    output.append(value)
                    continue
                if column == "debt_to_income_ratio":
                    if value not in _DTI_DOMAIN:
                        raise RawFeatureError(f"Invalid DTI category at row {row_number}: {value!r}.")
                    output.append(value)
                    continue
                if value in _NUMERIC_SPECIAL[column]:
                    output.append(value)
                    continue
                try:
                    number = float(value)
                except ValueError as exc:
                    raise RawFeatureError(f"Invalid numeric {column} at row {row_number}: {value!r}.") from exc
                if not math.isfinite(number):
                    raise RawFeatureError(f"Non-finite numeric {column} at row {row_number}.")
                if column in {"loan_amount", "loan_to_value_ratio", "loan_term"} and number <= 0:
                    raise RawFeatureError(f"Nonpositive numeric {column} at row {row_number}.")
                if column == "loan_term" and not number.is_integer():
                    raise RawFeatureError(f"Non-integer loan_term at row {row_number}.")
                if column == "income" and number < 0:
                    number = 0.0
                    income_negative += 1
                if column == "loan_to_value_ratio" and number > 1000:
                    ltv_extreme += 1
                output.append(number)
            cleaned[column] = pd.Series(output, index=raw.index, dtype=object)
            missing_counts[column] = missing_count

        diagnostics = ParseDiagnostics(
            rows=len(raw),
            values_to_missing=missing_counts,
            income_negative_to_zero=income_negative,
            ltv_gt_1000=ltv_extreme,
        )
        self._logger.info(
            "Parsed Dataset B raw features rows=%d income_negative_to_zero=%d values_to_missing=%d",
            len(raw),
            income_negative,
            sum(missing_counts.values()),
        )
        if ltv_extreme:
            self._logger.warning("Dataset B LTV >1000%% retained for %d rows", ltv_extreme)
        return ParsedFeatures(raw=raw, cleaned=cleaned, diagnostics=diagnostics)

    @staticmethod
    def _is_missing(value: object) -> bool:
        if isinstance(value, str):
            return value.strip() == "" or value.strip().lower() == "nan"
        return value is None or bool(pd.isna(value))
