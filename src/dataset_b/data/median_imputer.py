"""Fold-local median imputation for the four Dataset B numeric features."""

from __future__ import annotations

import logging
import math
from numbers import Real
from statistics import median

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .raw_parser import FEATURE_COLUMNS


NUMERIC_COLUMNS = ("income", "loan_amount", "loan_to_value_ratio", "loan_term")
_SPECIAL = {
    "income": frozenset({"NA"}),
    "loan_amount": frozenset(),
    "loan_to_value_ratio": frozenset({"NA", "Exempt"}),
    "loan_term": frozenset({"NA", "Exempt"}),
}
_LOGGER = logging.getLogger(__name__)


class DatasetBNumericMedianImputer(TransformerMixin, BaseEstimator):
    """Fit numeric placeholder medians on one fold-train.

    Input is the final 15-column semantic frame returned by the quality gate.
    ``NA`` and ``Exempt`` remain labels; their value-channel placeholders are
    supplied by ``DatasetBNumericTwoChannel`` from these fold-local medians.
    Create a fresh instance (or sklearn clone) for each cross-validation fold.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> DatasetBNumericMedianImputer:
        self._validate_frame(X)
        fitted: dict[str, float] = {}
        observed_counts: dict[str, int] = {}
        for column in NUMERIC_COLUMNS:
            values = self._numeric_values(X[column], column)
            if not values:
                raise ValueError(f"Cannot fit median for {column}: no observed numeric values in fold-train.")
            fitted[column] = float(median(values))
            observed_counts[column] = len(values)
        self.medians_ = fitted
        self.observed_counts_ = observed_counts
        _LOGGER.info("Fitted Dataset B fold-train numeric medians=%s observed_counts=%s", fitted, observed_counts)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "medians_")
        self._validate_frame(X)
        output = X.copy(deep=True)
        placeholder_counts: dict[str, int] = {}
        for column in NUMERIC_COLUMNS:
            self._numeric_values(output[column], column)
            placeholder_counts[column] = int(output[column].isin(_SPECIAL[column]).sum())
        _LOGGER.info(
            "Transformed Dataset B rows=%d numeric_special_placeholders=%s",
            len(output), placeholder_counts,
        )
        return output

    @staticmethod
    def _validate_frame(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame) or tuple(X.columns) != FEATURE_COLUMNS:
            raise ValueError("Dataset B median input must contain the 15 locked features in order.")

    @staticmethod
    def _numeric_values(series: pd.Series, column: str) -> list[float]:
        values: list[float] = []
        for row_number, value in enumerate(series.tolist(), start=1):
            if isinstance(value, str) and value in _SPECIAL[column]:
                continue
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"Invalid parsed numeric {column} at row {row_number}: {value!r}.")
            values.append(float(value))
        return values
