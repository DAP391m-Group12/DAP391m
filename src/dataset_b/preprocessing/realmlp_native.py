"""Typed native-input preprocessing for Dataset B RealMLP."""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .categorical_encoder import (
    CATEGORICAL_COLUMNS,
    CATEGORICAL_SCHEMA,
    NUMERIC_VALUE_COLUMNS,
    TWO_CHANNEL_COLUMNS,
)
from .neural_log_transformer import DatasetBNeuralLog1pTransformer
from .numeric_two_channel import DatasetBNumericTwoChannel


_LOGGER = logging.getLogger(__name__)


class RealMLPCategorySupportError(ValueError):
    """Raised when held-out input contains a category absent from fold-train."""


def _validate_native_frame(X: pd.DataFrame) -> None:
    if not isinstance(X, pd.DataFrame) or tuple(X.columns) != TWO_CHANNEL_COLUMNS:
        raise ValueError(
            "Dataset B RealMLP native input must contain the 19 two-channel columns in order."
        )

    try:
        numeric = X.loc[:, NUMERIC_VALUE_COLUMNS].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Dataset B RealMLP numeric value channels must be finite numbers.") from exc
    if not np.isfinite(numeric).all():
        raise ValueError("Dataset B RealMLP numeric value channels must be finite numbers.")

    for column in CATEGORICAL_COLUMNS:
        if not X[column].isin(CATEGORICAL_SCHEMA[column]).all():
            raise ValueError(f"Unexpected Dataset B category in {column}.")


def _category_counts(X: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        column: {
            category: int(X[column].eq(category).sum())
            for category in CATEGORICAL_SCHEMA[column]
        }
        for column in CATEGORICAL_COLUMNS
    }


def _support_report(
    fold_train_counts: Mapping[str, Mapping[str, int]],
    fold_validation: pd.DataFrame,
) -> dict[str, dict[str, dict[str, int | str]]]:
    validation_counts = _category_counts(fold_validation)
    report: dict[str, dict[str, dict[str, int | str]]] = {}
    failures: list[str] = []
    warnings: list[str] = []

    for column in CATEGORICAL_COLUMNS:
        column_report: dict[str, dict[str, int | str]] = {}
        for category in CATEGORICAL_SCHEMA[column]:
            train_count = int(fold_train_counts[column][category])
            validation_count = validation_counts[column][category]
            if train_count == 0 and validation_count > 0:
                outcome = "fail_absent_train_present_validation"
                failures.append(
                    f"column={column} category={category!r} "
                    f"fold_validation_support={validation_count}"
                )
            elif train_count == 0:
                outcome = "warning_absent_both"
                warnings.append(f"column={column} category={category!r}")
            else:
                outcome = "pass_supported_in_fold_train"
            column_report[category] = {
                "fold_train": train_count,
                "fold_validation": validation_count,
                "outcome": outcome,
            }
        report[column] = column_report

    if warnings:
        _LOGGER.warning(
            "Dataset B RealMLP categories absent from fold-train and held-out input count=%d",
            len(warnings),
        )
    if failures:
        raise RealMLPCategorySupportError(
            "Dataset B RealMLP absent-category gate failed: " + "; ".join(failures)
        )
    return report


def validate_realmlp_category_support(
    fold_train: pd.DataFrame,
    fold_validation: pd.DataFrame,
) -> dict[str, dict[str, dict[str, int | str]]]:
    """Validate the locked absent-category gate for one Train-only split."""
    _validate_native_frame(fold_train)
    _validate_native_frame(fold_validation)
    return _support_report(_category_counts(fold_train), fold_validation)


class DatasetBRealMLPNativeAdapter(TransformerMixin, BaseEstimator):
    """Emit finite float32 numeric values and explicit categorical columns.

    The adapter fits no numerical statistic. It records fold-train category
    support so a later transform fails when a schema-valid category appears in
    held-out input without any fold-train support.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> "DatasetBRealMLPNativeAdapter":
        _validate_native_frame(X)
        self.n_features_in_ = len(TWO_CHANNEL_COLUMNS)
        self.feature_names_in_ = np.asarray(TWO_CHANNEL_COLUMNS, dtype=object)
        self.numeric_columns_ = tuple(NUMERIC_VALUE_COLUMNS)
        self.categorical_columns_ = tuple(CATEGORICAL_COLUMNS)
        self.category_schema_ = {
            column: tuple(CATEGORICAL_SCHEMA[column]) for column in CATEGORICAL_COLUMNS
        }
        self.fold_train_category_counts_ = _category_counts(X)
        _LOGGER.info(
            "Fitted Dataset B RealMLP native adapter rows=%d numeric_columns=%d "
            "categorical_columns=%d",
            len(X),
            len(self.numeric_columns_),
            len(self.categorical_columns_),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(
            self,
            (
                "feature_names_in_",
                "numeric_columns_",
                "categorical_columns_",
                "category_schema_",
                "fold_train_category_counts_",
            ),
        )
        _validate_native_frame(X)
        support = _support_report(self.fold_train_category_counts_, X)

        output = X.copy(deep=True)
        numeric = output.loc[:, self.numeric_columns_].to_numpy(dtype=np.float32)
        if not np.isfinite(numeric).all():
            raise ValueError("Dataset B RealMLP float32 conversion produced NaN or infinity.")
        for position, column in enumerate(self.numeric_columns_):
            output[column] = numeric[:, position]
        for column in self.categorical_columns_:
            output[column] = output[column].astype(
                pd.CategoricalDtype(categories=list(self.category_schema_[column]), ordered=False)
            )

        self.last_category_support_ = support
        self.last_audit_ = {
            "rows": len(output),
            "numeric_dtype": "float32",
            "numeric_finite": True,
            "categorical_dtype": "category",
            "column_order": list(TWO_CHANNEL_COLUMNS),
        }
        _LOGGER.info(
            "Transformed Dataset B RealMLP native input rows=%d numeric_dtype=float32",
            len(output),
        )
        return output

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        if input_features is not None and tuple(input_features) != TWO_CHANNEL_COLUMNS:
            raise ValueError("Input feature names differ from Dataset B two-channel columns.")
        return np.asarray(TWO_CHANNEL_COLUMNS, dtype=object)


def make_realmlp_native_preprocessing_pipeline() -> Pipeline:
    """Build a fresh normalized-15 -> two-channel -> selective-log1p profile."""
    return Pipeline(
        [
            ("two_channel", DatasetBNumericTwoChannel()),
            ("log1p_numeric", DatasetBNeuralLog1pTransformer()),
            ("native", DatasetBRealMLPNativeAdapter()),
        ]
    )
