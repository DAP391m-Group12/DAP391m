"""Fixed-schema native tensor adapter for Dataset B TabMmini."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from src.dataset_b.data import (
    CATEGORICAL_COLUMNS,
    CATEGORICAL_SCHEMA,
    NUMERIC_VALUE_COLUMNS,
    TWO_CHANNEL_COLUMNS,
    DatasetBNumericTwoChannel,
)
from src.dataset_b.models.neural.common.log_transformer import DatasetBNeuralLog1pTransformer


_LOGGER = logging.getLogger(__name__)


class TabMMiniCategorySupportError(ValueError):
    """Raised when held-out data contains a category absent from fit data."""


@dataclass(frozen=True)
class TabMMiniNativeBatch:
    """TabMmini input tensors derived from the ordered semantic 19 columns."""

    x_num: torch.Tensor
    x_cat: torch.Tensor

    def __post_init__(self) -> None:
        if self.x_num.ndim != 2 or self.x_num.shape[1] != len(NUMERIC_VALUE_COLUMNS):
            raise ValueError("TabMmini numeric tensor must have shape [B, 4].")
        if self.x_cat.ndim != 2 or self.x_cat.shape[1] != len(CATEGORICAL_COLUMNS):
            raise ValueError("TabMmini categorical tensor must have shape [B, 15].")
        if self.x_num.shape[0] != self.x_cat.shape[0]:
            raise ValueError("TabMmini numeric and categorical tensors must align by row.")
        if self.x_num.dtype != torch.float32:
            raise ValueError("TabMmini numeric tensor must be float32.")
        if self.x_cat.dtype != torch.int64:
            raise ValueError("TabMmini categorical tensor must be int64.")
        if not torch.isfinite(self.x_num).all().item():
            raise ValueError("TabMmini numeric tensor contains NaN or infinity.")


def fixed_tabmmini_category_maps() -> dict[str, dict[str, int]]:
    """Return zero-based maps derived only from the authoritative schema."""
    maps = {
        column: {category: index for index, category in enumerate(CATEGORICAL_SCHEMA[column])}
        for column in CATEGORICAL_COLUMNS
    }
    for column, mapping in maps.items():
        forbidden = {"1111", "MISSING", "<UNK>"} & set(mapping)
        if forbidden:
            raise RuntimeError(
                f"TabMmini authoritative map for {column} contains forbidden categories: "
                f"{sorted(forbidden)}"
            )
    return maps


def tabmmini_category_cardinalities() -> tuple[int, ...]:
    """Return category cardinalities in the locked semantic-column order."""
    maps = fixed_tabmmini_category_maps()
    return tuple(len(maps[column]) for column in CATEGORICAL_COLUMNS)


def _validate_semantic_frame(X: pd.DataFrame) -> None:
    if not isinstance(X, pd.DataFrame) or tuple(X.columns) != TWO_CHANNEL_COLUMNS:
        raise ValueError(
            "Dataset B TabMmini input must contain the 19 two-channel columns in order."
        )
    try:
        numeric = X.loc[:, NUMERIC_VALUE_COLUMNS].to_numpy(dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Dataset B TabMmini numeric channels must be finite numbers.") from exc
    if not np.isfinite(numeric).all():
        raise ValueError("Dataset B TabMmini numeric channels must be finite numbers.")
    for column in CATEGORICAL_COLUMNS:
        if not X[column].isin(CATEGORICAL_SCHEMA[column]).all():
            unknown = sorted(
                {str(value) for value in X.loc[~X[column].isin(CATEGORICAL_SCHEMA[column]), column]}
            )
            raise ValueError(f"Unknown Dataset B TabMmini category in {column}: {unknown}")


def _category_counts(X: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        column: {
            category: int(X[column].eq(category).sum())
            for category in CATEGORICAL_SCHEMA[column]
        }
        for column in CATEGORICAL_COLUMNS
    }


def _support_report(
    fit_counts: Mapping[str, Mapping[str, int]],
    held_out: pd.DataFrame,
) -> dict[str, dict[str, dict[str, int | str]]]:
    held_counts = _category_counts(held_out)
    report: dict[str, dict[str, dict[str, int | str]]] = {}
    failures: list[str] = []
    warning_count = 0
    for column in CATEGORICAL_COLUMNS:
        column_report: dict[str, dict[str, int | str]] = {}
        for category in CATEGORICAL_SCHEMA[column]:
            fit_count = int(fit_counts[column][category])
            held_count = int(held_counts[column][category])
            if fit_count == 0 and held_count > 0:
                outcome = "FAIL_ABSENT_FIT_PRESENT_HELD_OUT"
                failures.append(
                    f"column={column} category={category!r} held_out_support={held_count}"
                )
            elif fit_count == 0:
                outcome = "WARN_ABSENT_BOTH"
                warning_count += 1
            else:
                outcome = "PASS_SUPPORTED_IN_FIT"
            column_report[category] = {
                "fit": fit_count,
                "held_out": held_count,
                "outcome": outcome,
            }
        report[column] = column_report
    if warning_count:
        _LOGGER.warning(
            "Dataset B TabMmini categories absent from fit and held-out count=%d",
            warning_count,
        )
    if failures:
        raise TabMMiniCategorySupportError(
            "Dataset B TabMmini category support gate failed: " + "; ".join(failures)
        )
    return report


def validate_tabmmini_category_support(
    fit_frame: pd.DataFrame,
    held_out_frame: pd.DataFrame,
) -> dict[str, dict[str, dict[str, int | str]]]:
    """Apply the locked support gate without fitting a vocabulary."""
    _validate_semantic_frame(fit_frame)
    _validate_semantic_frame(held_out_frame)
    return _support_report(_category_counts(fit_frame), held_out_frame)


class DatasetBTabMMiniNativeAdapter(TransformerMixin, BaseEstimator):
    """Map semantic 19 columns to finite float32 and fixed-schema int64 tensors."""

    def fit(self, X: pd.DataFrame, y: object = None) -> "DatasetBTabMMiniNativeAdapter":
        _validate_semantic_frame(X)
        self.n_features_in_ = len(TWO_CHANNEL_COLUMNS)
        self.feature_names_in_ = np.asarray(TWO_CHANNEL_COLUMNS, dtype=object)
        self.numeric_columns_ = tuple(NUMERIC_VALUE_COLUMNS)
        self.categorical_columns_ = tuple(CATEGORICAL_COLUMNS)
        self.category_maps_ = fixed_tabmmini_category_maps()
        self.fit_category_counts_ = _category_counts(X)
        return self

    def transform(self, X: pd.DataFrame) -> TabMMiniNativeBatch:
        check_is_fitted(
            self,
            (
                "feature_names_in_",
                "numeric_columns_",
                "categorical_columns_",
                "category_maps_",
                "fit_category_counts_",
            ),
        )
        _validate_semantic_frame(X)
        support = _support_report(self.fit_category_counts_, X)
        numeric = X.loc[:, self.numeric_columns_].to_numpy(dtype=np.float32, copy=True)
        categorical = np.empty((len(X), len(self.categorical_columns_)), dtype=np.int64)
        for position, column in enumerate(self.categorical_columns_):
            mapped = X[column].map(self.category_maps_[column])
            if mapped.isna().any():
                raise ValueError(f"Unknown Dataset B TabMmini category in {column}.")
            categorical[:, position] = mapped.to_numpy(dtype=np.int64)
        batch = TabMMiniNativeBatch(
            x_num=torch.from_numpy(numeric),
            x_cat=torch.from_numpy(categorical),
        )
        self.last_category_support_ = support
        self.last_audit_ = {
            "rows": len(X),
            "numeric_dtype": "float32",
            "categorical_dtype": "int64",
            "numeric_shape": list(batch.x_num.shape),
            "categorical_shape": list(batch.x_cat.shape),
            "column_order": list(TWO_CHANNEL_COLUMNS),
        }
        return batch

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        if input_features is not None and tuple(input_features) != TWO_CHANNEL_COLUMNS:
            raise ValueError("Input feature names differ from Dataset B two-channel columns.")
        return np.asarray(TWO_CHANNEL_COLUMNS, dtype=object)


def make_tabmmini_native_preprocessing_pipeline() -> Pipeline:
    """Build a fresh normalized-15 -> semantic-19 -> TabMmini native pipeline."""
    return Pipeline(
        [
            ("two_channel", DatasetBNumericTwoChannel()),
            ("log1p_numeric", DatasetBNeuralLog1pTransformer()),
            ("native", DatasetBTabMMiniNativeAdapter()),
        ]
    )
