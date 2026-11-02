"""Selective non-negative log transform for Dataset B PlainMLP inputs."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from src.dataset_b.data import NUMERIC_VALUE_COLUMNS, TWO_CHANNEL_COLUMNS
from src.dataset_b.models.classical import LOG1P_VALUE_COLUMNS


_LOGGER = logging.getLogger(__name__)


class DatasetBNeuralLog1pTransformer(TransformerMixin, BaseEstimator):
    """Apply ``log1p`` to income and loan amount after fit-local imputation."""

    def fit(self, X: pd.DataFrame, y: object = None) -> DatasetBNeuralLog1pTransformer:
        values = self._numeric_values(X)
        self.n_features_in_ = len(TWO_CHANNEL_COLUMNS)
        self.feature_names_in_ = np.asarray(TWO_CHANNEL_COLUMNS, dtype=object)
        self._validate_domain(self._selected(values))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "n_features_in_")
        values = self._numeric_values(X)
        selected = self._selected(values)
        self._validate_domain(selected)
        transformed = np.log1p(selected)
        if not np.isfinite(transformed).all():
            raise ValueError("Dataset B neural log1p produced NaN or infinity.")

        output = X.copy(deep=True)
        output.loc[:, list(LOG1P_VALUE_COLUMNS)] = transformed
        self.last_audit_ = {
            "rows": len(output),
            "log1p_columns": LOG1P_VALUE_COLUMNS,
            "unchanged_numeric_columns": tuple(
                column
                for column in NUMERIC_VALUE_COLUMNS
                if column not in LOG1P_VALUE_COLUMNS
            ),
            "minimum_before_log1p": float(selected.min()) if selected.size else None,
            "maximum_before_log1p": float(selected.max()) if selected.size else None,
            "finite_after_log1p": True,
        }
        _LOGGER.info(
            "Applied Dataset B PlainMLP selective log1p rows=%d columns=%s min_before=%s",
            len(output), LOG1P_VALUE_COLUMNS, self.last_audit_["minimum_before_log1p"],
        )
        return output

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        check_is_fitted(self, "n_features_in_")
        if input_features is not None and tuple(input_features) != TWO_CHANNEL_COLUMNS:
            raise ValueError("Input feature names differ from Dataset B two-channel columns.")
        return np.asarray(TWO_CHANNEL_COLUMNS, dtype=object)

    @staticmethod
    def _numeric_values(X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame) or tuple(X.columns) != TWO_CHANNEL_COLUMNS:
            raise ValueError("Dataset B neural log1p input must contain the 19 two-channel columns in order.")
        try:
            values = X.loc[:, NUMERIC_VALUE_COLUMNS].to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Dataset B neural numeric value channels must contain finite numbers.") from exc
        if not np.isfinite(values).all():
            raise ValueError("Dataset B neural numeric value channels must contain finite numbers.")
        return values

    @staticmethod
    def _validate_domain(values: np.ndarray) -> None:
        if (values < 0).any():
            raise ValueError("Dataset B neural log1p requires non-negative income and loan amount values.")

    @staticmethod
    def _selected(values: np.ndarray) -> np.ndarray:
        indices = [
            NUMERIC_VALUE_COLUMNS.index(column) for column in LOG1P_VALUE_COLUMNS
        ]
        return values[:, indices]
