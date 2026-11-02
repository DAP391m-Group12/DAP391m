"""Selective log transform for Dataset B logistic-regression preprocessing."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ...data.numeric_two_channel import NUMERIC_VALUE_COLUMNS, TWO_CHANNEL_COLUMNS


_LOGGER = logging.getLogger(__name__)
LOG1P_VALUE_COLUMNS = ("income_value", "loan_amount_value")


class DatasetBLRSelectiveLog1pTransformer(TransformerMixin, BaseEstimator):
    """Apply log1p to income and loan amount only, preserving all other columns."""

    def fit(self, X: pd.DataFrame, y: object = None) -> DatasetBLRSelectiveLog1pTransformer:
        self._validate(X)
        self.n_features_in_ = len(TWO_CHANNEL_COLUMNS)
        self.feature_names_in_ = np.asarray(TWO_CHANNEL_COLUMNS, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "n_features_in_")
        values = self._validate(X)
        selected_indices = [NUMERIC_VALUE_COLUMNS.index(column) for column in LOG1P_VALUE_COLUMNS]
        selected = values[:, selected_indices]
        if (selected < 0).any():
            raise ValueError("Dataset B LR selective log1p requires non-negative income and loan amount values.")
        transformed = np.log1p(selected)
        if not np.isfinite(transformed).all():
            raise ValueError("Dataset B LR selective log1p produced NaN or infinity.")

        output = X.copy(deep=True)
        output.loc[:, list(LOG1P_VALUE_COLUMNS)] = transformed
        self.last_audit_ = {
            "rows": len(output),
            "log1p_columns": LOG1P_VALUE_COLUMNS,
            "unchanged_numeric_columns": tuple(
                column for column in NUMERIC_VALUE_COLUMNS if column not in LOG1P_VALUE_COLUMNS
            ),
            "finite_after_log1p": True,
        }
        _LOGGER.info(
            "Applied Dataset B LR selective log1p rows=%d columns=%s",
            len(output), LOG1P_VALUE_COLUMNS,
        )
        return output

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        check_is_fitted(self, "n_features_in_")
        if input_features is not None and tuple(input_features) != TWO_CHANNEL_COLUMNS:
            raise ValueError("Input feature names differ from Dataset B two-channel columns.")
        return np.asarray(TWO_CHANNEL_COLUMNS, dtype=object)

    @staticmethod
    def _validate(X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame) or tuple(X.columns) != TWO_CHANNEL_COLUMNS:
            raise ValueError("Dataset B LR log1p input must contain the 19 two-channel columns in order.")
        try:
            values = X.loc[:, NUMERIC_VALUE_COLUMNS].to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Dataset B LR numeric value channels must contain finite numbers.") from exc
        if not np.isfinite(values).all():
            raise ValueError("Dataset B LR numeric value channels must contain finite numbers.")
        return values
