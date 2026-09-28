"""Non-negative log transform for Dataset B neural numeric value channels."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .categorical_encoder import NUMERIC_VALUE_COLUMNS, TWO_CHANNEL_COLUMNS


_LOGGER = logging.getLogger(__name__)


class DatasetBNeuralLog1pTransformer(TransformerMixin, BaseEstimator):
    """Apply ``log1p`` only after fold-local numeric placeholders exist."""

    def fit(self, X: pd.DataFrame, y: object = None) -> DatasetBNeuralLog1pTransformer:
        values = self._numeric_values(X)
        self.n_features_in_ = len(TWO_CHANNEL_COLUMNS)
        self.feature_names_in_ = np.asarray(TWO_CHANNEL_COLUMNS, dtype=object)
        self._validate_domain(values)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "n_features_in_")
        values = self._numeric_values(X)
        self._validate_domain(values)
        transformed = np.log1p(values)
        if not np.isfinite(transformed).all():
            raise ValueError("Dataset B neural log1p produced NaN or infinity.")

        output = X.copy(deep=True)
        output.loc[:, NUMERIC_VALUE_COLUMNS] = transformed
        self.last_audit_ = {
            "rows": len(output),
            "minimum_before_log1p": float(values.min()) if values.size else None,
            "maximum_before_log1p": float(values.max()) if values.size else None,
            "finite_after_log1p": True,
        }
        _LOGGER.info(
            "Applied Dataset B neural log1p rows=%d numeric_columns=%d min_before=%s",
            len(output), len(NUMERIC_VALUE_COLUMNS), self.last_audit_["minimum_before_log1p"],
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
            raise ValueError("Dataset B neural log1p input must contain the 16 two-channel columns in order.")
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
            raise ValueError("Dataset B neural log1p requires non-negative numeric value channels.")
