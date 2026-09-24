"""Preserve numeric values and reporting states as separate Dataset B channels."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .median_imputer import DatasetBNumericMedianImputer, NUMERIC_COLUMNS
from .raw_parser import FEATURE_COLUMNS


_LOGGER = logging.getLogger(__name__)


class DatasetBNumericTwoChannel(TransformerMixin, BaseEstimator):
    """Produce numeric ``_value`` and categorical ``_status`` channels.

    Fit only on parsed fold-train rows. ``NA`` and ``Exempt`` retain their
    meanings in the status channel; the fitted median is only a numeric
    placeholder for their value channel and for genuine ``MISSING`` values.
    Other categorical features pass through unchanged for later encoding.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> DatasetBNumericTwoChannel:
        self.imputer_ = DatasetBNumericMedianImputer().fit(X, y)
        _LOGGER.info("Fitted Dataset B numeric two-channel rows=%d", len(X))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "imputer_")
        checked = self.imputer_.transform(X)
        output: dict[str, pd.Series] = {}
        status_counts: dict[str, dict[str, int]] = {}

        for column in FEATURE_COLUMNS:
            if column not in NUMERIC_COLUMNS:
                output[column] = X[column].copy(deep=True)
                continue

            status = X[column].map(
                lambda value: value if isinstance(value, str) else "OBSERVED"
            )
            value = checked[column].map(
                lambda item: self.imputer_.medians_[column]
                if isinstance(item, str) else float(item)
            ).astype("float64")
            output[f"{column}_value"] = value
            output[f"{column}_status"] = status
            status_counts[column] = status.value_counts().to_dict()

        result = pd.DataFrame(output, index=X.index)
        _LOGGER.info(
            "Transformed Dataset B numeric two-channel rows=%d status_counts=%s",
            len(result), status_counts,
        )
        return result

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        check_is_fitted(self, "imputer_")
        if input_features is not None and tuple(input_features) != FEATURE_COLUMNS:
            raise ValueError("Input feature names differ from Dataset B raw columns.")
        names = [
            field
            for column in FEATURE_COLUMNS
            for field in ((f"{column}_value", f"{column}_status") if column in NUMERIC_COLUMNS else (column,))
        ]
        return np.asarray(names, dtype=object)
