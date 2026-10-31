"""Sklearn-compatible preprocessing profiles for Dataset B classical models."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from ...data.categorical_encoder import DatasetBOneHotEncoder
from ...data.numeric_two_channel import DatasetBNumericTwoChannel
from ...data.numeric_value_scaler import DatasetBNumericValueRobustScaler
from ...data.raw_parser import DatasetBRawParser, FEATURE_COLUMNS
from .lr_log_transformer import DatasetBLRSelectiveLog1pTransformer


_LOGGER = logging.getLogger(__name__)


class DatasetBRawParserStep(TransformerMixin, BaseEstimator):
    """Adapt the deterministic raw parser for a sklearn pipeline."""

    def fit(self, X: pd.DataFrame, y: object = None) -> DatasetBRawParserStep:
        DatasetBRawParser().parse(X)
        self.is_fitted_ = True
        return self

    def fit_transform(self, X: pd.DataFrame, y: object = None, **fit_params: object) -> pd.DataFrame:
        cleaned = DatasetBRawParser().parse(X).cleaned
        self.is_fitted_ = True
        return cleaned

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "is_fitted_")
        return DatasetBRawParser().parse(X).cleaned

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        check_is_fitted(self, "is_fitted_")
        if input_features is not None and tuple(input_features) != FEATURE_COLUMNS:
            raise ValueError("Input feature names differ from the Dataset B raw feature contract.")
        return np.asarray(FEATURE_COLUMNS, dtype=object)


def make_preprocessing_pipeline(profile: str) -> Pipeline:
    """Build a fresh fold-local pipeline; no class weighting is applied here."""
    if profile not in {"lr", "rf", "xgb"}:
        raise ValueError("Dataset B profile must be lr, rf or xgb.")
    selective_log = DatasetBLRSelectiveLog1pTransformer() if profile == "lr" else "passthrough"
    scale = DatasetBNumericValueRobustScaler() if profile == "lr" else "passthrough"
    _LOGGER.info(
        "Created Dataset B preprocessing profile=%s numeric_scaler=%s",
        profile,
        "RobustScaler" if profile == "lr" else "none",
    )
    return Pipeline(
        [
            ("two_channel", DatasetBNumericTwoChannel()),
            ("selective_log1p", selective_log),
            ("encode", DatasetBOneHotEncoder()),
            ("scale_numeric", scale),
        ]
    )
