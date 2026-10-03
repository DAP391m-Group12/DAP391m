"""Sklearn-compatible preprocessing profiles for Dataset B phase-one models."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, issparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.utils.validation import check_is_fitted

from .categorical_encoder import ENCODED_FEATURE_COUNT, DatasetBOneHotEncoder
from .lr_log_transformer import DatasetBLRSelectiveLog1pTransformer
from .numeric_two_channel import DatasetBNumericTwoChannel
from .raw_parser import DatasetBRawParser, FEATURE_COLUMNS


_LOGGER = logging.getLogger(__name__)
_NUMERIC_VALUE_COUNT = 4


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


class DatasetBNumericValueRobustScaler(TransformerMixin, BaseEstimator):
    """Center and IQR-scale only four numeric values, leaving OHE untouched."""

    def fit(self, X: csr_matrix, y: object = None) -> DatasetBNumericValueRobustScaler:
        self._validate_matrix(X)
        self.n_features_in_ = X.shape[1]
        self.scaler_ = RobustScaler().fit(X[:, :_NUMERIC_VALUE_COUNT].toarray())
        _LOGGER.info(
            "Fitted Dataset B numeric RobustScaler rows=%d center=%s scale=%s",
            X.shape[0], self.scaler_.center_.tolist(), self.scaler_.scale_.tolist(),
        )
        return self

    def transform(self, X: csr_matrix) -> csr_matrix:
        check_is_fitted(self, "scaler_")
        self._validate_matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Dataset B encoded feature count differs from fitted scaler input.")
        values = csr_matrix(self.scaler_.transform(X[:, :_NUMERIC_VALUE_COUNT].toarray()))
        result = hstack((values, X[:, _NUMERIC_VALUE_COUNT:]), format="csr")
        result = result.astype(np.float64, copy=False)
        _LOGGER.info("Transformed Dataset B numeric RobustScaler rows=%d columns=%d", *result.shape)
        return result

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        check_is_fitted(self, "scaler_")
        if input_features is None:
            return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        names = tuple(input_features)
        if len(names) != self.n_features_in_:
            raise ValueError("Input feature names differ from fitted scaler width.")
        return np.asarray(names, dtype=object)

    @staticmethod
    def _validate_matrix(X: csr_matrix) -> None:
        if not issparse(X) or X.ndim != 2 or X.shape[1] != ENCODED_FEATURE_COUNT:
            raise ValueError("Dataset B numeric scaler expects the 62-column sparse encoded matrix.")
        if not np.isfinite(X.data).all():
            raise ValueError("Dataset B encoded matrix contains non-finite values.")


def make_preprocessing_pipeline(profile: str) -> Pipeline:
    """Build a fresh fold-local pipeline; no class weighting is applied here."""
    if profile not in {"lr", "rf", "xgb"}:
        raise ValueError("Dataset B profile must be lr, rf or xgb.")
    selective_log = DatasetBLRSelectiveLog1pTransformer() if profile == "lr" else "passthrough"
    scale = DatasetBNumericValueRobustScaler() if profile == "lr" else "passthrough"
    _LOGGER.info("Created Dataset B preprocessing profile=%s numeric_scaler=%s", profile, "RobustScaler" if profile == "lr" else "none")
    return Pipeline(
        [
            ("two_channel", DatasetBNumericTwoChannel()),
            ("selective_log1p", selective_log),
            ("encode", DatasetBOneHotEncoder()),
            ("scale_numeric", scale),
        ]
    )
