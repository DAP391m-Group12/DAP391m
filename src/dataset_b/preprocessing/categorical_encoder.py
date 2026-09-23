"""One-hot encode Dataset B categories and numeric reporting states."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

from .median_imputer import NUMERIC_COLUMNS
from .raw_parser import FEATURE_COLUMNS


_LOGGER = logging.getLogger(__name__)

_DTI_CATEGORIES = (
    "<20%", "20%-<30%", "30%-<36%",
    *(str(value) for value in range(36, 50)),
    "50%-60%", ">60%", "NA", "Exempt", "MISSING",
)
_RAW_CATEGORIES = {
    "loan_type": ("1", "2", "3", "4", "MISSING"),
    "loan_purpose": ("1", "2", "31", "32", "4", "5", "MISSING"),
    "lien_status": ("1", "2", "MISSING"),
    "reverse_mortgage": ("1", "2", "1111", "MISSING"),
    "open-end_line_of_credit": ("1", "2", "1111", "MISSING"),
    "business_or_commercial_purpose": ("1", "2", "1111", "MISSING"),
    "debt_to_income_ratio": _DTI_CATEGORIES,
    "occupancy_type": ("1", "2", "3", "MISSING"),
}
_STATUS_CATEGORIES = {
    "income_status": ("OBSERVED", "NA", "MISSING"),
    "loan_amount_status": ("OBSERVED", "MISSING"),
    "loan_to_value_ratio_status": ("OBSERVED", "NA", "Exempt", "MISSING"),
    "loan_term_status": ("OBSERVED", "NA", "Exempt", "MISSING"),
}
_CATEGORIES = {**_RAW_CATEGORIES, **_STATUS_CATEGORIES}
_TWO_CHANNEL_COLUMNS = tuple(
    field
    for column in FEATURE_COLUMNS
    for field in ((f"{column}_value", f"{column}_status") if column in NUMERIC_COLUMNS else (column,))
)
_NUMERIC_VALUE_COLUMNS = tuple(f"{column}_value" for column in NUMERIC_COLUMNS)
_CATEGORICAL_COLUMNS = tuple(column for column in _TWO_CHANNEL_COLUMNS if column in _CATEGORIES)
_STATUS_REFERENCE_CATEGORY = "OBSERVED"
STATUS_REFERENCE_CATEGORIES = tuple(
    (f"{column}_status", _STATUS_REFERENCE_CATEGORY)
    for column in NUMERIC_COLUMNS
)
ENCODED_FEATURE_COUNT = 66


class DatasetBOneHotEncoder(TransformerMixin, BaseEstimator):
    """Create the same fixed-category OHE representation for LR, RF and XGBoost.

    Input is the 16-column output of ``DatasetBNumericTwoChannel``. Every
    schema-valid category has a column even when absent from fold-train.
    Unknown codes fail validation rather than silently becoming all-zero.
    ``OBSERVED`` is the reference category for each numeric status channel;
    ``NA``, ``Exempt`` and ``MISSING`` remain explicit where schema-valid.
    Output is CSR with four numeric value columns followed by OHE columns.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> DatasetBOneHotEncoder:
        self._validate_input(X)
        self.encoder_ = OneHotEncoder(
            categories=[list(_CATEGORIES[column]) for column in _CATEGORICAL_COLUMNS],
            handle_unknown="error",
            sparse_output=True,
            dtype=np.float64,
        ).fit(X.loc[:, _CATEGORICAL_COLUMNS])

        names = list(_NUMERIC_VALUE_COLUMNS)
        sources = list(NUMERIC_COLUMNS)
        categories: list[str | None] = [None] * len(NUMERIC_COLUMNS)
        encoded_keep_indices: list[int] = []
        encoded_position = 0
        for column in _CATEGORICAL_COLUMNS:
            raw_source = column.removesuffix("_status") if column in _STATUS_CATEGORIES else column
            for position, category in enumerate(_CATEGORIES[column]):
                if not (column in _STATUS_CATEGORIES and category == _STATUS_REFERENCE_CATEGORY):
                    encoded_keep_indices.append(encoded_position)
                    names.append(f"{column.replace('-', '_')}__{position:02d}")
                    sources.append(raw_source)
                    categories.append(category)
                encoded_position += 1
        if len(names) != ENCODED_FEATURE_COUNT:
            raise RuntimeError("Dataset B encoded feature contract must contain 66 columns.")
        self.encoded_keep_indices_ = tuple(encoded_keep_indices)
        self.feature_names_out_ = tuple(names)
        self.feature_sources_ = tuple(sources)
        self.feature_categories_ = tuple(categories)
        _LOGGER.info(
            "Fitted Dataset B OHE rows=%d numeric_columns=%d categorical_inputs=%d output_columns=%d",
            len(X), len(_NUMERIC_VALUE_COLUMNS), len(_CATEGORICAL_COLUMNS), len(names),
        )
        return self

    def transform(self, X: pd.DataFrame) -> csr_matrix:
        check_is_fitted(self, "encoder_")
        self._validate_input(X)
        numeric = csr_matrix(X.loc[:, _NUMERIC_VALUE_COLUMNS].to_numpy(dtype=np.float64))
        categorical = self.encoder_.transform(X.loc[:, _CATEGORICAL_COLUMNS])
        categorical = categorical[:, self.encoded_keep_indices_]
        result = hstack((numeric, categorical), format="csr", dtype=np.float64)
        _LOGGER.info("Transformed Dataset B OHE rows=%d columns=%d", *result.shape)
        return result

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        check_is_fitted(self, "encoder_")
        if input_features is not None and tuple(input_features) != _TWO_CHANNEL_COLUMNS:
            raise ValueError("Input feature names differ from Dataset B two-channel columns.")
        return np.asarray(self.feature_names_out_, dtype=object)

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame) or tuple(X.columns) != _TWO_CHANNEL_COLUMNS:
            raise ValueError("Dataset B OHE input must contain the 16 two-channel columns in order.")
        numeric = X.loc[:, _NUMERIC_VALUE_COLUMNS]
        try:
            values = numeric.to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Dataset B numeric value channels must contain finite numbers.") from exc
        if not np.isfinite(values).all():
            raise ValueError("Dataset B numeric value channels must contain finite numbers.")
        for column in _CATEGORICAL_COLUMNS:
            if not X[column].isin(_CATEGORIES[column]).all():
                raise ValueError(f"Unexpected Dataset B category in {column}.")
