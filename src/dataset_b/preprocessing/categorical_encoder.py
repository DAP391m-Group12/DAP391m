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
from .numeric_two_channel import NUMERIC_VALUE_COLUMNS, TWO_CHANNEL_COLUMNS


_LOGGER = logging.getLogger(__name__)

_DTI_CATEGORIES = (
    "<20%", "20%-<30%", "30%-<36%",
    *(str(value) for value in range(36, 50)),
    "50%-60%", ">60%", "NA", "Exempt",
)
_RAW_CATEGORIES = {
    "loan_type": ("1", "2", "3", "4"),
    "loan_purpose": ("1", "2", "31", "32", "4", "5"),
    "lien_status": ("1", "2"),
    "reverse_mortgage": ("1", "2", "Exempt"),
    "open-end_line_of_credit": ("1", "2", "Exempt"),
    "business_or_commercial_purpose": ("1", "2", "Exempt"),
    "debt_to_income_ratio": _DTI_CATEGORIES,
    "occupancy_type": ("1", "2", "3"),
    "preapproval": ("1", "2"),
    "interest_only_payment": ("1", "2", "Exempt"),
    "balloon_payment": ("1", "2", "Exempt"),
}
_STATUS_CATEGORIES = {
    "income_status": ("OBSERVED", "NA"),
    "loan_amount_status": ("OBSERVED",),
    "loan_to_value_ratio_status": ("OBSERVED", "NA", "Exempt"),
    "loan_term_status": ("OBSERVED", "NA", "Exempt"),
}
_CATEGORIES = {**_RAW_CATEGORIES, **_STATUS_CATEGORIES}
_CATEGORICAL_COLUMNS = tuple(column for column in TWO_CHANNEL_COLUMNS if column in _CATEGORIES)
_STATUS_REFERENCE_CATEGORY = "OBSERVED"
STATUS_REFERENCE_CATEGORIES = tuple(
    (f"{column}_status", _STATUS_REFERENCE_CATEGORY)
    for column in NUMERIC_COLUMNS
)
ENCODED_FEATURE_COUNT = 62
CATEGORICAL_SCHEMA = {column: tuple(categories) for column, categories in _CATEGORIES.items()}
CATEGORICAL_COLUMNS = _CATEGORICAL_COLUMNS


class DatasetBOneHotEncoder(TransformerMixin, BaseEstimator):
    """Create the same fixed-category OHE representation for LR, RF and XGBoost.

    Input is the 19-column output of ``DatasetBNumericTwoChannel``. Every
    schema-valid category has a column even when absent from fold-train.
    Unknown codes fail validation rather than silently becoming all-zero.
    ``OBSERVED`` is the reference category for each numeric status channel;
    ``NA`` and ``Exempt`` remain explicit where schema-valid.
    Output is CSR with four numeric value columns followed by OHE columns.
    The default remains ``float64`` for classical backward compatibility;
    neural profiles must request ``float32`` explicitly.
    """

    def __init__(self, *, dtype: object = np.float64) -> None:
        self.dtype = dtype

    def fit(self, X: pd.DataFrame, y: object = None) -> DatasetBOneHotEncoder:
        self._validate_input(X)
        output_dtype = np.dtype(self.dtype)
        if output_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError("Dataset B OHE dtype must be float32 or float64.")
        self.encoder_ = OneHotEncoder(
            categories=[list(_CATEGORIES[column]) for column in _CATEGORICAL_COLUMNS],
            handle_unknown="error",
            sparse_output=True,
            dtype=output_dtype,
        ).fit(X.loc[:, _CATEGORICAL_COLUMNS])

        names = list(NUMERIC_VALUE_COLUMNS)
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
            raise RuntimeError("Dataset B encoded feature contract must contain 62 columns.")
        self.encoded_keep_indices_ = tuple(encoded_keep_indices)
        self.feature_names_out_ = tuple(names)
        self.feature_sources_ = tuple(sources)
        self.feature_categories_ = tuple(categories)
        self.output_dtype_ = output_dtype
        _LOGGER.info(
            "Fitted Dataset B OHE rows=%d numeric_columns=%d categorical_inputs=%d output_columns=%d",
            len(X), len(NUMERIC_VALUE_COLUMNS), len(_CATEGORICAL_COLUMNS), len(names),
        )
        return self

    def transform(self, X: pd.DataFrame) -> csr_matrix:
        check_is_fitted(self, "encoder_")
        self._validate_input(X)
        numeric = csr_matrix(X.loc[:, NUMERIC_VALUE_COLUMNS].to_numpy(dtype=self.output_dtype_))
        categorical = self.encoder_.transform(X.loc[:, _CATEGORICAL_COLUMNS])
        categorical = categorical[:, self.encoded_keep_indices_]
        result = hstack((numeric, categorical), format="csr", dtype=self.output_dtype_)
        _LOGGER.info("Transformed Dataset B OHE rows=%d columns=%d", *result.shape)
        return result

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        check_is_fitted(self, "encoder_")
        if input_features is not None and tuple(input_features) != TWO_CHANNEL_COLUMNS:
            raise ValueError("Input feature names differ from Dataset B two-channel columns.")
        return np.asarray(self.feature_names_out_, dtype=object)

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame) or tuple(X.columns) != TWO_CHANNEL_COLUMNS:
            raise ValueError("Dataset B OHE input must contain the 19 two-channel columns in order.")
        numeric = X.loc[:, NUMERIC_VALUE_COLUMNS]
        try:
            values = numeric.to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Dataset B numeric value channels must contain finite numbers.") from exc
        if not np.isfinite(values).all():
            raise ValueError("Dataset B numeric value channels must contain finite numbers.")
        for column in _CATEGORICAL_COLUMNS:
            if not X[column].isin(_CATEGORIES[column]).all():
                raise ValueError(f"Unexpected Dataset B category in {column}.")
