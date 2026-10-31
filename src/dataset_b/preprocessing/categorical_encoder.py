"""Compatibility exports for canonical Dataset B categorical encoding."""

from src.dataset_b.data.categorical_encoder import (
    CATEGORICAL_COLUMNS,
    CATEGORICAL_SCHEMA,
    ENCODED_FEATURE_COUNT,
    STATUS_REFERENCE_CATEGORIES,
    DatasetBOneHotEncoder,
)
from src.dataset_b.data.numeric_two_channel import (
    NUMERIC_VALUE_COLUMNS,
    TWO_CHANNEL_COLUMNS,
)

__all__ = [
    "CATEGORICAL_COLUMNS",
    "CATEGORICAL_SCHEMA",
    "DatasetBOneHotEncoder",
    "ENCODED_FEATURE_COUNT",
    "NUMERIC_VALUE_COLUMNS",
    "STATUS_REFERENCE_CATEGORIES",
    "TWO_CHANNEL_COLUMNS",
]
