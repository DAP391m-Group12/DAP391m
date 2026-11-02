"""Shared Dataset B cohort and split contracts."""

from .categorical_encoder import (
    CATEGORICAL_COLUMNS,
    CATEGORICAL_SCHEMA,
    ENCODED_FEATURE_COUNT,
    STATUS_REFERENCE_CATEGORIES,
    DatasetBOneHotEncoder,
)
from .dataset_view import DatasetBView, DatasetBViewBuilder, DatasetBViewError
from .median_imputer import DatasetBNumericMedianImputer, NUMERIC_COLUMNS
from .numeric_two_channel import (
    NUMERIC_VALUE_COLUMNS,
    TWO_CHANNEL_COLUMNS,
    DatasetBNumericTwoChannel,
)
from .numeric_value_scaler import DatasetBNumericValueRobustScaler
from .raw_parser import (
    EXEMPTION_FLAG_COLUMNS,
    FEATURE_COLUMNS,
    DatasetBRawParser,
    ParseDiagnostics,
    ParsedFeatures,
    RawFeatureError,
)
from .split_manifest import DatasetBSplitter, SplitManifest, SplitManifestError
from .split_inputs import (
    DatasetBInputQualityAudit,
    DatasetBInputQualityGate,
    DatasetBPartition,
    DatasetBQualityGatedPartition,
    DatasetBSplitLoader,
    InputQualityError,
    SplitInputError,
)

__all__ = [
    "CATEGORICAL_COLUMNS",
    "CATEGORICAL_SCHEMA",
    "DatasetBInputQualityAudit",
    "DatasetBInputQualityGate",
    "DatasetBNumericMedianImputer",
    "DatasetBNumericTwoChannel",
    "DatasetBNumericValueRobustScaler",
    "DatasetBOneHotEncoder",
    "DatasetBPartition",
    "DatasetBQualityGatedPartition",
    "DatasetBSplitter",
    "DatasetBSplitLoader",
    "DatasetBView",
    "DatasetBViewBuilder",
    "DatasetBViewError",
    "EXEMPTION_FLAG_COLUMNS",
    "ENCODED_FEATURE_COUNT",
    "FEATURE_COLUMNS",
    "InputQualityError",
    "NUMERIC_COLUMNS",
    "NUMERIC_VALUE_COLUMNS",
    "ParseDiagnostics",
    "ParsedFeatures",
    "RawFeatureError",
    "SplitInputError",
    "SplitManifest",
    "SplitManifestError",
    "STATUS_REFERENCE_CATEGORIES",
    "TWO_CHANNEL_COLUMNS",
]
