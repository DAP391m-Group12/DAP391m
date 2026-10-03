"""Fixed-split inputs and fold-local Dataset B preprocessing stages."""

from .categorical_encoder import (
    CATEGORICAL_COLUMNS, CATEGORICAL_SCHEMA, ENCODED_FEATURE_COUNT,
    DatasetBOneHotEncoder,
)
from .median_imputer import DatasetBNumericMedianImputer, NUMERIC_COLUMNS
from .lr_log_transformer import DatasetBLRSelectiveLog1pTransformer, LOG1P_VALUE_COLUMNS
from .neural_log_transformer import DatasetBNeuralLog1pTransformer
from .neural_profiles import make_plain_mlp_preprocessing_pipeline
from .numeric_two_channel import DatasetBNumericTwoChannel, NUMERIC_VALUE_COLUMNS, TWO_CHANNEL_COLUMNS
from .profiles import DatasetBNumericValueRobustScaler, DatasetBRawParserStep, make_preprocessing_pipeline
from .raw_parser import (
    EXEMPTION_FLAG_COLUMNS,
    DatasetBRawParser,
    ParseDiagnostics,
    ParsedFeatures,
    RawFeatureError,
)
from .realmlp_native import (
    DatasetBRealMLPNativeAdapter,
    RealMLPCategorySupportError,
    make_realmlp_native_preprocessing_pipeline,
    validate_realmlp_category_support,
)
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
    "DatasetBPartition",
    "DatasetBQualityGatedPartition",
    "DatasetBInputQualityAudit",
    "DatasetBInputQualityGate",
    "DatasetBSplitLoader",
    "DatasetBRawParser",
    "DatasetBNumericMedianImputer",
    "DatasetBNumericTwoChannel",
    "DatasetBLRSelectiveLog1pTransformer",
    "DatasetBOneHotEncoder",
    "DatasetBNeuralLog1pTransformer",
    "DatasetBRealMLPNativeAdapter",
    "DatasetBNumericValueRobustScaler",
    "DatasetBRawParserStep",
    "make_preprocessing_pipeline",
    "make_plain_mlp_preprocessing_pipeline",
    "make_realmlp_native_preprocessing_pipeline",
    "validate_realmlp_category_support",
    "CATEGORICAL_COLUMNS",
    "CATEGORICAL_SCHEMA",
    "EXEMPTION_FLAG_COLUMNS",
    "ENCODED_FEATURE_COUNT",
    "NUMERIC_COLUMNS",
    "NUMERIC_VALUE_COLUMNS",
    "TWO_CHANNEL_COLUMNS",
    "LOG1P_VALUE_COLUMNS",
    "ParseDiagnostics",
    "ParsedFeatures",
    "RawFeatureError",
    "RealMLPCategorySupportError",
    "InputQualityError",
    "SplitInputError",
]
