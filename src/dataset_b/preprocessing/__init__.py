"""Fixed-split inputs and lazily loaded Dataset B preprocessing stages."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .categorical_encoder import (
    CATEGORICAL_COLUMNS, CATEGORICAL_SCHEMA, ENCODED_FEATURE_COUNT,
    DatasetBOneHotEncoder,
)
from .median_imputer import DatasetBNumericMedianImputer, NUMERIC_COLUMNS
from .lr_log_transformer import DatasetBLRSelectiveLog1pTransformer, LOG1P_VALUE_COLUMNS
from .numeric_two_channel import DatasetBNumericTwoChannel, NUMERIC_VALUE_COLUMNS, TWO_CHANNEL_COLUMNS
from .profiles import DatasetBNumericValueRobustScaler, DatasetBRawParserStep, make_preprocessing_pipeline
from .raw_parser import (
    EXEMPTION_FLAG_COLUMNS,
    DatasetBRawParser,
    ParseDiagnostics,
    ParsedFeatures,
    RawFeatureError,
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
    "DatasetBTabMMiniNativeAdapter",
    "DatasetBNumericValueRobustScaler",
    "DatasetBRawParserStep",
    "make_preprocessing_pipeline",
    "make_plain_mlp_preprocessing_pipeline",
    "make_realmlp_native_preprocessing_pipeline",
    "make_tabmmini_native_preprocessing_pipeline",
    "validate_realmlp_category_support",
    "validate_tabmmini_category_support",
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
    "TabMMiniCategorySupportError",
    "TabMMiniNativeBatch",
    "fixed_tabmmini_category_maps",
    "tabmmini_category_cardinalities",
    "InputQualityError",
    "SplitInputError",
]

_LAZY_EXPORTS = {
    "DatasetBNeuralLog1pTransformer": "src.dataset_b.models.neural.common.log_transformer",
    "make_plain_mlp_preprocessing_pipeline": "src.dataset_b.models.neural.plain_mlp.preprocessing",
    "DatasetBRealMLPNativeAdapter": "src.dataset_b.models.neural.realmlp.preprocessing",
    "RealMLPCategorySupportError": "src.dataset_b.models.neural.realmlp.preprocessing",
    "make_realmlp_native_preprocessing_pipeline": "src.dataset_b.models.neural.realmlp.preprocessing",
    "validate_realmlp_category_support": "src.dataset_b.models.neural.realmlp.preprocessing",
    "DatasetBTabMMiniNativeAdapter": "src.dataset_b.models.neural.tabmmini.preprocessing",
    "TabMMiniCategorySupportError": "src.dataset_b.models.neural.tabmmini.preprocessing",
    "TabMMiniNativeBatch": "src.dataset_b.models.neural.tabmmini.preprocessing",
    "fixed_tabmmini_category_maps": "src.dataset_b.models.neural.tabmmini.preprocessing",
    "make_tabmmini_native_preprocessing_pipeline": "src.dataset_b.models.neural.tabmmini.preprocessing",
    "tabmmini_category_cardinalities": "src.dataset_b.models.neural.tabmmini.preprocessing",
    "validate_tabmmini_category_support": "src.dataset_b.models.neural.tabmmini.preprocessing",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
