"""Compatibility exports for the canonical Dataset B semantic contract."""

from src.dataset_b.core.feature_contract import (
    DEFAULT_FEATURE_CONTRACT_PATH,
    DEFAULT_SPLIT_DIR,
    DEFAULT_SPLIT_MANIFEST_PATH,
    EXPECTED_AUDIT_COLUMNS,
    EXPECTED_MODEL_INPUTS,
    EXPECTED_PARTITION_ROWS,
    LOCKED_SPLIT_MANIFEST_SHA256,
    DatasetBSemanticValidationError,
    FeatureContractValidation,
    PartitionDataQuality,
    SplitDataQualityValidation,
    validate_dataset_b_semantics,
    validate_feature_contract,
    validate_split_data_quality,
)

__all__ = [
    "DEFAULT_FEATURE_CONTRACT_PATH",
    "DEFAULT_SPLIT_DIR",
    "DEFAULT_SPLIT_MANIFEST_PATH",
    "DatasetBSemanticValidationError",
    "EXPECTED_AUDIT_COLUMNS",
    "EXPECTED_MODEL_INPUTS",
    "EXPECTED_PARTITION_ROWS",
    "FeatureContractValidation",
    "LOCKED_SPLIT_MANIFEST_SHA256",
    "PartitionDataQuality",
    "SplitDataQualityValidation",
    "validate_dataset_b_semantics",
    "validate_feature_contract",
    "validate_split_data_quality",
]
