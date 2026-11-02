"""Compatibility imports for canonical Dataset B raw-input semantics."""

from src.dataset_b.data.raw_parser import (
    EXEMPTION_FLAG_COLUMNS,
    FEATURE_COLUMNS,
    DatasetBRawParser,
    ParseDiagnostics,
    ParsedFeatures,
    RawFeatureError,
)

__all__ = [
    "DatasetBRawParser",
    "EXEMPTION_FLAG_COLUMNS",
    "FEATURE_COLUMNS",
    "ParseDiagnostics",
    "ParsedFeatures",
    "RawFeatureError",
]
