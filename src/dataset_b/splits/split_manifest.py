"""Compatibility imports for the canonical Dataset B split contract."""

from src.dataset_b.data.split_manifest import (
    DatasetBSplitter,
    SplitManifest,
    SplitManifestError,
)

__all__ = ["DatasetBSplitter", "SplitManifest", "SplitManifestError"]
