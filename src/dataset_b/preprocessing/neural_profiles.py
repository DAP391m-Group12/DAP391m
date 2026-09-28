"""Sklearn-compatible Dataset B preprocessing profile for plain MLP."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.pipeline import Pipeline

from .categorical_encoder import DatasetBOneHotEncoder
from .neural_log_transformer import DatasetBNeuralLog1pTransformer
from .numeric_two_channel import DatasetBNumericTwoChannel
from .profiles import DatasetBRawParserStep


_LOGGER = logging.getLogger(__name__)


def make_plain_mlp_preprocessing_pipeline() -> Pipeline:
    """Build a fresh fold-local parser -> two-channel -> log1p -> OHE profile."""
    _LOGGER.info("Created Dataset B plain-MLP preprocessing profile dtype=float32 scaler=none")
    return Pipeline(
        [
            ("parse", DatasetBRawParserStep()),
            ("two_channel", DatasetBNumericTwoChannel()),
            ("log1p_numeric", DatasetBNeuralLog1pTransformer()),
            ("encode", DatasetBOneHotEncoder(dtype=np.float32)),
        ]
    )
