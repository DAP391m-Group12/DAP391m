"""Sklearn-compatible Dataset B preprocessing profile for plain MLP."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.pipeline import Pipeline

from .categorical_encoder import DatasetBOneHotEncoder
from .neural_log_transformer import DatasetBNeuralLog1pTransformer
from .numeric_two_channel import DatasetBNumericTwoChannel
from .profiles import DatasetBNumericValueRobustScaler


_LOGGER = logging.getLogger(__name__)


def make_plain_mlp_preprocessing_pipeline() -> Pipeline:
    """Build a fresh fit-local pipeline for normalized 15-column input."""
    _LOGGER.info(
        "Created Dataset B PlainMLP preprocessing profile "
        "schema=15/19/62 dtype=float64 scaler=RobustScaler"
    )
    return Pipeline(
        [
            ("two_channel", DatasetBNumericTwoChannel()),
            ("log1p_numeric", DatasetBNeuralLog1pTransformer()),
            ("encode", DatasetBOneHotEncoder(dtype=np.float64)),
            ("scale_numeric", DatasetBNumericValueRobustScaler()),
        ]
    )
