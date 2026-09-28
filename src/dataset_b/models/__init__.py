"""Dataset B estimator wrappers."""

from .plain_mlp import PlainMLPClassifier
from .realmlp import (
    RealMLPClassifier,
    RealMLPConfigurationError,
    load_realmlp_v1_config,
)

__all__ = [
    "PlainMLPClassifier",
    "RealMLPClassifier",
    "RealMLPConfigurationError",
    "load_realmlp_v1_config",
]
