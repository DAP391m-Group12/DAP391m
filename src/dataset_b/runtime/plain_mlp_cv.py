"""Public facade for the PMLP-F04 Train-only CV runner."""

from .plain_mlp_training import (
    CV_CONTRACT_ID,
    DEFAULT_CV_OUTPUT_DIR,
    EXPECTED_TRAIN_ROWS,
    PlainMLPCVError,
    run_plain_mlp_cv,
    validate_plain_mlp_cv,
)

__all__ = [
    "CV_CONTRACT_ID",
    "DEFAULT_CV_OUTPUT_DIR",
    "EXPECTED_TRAIN_ROWS",
    "PlainMLPCVError",
    "run_plain_mlp_cv",
    "validate_plain_mlp_cv",
]
