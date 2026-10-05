"""Stable reporting helpers for Dataset B model notebooks."""

from .model_reporting import (
    EXPECTED_REPORTING_FILES,
    FINAL_MODEL_KEYS,
    MODEL_COLORS,
    REPORTING_CSV_COLUMNS,
    ReportingContractError,
    build_lr07_reporting_summary,
    cv_fold_frame,
    plot_cv_roc_auc,
    plot_calibration_curve,
    plot_threshold_balanced_accuracy,
    validate_cross_model_summaries,
    validate_model_summary,
    validate_reporting_directory,
    write_lr_reporting_package,
)
from .realmlp_reporting import write_realmlp_reporting

__all__ = [
    "EXPECTED_REPORTING_FILES",
    "FINAL_MODEL_KEYS",
    "MODEL_COLORS",
    "REPORTING_CSV_COLUMNS",
    "ReportingContractError",
    "build_lr07_reporting_summary",
    "cv_fold_frame",
    "plot_cv_roc_auc",
    "plot_calibration_curve",
    "plot_threshold_balanced_accuracy",
    "validate_cross_model_summaries",
    "validate_model_summary",
    "validate_reporting_directory",
    "write_lr_reporting_package",
    "write_realmlp_reporting",
]
