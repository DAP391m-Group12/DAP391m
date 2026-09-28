"""Kiểm tra chỉ số và ranh giới đánh giá Test của RealMLP."""

import numpy as np

from src.dataset_b.evaluation.realmlp_test_evaluation import (
    LOCKED_THRESHOLD,
    _calibration_curve,
    _metrics,
)


def test_metrics_use_locked_threshold_and_report_approved_denied_counts():
    y = np.asarray([1, 1, 0, 0], dtype=np.int8)
    p = np.asarray([0.9, 0.7, 0.8, 0.2], dtype=np.float64)

    values, invalid, counts = _metrics(y, p, LOCKED_THRESHOLD)

    assert invalid == {}
    assert counts == {
        "tp_approved": 1,
        "fn_approved": 1,
        "tn_denied": 1,
        "fp_denied": 1,
        "n_approved": 2,
        "n_denied": 2,
    }
    assert values["balanced_accuracy"] == 0.5
    assert values["recall_approved"] == 0.5
    assert values["recall_denied"] == 0.5


def test_calibration_curve_collapses_duplicate_quantile_edges_and_covers_rows():
    y = np.asarray([0, 1, 0, 1, 1], dtype=np.int8)
    p = np.asarray([0.1, 0.1, 0.5, 0.9, 0.9], dtype=np.float64)

    bins = _calibration_curve(y, p)

    assert sum(row["count"] for row in bins) == len(y)
    assert all(row["count"] > 0 for row in bins)


def test_metric_reports_undefined_denied_precision_instead_of_inventing_value():
    y = np.asarray([1, 1, 0, 0], dtype=np.int8)
    p = np.asarray([0.9, 0.8, 0.8, 0.8], dtype=np.float64)

    values, invalid, _ = _metrics(y, p, LOCKED_THRESHOLD)

    assert np.isnan(values["precision_denied"])
    assert invalid["precision_denied"] == "no_predicted_denied"
