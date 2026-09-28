"""Contract tests for production Dataset B RealMLP five-fold CV code."""

import numpy as np
import pytest

from src.dataset_b.runtime import realmlp_cv
from src.dataset_b.runtime.realmlp_cv import (
    RealMLPCVError,
    _metric_summary,
    _order_and_validate_oof_rows,
    _validate_published_oof_order,
    _validate_running_metrics,
)


def _row(record_id, label, probability, fold):
    return {
        "record_id": record_id,
        "y_true": str(label),
        "probability_0": repr(1.0 - probability),
        "probability_1": repr(probability),
        "fold": str(fold),
    }


def test_oof_rows_are_reordered_to_immutable_train_order(monkeypatch):
    monkeypatch.setattr(realmlp_cv, "EXPECTED_TRAIN_ROWS", 3)
    rows = [_row("r3", 1, 0.9, 3), _row("r1", 0, 0.1, 1), _row("r2", 1, 0.8, 2)]

    ordered = _order_and_validate_oof_rows(
        ("r1", "r2", "r3"), rows, expected_labels={"r1": 0, "r2": 1, "r3": 1}
    )

    assert [row["record_id"] for row in ordered] == ["r1", "r2", "r3"]


def test_oof_rows_reject_duplicate_or_incomplete_membership(monkeypatch):
    monkeypatch.setattr(realmlp_cv, "EXPECTED_TRAIN_ROWS", 2)
    rows = [_row("r1", 0, 0.1, 1), _row("r1", 0, 0.2, 2)]

    with pytest.raises(RealMLPCVError, match="duplicate"):
        _order_and_validate_oof_rows(
            ("r1", "r2"), rows, expected_labels={"r1": 0, "r2": 1}
        )


def test_oof_rows_reject_label_or_probability_contract_violation(monkeypatch):
    monkeypatch.setattr(realmlp_cv, "EXPECTED_TRAIN_ROWS", 2)
    wrong_label = [_row("r1", 1, 0.1, 1), _row("r2", 1, 0.8, 2)]
    with pytest.raises(RealMLPCVError, match="label"):
        _order_and_validate_oof_rows(
            ("r1", "r2"), wrong_label, expected_labels={"r1": 0, "r2": 1}
        )

    invalid_probability = [_row("r1", 0, 0.1, 1), _row("r2", 1, 1.2, 2)]
    with pytest.raises(RealMLPCVError, match="probabilities"):
        _order_and_validate_oof_rows(
            ("r1", "r2"),
            invalid_probability,
            expected_labels={"r1": 0, "r2": 1},
        )


def test_metric_summary_uses_population_std_and_oof_auc():
    rows = [_row("r1", 0, 0.1, 1), _row("r2", 1, 0.9, 2)]
    fold_results = [
        {"metrics": {"roc_auc": value}}
        for value in (0.80, 0.82, 0.84, 0.86, 0.88)
    ]

    metrics = _metric_summary(rows, fold_results)

    assert metrics["per_fold_roc_auc"] == [0.80, 0.82, 0.84, 0.86, 0.88]
    assert metrics["mean_roc_auc"] == pytest.approx(0.84)
    assert metrics["std_ddof"] == 0
    assert metrics["oof_roc_auc"] == 1.0


def test_published_oof_requires_source_order_and_positions():
    rows = [
        {"source_position": "0", **_row("r1", 0, 0.1, 1)},
        {"source_position": "1", **_row("r2", 1, 0.9, 2)},
    ]
    _validate_published_oof_order(rows, ("r1", "r2"))

    with pytest.raises(RealMLPCVError, match="source order"):
        _validate_published_oof_order(rows[::-1], ("r1", "r2"))


def test_running_metrics_are_recomputed_for_every_fold():
    aucs = [0.80, 0.82, 0.84, 0.86, 0.88]
    results = []
    for index, auc in enumerate(aucs, start=1):
        prefix = aucs[:index]
        results.append(
            {
                "metrics": {
                    "roc_auc": auc,
                    "running_completed_folds": index,
                    "running_mean_roc_auc": sum(prefix) / len(prefix),
                    "running_std_roc_auc": float(np.std(prefix, ddof=0)),
                    "running_std_ddof": 0,
                }
            }
        )

    _validate_running_metrics(results)
