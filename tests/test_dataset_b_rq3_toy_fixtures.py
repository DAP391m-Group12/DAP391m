from __future__ import annotations

import numpy as np
import pytest

from src.dataset_b.fairness.metrics import (
    DenominatorRules,
    compute_global_race_ranges,
    compute_group_metrics,
    compute_pairwise_gaps,
)


TOY_RULES = DenominatorRules(
    minimum_group_rows=4,
    minimum_actual_approved=2,
    minimum_actual_denied=2,
)


def _group(y: list[int], p: list[float], predicted: list[int]):
    return compute_group_metrics(y, p, predicted, rules=TOY_RULES)


def test_approved_positive_group_metrics_match_hand_calculation() -> None:
    metrics = _group(
        y=[1, 1, 0, 0],
        p=[0.9, 0.4, 0.6, 0.1],
        predicted=[1, 0, 1, 0],
    )
    assert metrics["selection_rate"].value == pytest.approx(0.5)
    assert metrics["tpr_approved"].value == pytest.approx(0.5)
    assert metrics["fnr_approved"].value == pytest.approx(0.5)
    assert metrics["fpr_approved"].value == pytest.approx(0.5)
    assert metrics["brier_score"].value == pytest.approx(0.185)
    assert metrics["calibration_residual"].value == pytest.approx(0.0)
    assert metrics["absolute_calibration_residual"].value == pytest.approx(0.0)


def test_pairwise_gaps_are_target_minus_reference_and_keep_direction() -> None:
    male = _group([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    female = _group([1, 1, 0, 0], [0.8, 0.3, 0.7, 0.2], [1, 0, 1, 0])
    gaps = compute_pairwise_gaps(female, male)
    assert gaps["selection_rate_gap_signed"].value == pytest.approx(0.0)
    assert gaps["tpr_approved_gap_signed"].value == pytest.approx(-0.5)
    assert gaps["tpr_approved_gap_absolute"].value == pytest.approx(0.5)
    assert gaps["fpr_approved_gap_signed"].value == pytest.approx(0.5)
    assert gaps["fnr_approved_gap_signed"].value == pytest.approx(0.5)
    assert gaps["equalized_odds_gap"].value == pytest.approx(0.5)


def test_calibration_gap_absolute_is_absolute_signed_gap() -> None:
    reference = _group([1, 1, 0, 0], [0.9, 0.9, 0.3, 0.3], [1, 1, 0, 0])
    target = _group([1, 1, 0, 0], [0.6, 0.6, 0.1, 0.1], [1, 1, 0, 0])
    gaps = compute_pairwise_gaps(target, reference)
    signed = gaps["calibration_residual_gap_signed"].value
    absolute = gaps["calibration_residual_gap_absolute"].value
    assert signed == pytest.approx(-0.25)
    assert absolute == pytest.approx(abs(signed))


def test_denominator_failures_are_undefined_not_zero() -> None:
    metrics = compute_group_metrics(
        [1, 0, 0],
        [0.8, 0.4, 0.2],
        [1, 0, 0],
        rules=TOY_RULES,
    )
    assert metrics["selection_rate"].value is None
    assert metrics["selection_rate"].status == "undefined_insufficient_group_rows"
    assert metrics["tpr_approved"].value is None
    assert metrics["tpr_approved"].status == "undefined_insufficient_actual_approved"
    assert metrics["fpr_approved"].value == pytest.approx(0.0)
    assert metrics["fpr_approved"].status == "defined"


def test_global_race_ranges_use_only_passed_eligible_groups() -> None:
    white = _group([1, 1, 0, 0], [0.9, 0.8, 0.7, 0.1], [1, 1, 1, 0])
    black = _group([1, 1, 0, 0], [0.8, 0.3, 0.2, 0.1], [1, 0, 0, 0])
    asian = _group([1, 1, 0, 0], [0.9, 0.7, 0.4, 0.3], [1, 1, 0, 0])
    ranges = compute_global_race_ranges({"White": white, "Black": black, "Asian": asian})
    assert ranges["demographic_parity_difference_range"].value == pytest.approx(0.5)
    assert ranges["tpr_approved_range"].value == pytest.approx(0.5)
    assert ranges["fpr_approved_range"].value == pytest.approx(0.5)
    assert ranges["equalized_odds_difference_range"].value == pytest.approx(0.5)
    assert "Race Not Available" not in ranges["tpr_approved_range"].groups_used


def test_invalid_orientation_inputs_fail_loudly() -> None:
    with pytest.raises(ValueError, match="y_approved"):
        _group([1, -1, 0, 0], [0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        _group([1, 1, 0, 0], [0.9, 1.2, 0.2, 0.1], [1, 1, 0, 0])
    with pytest.raises(ValueError, match="predicted_approved"):
        _group([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], [1, 2, 0, 0])


def test_global_range_is_undefined_with_fewer_than_two_defined_groups() -> None:
    defined = _group([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    undefined = compute_group_metrics(
        [1, 0], [0.8, 0.2], [1, 0], rules=TOY_RULES
    )
    ranges = compute_global_race_ranges({"White": defined, "AIAN": undefined})
    assert ranges["demographic_parity_difference_range"].value is None
    assert ranges["demographic_parity_difference_range"].status == (
        "undefined_fewer_than_two_eligible_groups"
    )


def test_group_metric_values_are_finite_when_defined() -> None:
    metrics = _group([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    assert all(
        estimate.value is None or np.isfinite(estimate.value)
        for estimate in metrics.values()
    )
