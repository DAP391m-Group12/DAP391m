from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score

from src.dataset_b.fairness.metrics import DenominatorRules, compute_global_race_ranges
from src.dataset_b.mitigation.inputs import LOCKED_G3, LOCKED_G6
from src.dataset_b.mitigation.metrics import (
    EOD_INVALID_REASON,
    GROUP_ABSENT_STATUS,
    balanced_accuracy,
    ba_cost,
    compute_fixed_roster_eod,
    compute_roster_group_metrics,
    decision_changes,
    delta_eod,
    denied_metrics,
    extremum_roles,
)


TOY_RULES = DenominatorRules(minimum_group_rows=4, minimum_actual_approved=2, minimum_actual_denied=2)


def _toy_rows(spec: dict[str, tuple[list[int], list[int]]]):
    """spec: group -> (y, decisions); p is a constant valid probability."""
    y, d, g = [], [], []
    for group, (ys, ds) in spec.items():
        y += ys
        d += ds
        g += [group] * len(ys)
    return np.array(y), np.full(len(y), 0.5), np.array(d), np.array(g)


G3_SPEC = {
    "White": ([1, 1, 0, 0], [1, 1, 0, 0]),
    "Black or African American": ([1, 1, 0, 0], [1, 0, 1, 0]),
    "Asian": ([1, 1, 0, 0], [1, 1, 1, 0]),
}


def test_roster_group_metrics_cover_exact_roster_in_order() -> None:
    y, p, d, g = _toy_rows(G3_SPEC)
    metrics = compute_roster_group_metrics(y, p, d, g, LOCKED_G3, rules=TOY_RULES)
    assert tuple(metrics) == LOCKED_G3
    six = compute_roster_group_metrics(y, p, d, g, LOCKED_G6, rules=TOY_RULES)
    assert tuple(six) == LOCKED_G6
    assert six["Joint"]["tpr_approved"].status == GROUP_ABSENT_STATUS
    assert not six["Joint"]["tpr_approved"].is_defined


def test_fixed_roster_eod_matches_hand_calculation() -> None:
    y, p, d, g = _toy_rows(G3_SPEC)
    eod = compute_fixed_roster_eod(compute_roster_group_metrics(y, p, d, g, LOCKED_G3, rules=TOY_RULES), LOCKED_G3)
    # TPR: 1.0, 0.5, 1.0 -> range 0.5; FPR: 0.0, 0.5, 0.5 -> range 0.5
    assert eod.valid and eod.reason_code == "none"
    assert eod.tpr_range == pytest.approx(0.5)
    assert eod.fpr_range == pytest.approx(0.5)
    assert eod.value == pytest.approx(0.5)


def test_one_required_undefined_group_invalidates_whole_eod() -> None:
    spec = dict(G3_SPEC)
    spec["Asian"] = ([1, 0, 0, 0], [1, 1, 0, 0])  # one actual Approved < minimum 2 -> TPR undefined
    y, p, d, g = _toy_rows(spec)
    metrics = compute_roster_group_metrics(y, p, d, g, LOCKED_G3, rules=TOY_RULES)
    eod = compute_fixed_roster_eod(metrics, LOCKED_G3)
    assert not eod.valid
    assert eod.value is None and eod.tpr_range is None and eod.fpr_range is None
    assert eod.invalid_groups == ("Asian",)
    assert eod.reason_code == EOD_INVALID_REASON
    # RQ3 global ranges would silently shrink to the defined groups; RQ4a must not.
    rq3 = compute_global_race_ranges(metrics)
    assert rq3["tpr_approved_range"].is_defined and "Asian" not in rq3["tpr_approved_range"].groups_used


def test_absent_roster_group_invalidates_g6_eod() -> None:
    y, p, d, g = _toy_rows(G3_SPEC)
    eod6 = compute_fixed_roster_eod(compute_roster_group_metrics(y, p, d, g, LOCKED_G6, rules=TOY_RULES), LOCKED_G6)
    assert not eod6.valid
    assert eod6.invalid_groups == LOCKED_G6[3:]


def test_no_dynamic_shrink_extra_or_reordered_groups() -> None:
    y, p, d, g = _toy_rows(G3_SPEC)
    metrics = compute_roster_group_metrics(y, p, d, g, LOCKED_G3, rules=TOY_RULES)
    shrunk = {k: v for k, v in metrics.items() if k != "Asian"}
    with pytest.raises(ValueError):
        compute_fixed_roster_eod(shrunk, LOCKED_G3)
    with pytest.raises(ValueError):
        compute_fixed_roster_eod(metrics, LOCKED_G3[:2])
    reordered = {k: metrics[k] for k in reversed(LOCKED_G3)}
    with pytest.raises(ValueError):
        compute_fixed_roster_eod(reordered, LOCKED_G3)
    with pytest.raises(ValueError):
        compute_roster_group_metrics(y, p, d, g, ("White", "White"), rules=TOY_RULES)


def test_full_partition_balanced_accuracy_matches_sklearn() -> None:
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 300)
    d = rng.integers(0, 2, 300)
    assert balanced_accuracy(y, d) == pytest.approx(balanced_accuracy_score(y, d))
    assert balanced_accuracy([1, 1, 1], [1, 0, 1]) is None


def test_denied_metrics_use_denied_as_positive_class() -> None:
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 300)
    d = rng.integers(0, 2, 300)
    out = denied_metrics(y, d)
    assert out["denied_precision"] == pytest.approx(precision_score(y, d, pos_label=0))
    assert out["denied_recall"] == pytest.approx(recall_score(y, d, pos_label=0))
    assert out["denied_f1"] == pytest.approx(f1_score(y, d, pos_label=0))
    # No predicted Denied: precision None, recall 0, F1 = 2*0/(1+0) = 0.
    no_predicted = denied_metrics([1, 1, 0], [1, 1, 1])
    assert no_predicted["denied_precision"] is None
    assert no_predicted["denied_recall"] == 0.0 and no_predicted["denied_f1"] == 0.0
    # Positive denominator with true_denied = 0: F1 is 0, not None.
    zero = denied_metrics([1, 0], [0, 1])
    assert (zero["denied_precision"], zero["denied_recall"], zero["denied_f1"]) == (0.0, 0.0, 0.0)
    # No actual and no predicted Denied: recall and F1 None (denominator 0).
    none = denied_metrics([1, 1], [1, 1])
    assert none["denied_precision"] is None and none["denied_recall"] is None and none["denied_f1"] is None
    # No actual Denied but predicted Denied: recall None, F1 = 2*0/(0+1) = 0.
    no_actual = denied_metrics([1, 1], [0, 1])
    assert no_actual["denied_recall"] is None and no_actual["denied_f1"] == 0.0 and no_actual["denied_precision"] == 0.0


def test_decision_changes_overall_and_by_group() -> None:
    baseline = np.array([1, 1, 0, 0, 1])
    mitigated = np.array([1, 0, 1, 0, 0])
    groups = np.array(["White", "White", "Asian", "Asian", "Joint"])
    rows = decision_changes(baseline, mitigated, groups, ["White", "Asian", "Joint"])
    assert rows[0] == {"scope": "overall", "group": None, "n_rows": 5, "n_changed": 3, "change_rate": 0.6,
                       "approved_to_denied": 2, "denied_to_approved": 1}
    assert [r["group"] for r in rows[1:]] == ["White", "Asian", "Joint"]
    assert rows[1]["approved_to_denied"] == 1 and rows[2]["denied_to_approved"] == 1
    with pytest.raises(ValueError):
        decision_changes(baseline, mitigated, groups, ["White", "Asian"])
    with_empty = decision_changes(baseline, mitigated, groups, ["White", "Asian", "Joint", "Black or African American"])
    empty = with_empty[-1]
    assert empty["group"] == "Black or African American" and empty["n_rows"] == 0
    assert empty["change_rate"] is None and empty["n_changed"] == 0


def test_extremum_roles_per_metric() -> None:
    spec = dict(G3_SPEC)
    y, p, d, g = _toy_rows(spec)
    metrics = compute_roster_group_metrics(y, p, d, g, LOCKED_G6, rules=TOY_RULES)
    tpr_roles = extremum_roles(metrics, "tpr_approved")
    assert tpr_roles["White"] == "max" and tpr_roles["Asian"] == "max"
    assert tpr_roles["Black or African American"] == "min"
    assert tpr_roles["Joint"] == "not_evaluable"
    fpr_roles = extremum_roles(metrics, "fpr_approved")
    assert fpr_roles["White"] == "min"
    assert set(tpr_roles.values()) <= {"min", "max", "none", "not_evaluable"}
    equal = compute_roster_group_metrics(
        *_toy_rows({"White": ([1, 1, 0, 0], [1, 1, 0, 0]), "Asian": ([1, 1, 0, 0], [1, 1, 0, 0])})[:3],
        _toy_rows({"White": ([1, 1, 0, 0], [1, 1, 0, 0]), "Asian": ([1, 1, 0, 0], [1, 1, 0, 0])})[3],
        ("White", "Asian"), rules=TOY_RULES,
    )
    assert set(extremum_roles(equal, "tpr_approved").values()) == {"none"}


def test_paired_delta_signs() -> None:
    y, p, d, g = _toy_rows(G3_SPEC)
    baseline = compute_fixed_roster_eod(compute_roster_group_metrics(y, p, d, g, LOCKED_G3, rules=TOY_RULES), LOCKED_G3)
    fair = compute_fixed_roster_eod(compute_roster_group_metrics(y, p, y, g, LOCKED_G3, rules=TOY_RULES), LOCKED_G3)
    assert fair.value == 0.0
    assert delta_eod(baseline, fair) == pytest.approx(-0.5)  # mitigated - baseline
    invalid = compute_fixed_roster_eod(compute_roster_group_metrics(y, p, d, g, LOCKED_G6, rules=TOY_RULES), LOCKED_G6)
    assert delta_eod(baseline, invalid) is None
    assert ba_cost(0.80, 0.79) == pytest.approx(0.01)  # baseline - mitigated
    assert ba_cost(None, 0.79) is None
