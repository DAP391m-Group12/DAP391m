"""Canonical Pure RQ4a metric definitions on fixed rosters (Approved is the positive class).

Group metrics reuse the accepted RQ3 `compute_group_metrics` and its denominator
gates. RQ3 `compute_global_race_ranges` is deliberately not reused because it
drops undefined groups; RQ4a EOD is invalid instead of shrinking the roster.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from src.dataset_b.research.rq3.metrics import DenominatorRules, MetricEstimate, compute_group_metrics


GROUP_ABSENT_STATUS = "undefined_group_absent"
EOD_INVALID_REASON = "required_group_metric_invalid"
EXTREMUM_ROLES = ("min", "max", "none", "not_evaluable")


@dataclass(frozen=True)
class EODEstimate:
    value: float | None
    valid: bool
    tpr_range: float | None
    fpr_range: float | None
    roster: tuple[str, ...]
    invalid_groups: tuple[str, ...]
    reason_code: str


def _vector(values: Sequence | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    return array


def _binary(values: Sequence | np.ndarray, name: str) -> np.ndarray:
    array = _vector(values, name)
    if not np.isin(array, [0, 1]).all():
        raise ValueError(f"{name} must contain only 0 and 1.")
    return array.astype(np.int8)


def _absent_group_metrics() -> dict[str, MetricEstimate]:
    absent = MetricEstimate(None, GROUP_ABSENT_STATUS, 0, 0, 0, 0)
    return {
        name: absent
        for name in (
            "selection_rate",
            "tpr_approved",
            "fpr_approved",
            "fnr_approved",
            "brier_score",
            "calibration_residual",
            "absolute_calibration_residual",
        )
    }


def compute_roster_group_metrics(
    y_approved: Sequence[int] | np.ndarray,
    p_approved: Sequence[float] | np.ndarray,
    decisions: Sequence[int] | np.ndarray,
    groups: Sequence[str] | np.ndarray,
    roster: Sequence[str],
    *,
    rules: DenominatorRules = DenominatorRules(),
) -> dict[str, dict[str, MetricEstimate]]:
    """Group metrics for every roster group in roster order; absent groups stay undefined."""

    y = _binary(y_approved, "y_approved")
    predicted = _binary(decisions, "decisions")
    p = np.asarray(p_approved, dtype=np.float64)
    labels = np.asarray(groups).astype(str)
    if not (len(y) == len(p) == len(predicted) == len(labels)):
        raise ValueError("Metric inputs must have equal lengths.")
    if len(set(roster)) != len(roster):
        raise ValueError("Roster must not contain duplicate groups.")
    output: dict[str, dict[str, MetricEstimate]] = {}
    for group in roster:
        mask = labels == group
        if not mask.any():
            output[group] = _absent_group_metrics()
        else:
            output[group] = compute_group_metrics(y[mask], p[mask], predicted[mask], rules=rules)
    return output


def compute_fixed_roster_eod(
    group_metrics: Mapping[str, Mapping[str, MetricEstimate]],
    roster: Sequence[str],
) -> EODEstimate:
    """EOD = max(TPR range, FPR range) over the exact roster; any undefined component invalidates it."""

    roster = tuple(roster)
    if tuple(group_metrics) != roster:
        raise ValueError("Group metrics must cover exactly the configured roster in order; no dynamic shrink.")
    invalid = tuple(
        group
        for group in roster
        if not group_metrics[group]["tpr_approved"].is_defined or not group_metrics[group]["fpr_approved"].is_defined
    )
    if invalid:
        return EODEstimate(None, False, None, None, roster, invalid, EOD_INVALID_REASON)
    tpr = np.asarray([group_metrics[g]["tpr_approved"].value for g in roster], dtype=np.float64)
    fpr = np.asarray([group_metrics[g]["fpr_approved"].value for g in roster], dtype=np.float64)
    tpr_range = float(tpr.max() - tpr.min())
    fpr_range = float(fpr.max() - fpr.min())
    return EODEstimate(max(tpr_range, fpr_range), True, tpr_range, fpr_range, roster, tuple(), "none")


def balanced_accuracy(y_approved: Sequence[int] | np.ndarray, decisions: Sequence[int] | np.ndarray) -> float | None:
    """Full-partition Balanced Accuracy; None (BA-invalid) unless both actual classes are present."""

    y = _binary(y_approved, "y_approved")
    predicted = _binary(decisions, "decisions")
    if len(y) != len(predicted):
        raise ValueError("Inputs must have equal lengths.")
    if not ((y == 1).any() and (y == 0).any()):
        return None
    tpr = float(np.mean(predicted[y == 1] == 1, dtype=np.float64))
    tnr = float(np.mean(predicted[y == 0] == 0, dtype=np.float64))
    return (tpr + tnr) / 2.0


def denied_metrics(y_approved: Sequence[int] | np.ndarray, decisions: Sequence[int] | np.ndarray) -> dict[str, float | None]:
    """Denied (0) is the positive class.

    Precision is None without predicted Denied; recall is None without actual Denied;
    F1 = 2 * true_denied / (actual_denied + predicted_denied), None only when that
    denominator is 0 (so F1 = 0 when the denominator is positive and true_denied = 0).
    """

    y = _binary(y_approved, "y_approved")
    predicted = _binary(decisions, "decisions")
    if len(y) != len(predicted):
        raise ValueError("Inputs must have equal lengths.")
    true_denied = int(np.sum((y == 0) & (predicted == 0)))
    predicted_denied = int(np.sum(predicted == 0))
    actual_denied = int(np.sum(y == 0))
    precision = true_denied / predicted_denied if predicted_denied else None
    recall = true_denied / actual_denied if actual_denied else None
    denominator = actual_denied + predicted_denied
    f1 = 2.0 * true_denied / denominator if denominator else None
    return {"denied_precision": precision, "denied_recall": recall, "denied_f1": f1}


def _change_counts(baseline: np.ndarray, mitigated: np.ndarray) -> dict[str, float | int | None]:
    n_rows = int(len(baseline))
    changed = baseline != mitigated
    return {
        "n_rows": n_rows,
        "n_changed": int(changed.sum()),
        "change_rate": float(changed.mean()) if n_rows else None,
        "approved_to_denied": int(np.sum((baseline == 1) & (mitigated == 0))),
        "denied_to_approved": int(np.sum((baseline == 0) & (mitigated == 1))),
    }


def decision_changes(
    baseline_decisions: Sequence[int] | np.ndarray,
    mitigated_decisions: Sequence[int] | np.ndarray,
    groups: Sequence[str] | np.ndarray,
    group_order: Sequence[str],
) -> list[dict[str, object]]:
    """Overall row first, then one row per group in `group_order` (which must cover every row)."""

    baseline = _binary(baseline_decisions, "baseline_decisions")
    mitigated = _binary(mitigated_decisions, "mitigated_decisions")
    labels = np.asarray(groups).astype(str)
    if not (len(baseline) == len(mitigated) == len(labels)):
        raise ValueError("Inputs must have equal lengths.")
    if not set(labels) <= set(group_order):
        raise ValueError("group_order must cover every group present in the rows.")
    rows: list[dict[str, object]] = [{"scope": "overall", "group": None, **_change_counts(baseline, mitigated)}]
    for group in group_order:
        mask = labels == group
        rows.append({"scope": "group", "group": group, **_change_counts(baseline[mask], mitigated[mask])})
    return rows


def extremum_roles(
    group_metrics: Mapping[str, Mapping[str, MetricEstimate]],
    metric: str,
) -> dict[str, str]:
    """Per-group role for one metric: `min`/`max` (ties share a role), `none`, or `not_evaluable`.

    Undefined groups are `not_evaluable`. With fewer than two defined groups, or when
    every defined value is equal, defined groups are `none`.
    """

    defined = {g: float(m[metric].value) for g, m in group_metrics.items() if m[metric].is_defined}
    roles = {g: ("none" if g in defined else "not_evaluable") for g in group_metrics}
    if len(defined) < 2:
        return roles
    low, high = min(defined.values()), max(defined.values())
    if low == high:
        return roles
    for group, value in defined.items():
        if value == low:
            roles[group] = "min"
        elif value == high:
            roles[group] = "max"
    return roles


def delta_eod(baseline_eod: EODEstimate, mitigated_eod: EODEstimate) -> float | None:
    """Locked sign: mitigated minus baseline; None unless both are valid."""

    if not (baseline_eod.valid and mitigated_eod.valid):
        return None
    return float(mitigated_eod.value) - float(baseline_eod.value)


def ba_cost(baseline_ba: float | None, mitigated_ba: float | None) -> float | None:
    """Locked sign: baseline minus mitigated; None unless both are defined."""

    if baseline_ba is None or mitigated_ba is None:
        return None
    return float(baseline_ba) - float(mitigated_ba)
