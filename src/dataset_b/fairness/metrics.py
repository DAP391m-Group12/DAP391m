"""Pure numerical definitions for the Approved-positive Dataset B RQ3 audit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


GROUP_METRIC_NAMES = (
    "selection_rate",
    "tpr_approved",
    "fpr_approved",
    "fnr_approved",
    "brier_score",
    "calibration_residual",
    "absolute_calibration_residual",
)


@dataclass(frozen=True)
class DenominatorRules:
    minimum_group_rows: int = 100
    minimum_actual_approved: int = 30
    minimum_actual_denied: int = 30

    def __post_init__(self) -> None:
        if min(
            self.minimum_group_rows,
            self.minimum_actual_approved,
            self.minimum_actual_denied,
        ) < 1:
            raise ValueError("All denominator minimums must be positive integers.")


@dataclass(frozen=True)
class MetricEstimate:
    value: float | None
    status: str
    denominator: int
    group_rows: int
    actual_approved: int
    actual_denied: int

    @property
    def is_defined(self) -> bool:
        return self.value is not None and self.status == "defined"


@dataclass(frozen=True)
class GapEstimate:
    value: float | None
    status: str
    target_value: float | None
    reference_value: float | None

    @property
    def is_defined(self) -> bool:
        return self.value is not None and self.status == "defined"


@dataclass(frozen=True)
class RangeEstimate:
    value: float | None
    status: str
    groups_used: tuple[str, ...]

    @property
    def is_defined(self) -> bool:
        return self.value is not None and self.status == "defined"


def _validated_arrays(
    y_approved: Sequence[int] | np.ndarray,
    p_approved: Sequence[float] | np.ndarray,
    predicted_approved: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y_approved)
    p = np.asarray(p_approved, dtype=np.float64)
    predicted = np.asarray(predicted_approved)
    if y.ndim != 1 or p.ndim != 1 or predicted.ndim != 1:
        raise ValueError("RQ3 metric inputs must be one-dimensional.")
    if not (len(y) == len(p) == len(predicted)):
        raise ValueError("RQ3 metric inputs must have equal lengths.")
    if len(y) == 0:
        raise ValueError("RQ3 metric inputs must not be empty.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_approved must contain only 0 and 1.")
    if not np.isin(predicted, [0, 1]).all():
        raise ValueError("predicted_approved must contain only 0 and 1.")
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p_approved must be finite and within [0, 1].")
    return y.astype(np.int8), p, predicted.astype(np.int8)


def _estimate(
    value: float | None,
    status: str,
    denominator: int,
    *,
    group_rows: int,
    actual_approved: int,
    actual_denied: int,
) -> MetricEstimate:
    return MetricEstimate(
        value=value,
        status=status,
        denominator=denominator,
        group_rows=group_rows,
        actual_approved=actual_approved,
        actual_denied=actual_denied,
    )


def compute_group_metrics(
    y_approved: Sequence[int] | np.ndarray,
    p_approved: Sequence[float] | np.ndarray,
    predicted_approved: Sequence[int] | np.ndarray,
    *,
    rules: DenominatorRules = DenominatorRules(),
) -> dict[str, MetricEstimate]:
    """Compute locked Approved-positive group metrics with explicit gates."""

    y, p, predicted = _validated_arrays(y_approved, p_approved, predicted_approved)
    group_rows = int(len(y))
    actual_approved = int(np.sum(y == 1))
    actual_denied = int(np.sum(y == 0))
    common = {
        "group_rows": group_rows,
        "actual_approved": actual_approved,
        "actual_denied": actual_denied,
    }

    if group_rows >= rules.minimum_group_rows:
        selection_rate = _estimate(
            float(np.mean(predicted, dtype=np.float64)),
            "defined",
            group_rows,
            **common,
        )
        brier_score = _estimate(
            float(np.mean(np.square(p - y), dtype=np.float64)),
            "defined",
            group_rows,
            **common,
        )
        residual_value = float(np.mean(p - y, dtype=np.float64))
        calibration_residual = _estimate(
            residual_value, "defined", group_rows, **common
        )
        absolute_calibration_residual = _estimate(
            abs(residual_value), "defined", group_rows, **common
        )
    else:
        status = "undefined_insufficient_group_rows"
        selection_rate = _estimate(None, status, group_rows, **common)
        brier_score = _estimate(None, status, group_rows, **common)
        calibration_residual = _estimate(None, status, group_rows, **common)
        absolute_calibration_residual = _estimate(None, status, group_rows, **common)

    if actual_approved >= rules.minimum_actual_approved:
        true_positive_rate = float(np.mean(predicted[y == 1], dtype=np.float64))
        tpr = _estimate(true_positive_rate, "defined", actual_approved, **common)
        fnr = _estimate(1.0 - true_positive_rate, "defined", actual_approved, **common)
    else:
        status = "undefined_insufficient_actual_approved"
        tpr = _estimate(None, status, actual_approved, **common)
        fnr = _estimate(None, status, actual_approved, **common)

    if actual_denied >= rules.minimum_actual_denied:
        fpr = _estimate(
            float(np.mean(predicted[y == 0], dtype=np.float64)),
            "defined",
            actual_denied,
            **common,
        )
    else:
        fpr = _estimate(
            None,
            "undefined_insufficient_actual_denied",
            actual_denied,
            **common,
        )

    return {
        "selection_rate": selection_rate,
        "tpr_approved": tpr,
        "fpr_approved": fpr,
        "fnr_approved": fnr,
        "brier_score": brier_score,
        "calibration_residual": calibration_residual,
        "absolute_calibration_residual": absolute_calibration_residual,
    }


def _gap(target: MetricEstimate, reference: MetricEstimate, *, absolute: bool) -> GapEstimate:
    if not target.is_defined or not reference.is_defined:
        return GapEstimate(None, "undefined_component_metric", target.value, reference.value)
    signed = float(target.value - reference.value)
    return GapEstimate(abs(signed) if absolute else signed, "defined", target.value, reference.value)


def compute_pairwise_gaps(
    target: Mapping[str, MetricEstimate],
    reference: Mapping[str, MetricEstimate],
) -> dict[str, GapEstimate]:
    """Return target-minus-reference signed gaps and their absolute magnitudes."""

    output: dict[str, GapEstimate] = {}
    for metric in (
        "selection_rate",
        "tpr_approved",
        "fpr_approved",
        "fnr_approved",
        "calibration_residual",
    ):
        output[f"{metric}_gap_signed"] = _gap(target[metric], reference[metric], absolute=False)
        output[f"{metric}_gap_absolute"] = _gap(target[metric], reference[metric], absolute=True)

    tpr_gap = output["tpr_approved_gap_signed"]
    fpr_gap = output["fpr_approved_gap_signed"]
    if tpr_gap.is_defined and fpr_gap.is_defined:
        output["equalized_odds_gap"] = GapEstimate(
            max(abs(float(tpr_gap.value)), abs(float(fpr_gap.value))),
            "defined",
            None,
            None,
        )
    else:
        output["equalized_odds_gap"] = GapEstimate(
            None, "undefined_component_metric", None, None
        )
    return output


def _range_for_metric(
    group_metrics: Mapping[str, Mapping[str, MetricEstimate]], metric: str
) -> RangeEstimate:
    values = [
        (group, estimates[metric])
        for group, estimates in group_metrics.items()
        if estimates[metric].is_defined
    ]
    if len(values) < 2:
        return RangeEstimate(None, "undefined_fewer_than_two_eligible_groups", tuple(g for g, _ in values))
    used = tuple(group for group, _ in values)
    numeric = np.asarray([estimate.value for _, estimate in values], dtype=np.float64)
    return RangeEstimate(float(np.max(numeric) - np.min(numeric)), "defined", used)


def compute_global_race_ranges(
    eligible_group_metrics: Mapping[str, Mapping[str, MetricEstimate]],
) -> dict[str, RangeEstimate]:
    """Compute multi-group ranges after coverage-only groups have been excluded."""

    selection = _range_for_metric(eligible_group_metrics, "selection_rate")
    tpr = _range_for_metric(eligible_group_metrics, "tpr_approved")
    fpr = _range_for_metric(eligible_group_metrics, "fpr_approved")
    if tpr.is_defined and fpr.is_defined:
        equalized_odds = RangeEstimate(
            max(float(tpr.value), float(fpr.value)),
            "defined",
            tuple(group for group in eligible_group_metrics if group in tpr.groups_used and group in fpr.groups_used),
        )
    else:
        equalized_odds = RangeEstimate(None, "undefined_component_metric", tuple())
    return {
        "demographic_parity_difference_range": selection,
        "tpr_approved_range": tpr,
        "fpr_approved_range": fpr,
        "equalized_odds_difference_range": equalized_odds,
    }
