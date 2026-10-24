"""RQ4b candidate selection, final-arm reuse and Test evaluation (pure logic).

All rules come from the approved RQ4b contract snapshot
(``rq4b_inputs.APPROVED_CONTRACT``). Every matched delta is
**adversarial minus same-seed lambda=0 control**. The RQ4a ``ba_cost`` helper
(baseline minus mitigated) is deliberately not used; ``delta_ba`` here has the
opposite sign and is tested against it.

Accepted RQ3/RQ4a pure metrics are reused read-only after
``assert_metric_compatibility`` confirms the semantics this module relies on.
Undefined quantities stay ``None`` with a reason; rosters, seeds and bootstrap
replicates are never shrunk, replaced or filled in. Scientific fields are kept
separate from execution status.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from src.dataset_b.fairness.metrics import DenominatorRules, MetricEstimate
from src.dataset_b.mitigation import metrics as rq4a_metrics
from src.dataset_b.mitigation import rq4b_inputs as ri


_CONTRACT = ri.APPROVED_CONTRACT
THRESHOLDS = _CONTRACT["threshold_calibration"]
SELECTION = _CONTRACT["candidate_selection"]
UNCERTAINTY = _CONTRACT["uncertainty"]
ASSESSMENT = _CONTRACT["scientific_assessment"]
METRICS = _CONTRACT["metrics"]
DENOMINATORS = _CONTRACT["denominators"]
RECIPE = _CONTRACT["realmlp_recipe"]

LAMBDA_GRID = tuple(float(value) for value in RECIPE["lambda_grid"])
DEVELOPMENT_SEEDS = tuple(RECIPE["development_seeds"])
FINAL_SEEDS = tuple(RECIPE["final_seeds"])
RETAINED_DEVELOPMENT_UNITS = tuple((float(unit["lambda"]), int(unit["seed"])) for unit in _CONTRACT["development_units"]["retained_completed"])
ADDITIONAL_DEVELOPMENT_UNITS = tuple((float(unit["lambda"]), int(unit["seed"])) for unit in _CONTRACT["development_units"]["additional_approved"])
DEVELOPMENT_UNITS = (*RETAINED_DEVELOPMENT_UNITS, *ADDITIONAL_DEVELOPMENT_UNITS)
SELECTION_POLICY_STATUS = SELECTION["policy_status"]
G3 = tuple(_CONTRACT["groups"]["g3"])
G6 = tuple(_CONTRACT["groups"]["g6"])
# Sex input representation: canonical RQ3 applicant_sex labels (never raw HMDA codes).
# The sex EOD safety audit uses the RQ3 primary groups; coverage-only labels are
# valid inputs that are counted for coverage but are not on the EOD roster.
SEX_ROSTER = ("Male", "Female")
SEX_COVERAGE_ONLY = ("Information not provided", "Not applicable", "Applicant selected both male and female")
SEX_LABELS = (*SEX_ROSTER, *SEX_COVERAGE_ONLY)
RQ3_SEX_CODES = {"1": "Male", "2": "Female", "3": "Information not provided", "4": "Not applicable", "6": "Applicant selected both male and female"}
BA_BUDGET = float(METRICS["ba_budget_primary"])
ELIGIBILITY_MINIMUM = float(SELECTION["eligibility_minimum"])
REPLICATES = int(UNCERTAINTY["within_seed"]["replicates"])
QUANTILES = tuple(UNCERTAINTY["within_seed"]["quantiles"])
QUANTILE_METHOD = UNCERTAINTY["within_seed"]["quantile_interpolation"]
# The cross-run Test contract exports neutral metrics only.  Legacy pure
# assessment helpers remain available for historical synthetic tests, but no
# aggregate support count is active or consumed by the production Test route.
STRONG_SUPPORT_REQUIRED = None
DDOF = int(UNCERTAINTY["across_seed"]["standard_deviation_ddof"])
RULES = DenominatorRules(
    minimum_group_rows=DENOMINATORS["minimum_group_rows"],
    minimum_actual_approved=DENOMINATORS["minimum_actual_approved"],
    minimum_actual_denied=DENOMINATORS["minimum_actual_denied"],
)
DELTA_QUANTITIES = ("delta_eod3", "delta_ba", "delta_eod6")
REASON_GROUP_INVALID = "required_group_metric_invalid"
REASON_BA_UNDEFINED = "ba_undefined_single_class"
REASON_INSUFFICIENT = "insufficient_valid_bootstrap_replicates"


class RQ4bEvaluationError(RuntimeError):
    """Fail-closed RQ4b evaluation error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RQ4bEvaluationError(message)


# ---------------------------------------------------------------------------
# Read-only reuse of accepted metrics, after compatibility checks
# ---------------------------------------------------------------------------


def assert_metric_compatibility() -> dict[str, bool]:
    """Confirm the reused RQ3/RQ4a metric semantics on small fixed fixtures."""
    y = np.array([1] * 40 + [0] * 40 + [1] * 40 + [0] * 40, dtype=np.int8)
    decisions = np.array([1] * 30 + [0] * 10 + [1] * 8 + [0] * 32 + [1] * 20 + [0] * 20 + [1] * 4 + [0] * 36, dtype=np.int8)
    groups = np.array(["A"] * 80 + ["B"] * 80)
    p = np.full(len(y), 0.5)
    roster_metrics = rq4a_metrics.compute_roster_group_metrics(y, p, decisions, groups, ("A", "B"), rules=RULES)
    eod = rq4a_metrics.compute_fixed_roster_eod(roster_metrics, ("A", "B"))
    absent = rq4a_metrics.compute_fixed_roster_eod(
        rq4a_metrics.compute_roster_group_metrics(y, p, decisions, groups, ("A", "B", "C"), rules=RULES), ("A", "B", "C")
    )
    checks = {
        "denominator_rules_match_contract": (RULES.minimum_group_rows, RULES.minimum_actual_approved, RULES.minimum_actual_denied) == (100, 30, 30),
        "tpr_is_approved_positive": roster_metrics["A"]["tpr_approved"].value == 30 / 40,
        "fpr_is_predicted_approved_among_denied": roster_metrics["A"]["fpr_approved"].value == 8 / 40,
        "eod_is_max_of_tpr_and_fpr_range": eod.valid and eod.value == max(abs(30 / 40 - 20 / 40), abs(8 / 40 - 4 / 40)),
        "absent_group_invalidates_without_shrink": (not absent.valid) and absent.value is None and absent.invalid_groups == ("C",),
        "ba_is_mean_of_tpr_and_tnr": rq4a_metrics.balanced_accuracy(y, decisions) == ((50 / 80) + (68 / 80)) / 2,
        "ba_none_for_single_class": rq4a_metrics.balanced_accuracy(np.ones(4, dtype=np.int8), np.ones(4, dtype=np.int8)) is None,
        "rq4a_ba_cost_is_baseline_minus_mitigated": rq4a_metrics.ba_cost(0.8, 0.7) == 0.8 - 0.7,
    }
    _require(all(checks.values()), f"Reused metric semantics differ from RQ4b expectations: {checks}")
    return checks


_COMPATIBILITY: dict[str, bool] = {}


def _ensure_compatible() -> None:
    if not _COMPATIBILITY:
        _COMPATIBILITY.update(assert_metric_compatibility())


def delta(adversarial: float | None, control: float | None) -> float | None:
    """Locked RQ4b sign: adversarial minus same-seed lambda=0 control; None unless both are defined."""
    if adversarial is None or control is None:
        return None
    return float(adversarial) - float(control)


def delta_ba(adversarial_ba: float | None, control_ba: float | None) -> float | None:
    """adversarial - control (the negative of RQ4a ``ba_cost(control, adversarial)``)."""
    return delta(adversarial_ba, control_ba)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PartitionPredictions:
    """One arm's predictions for a complete partition, aligned to its record IDs."""

    record_ids: tuple[str, ...]
    y: np.ndarray
    p_approved: np.ndarray
    race: np.ndarray
    sex: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.record_ids)
        _require(n > 0 and len(set(self.record_ids)) == n, "Prediction record IDs must be nonempty and unique.")
        _require(len(self.y) == len(self.p_approved) == len(self.race) == len(self.sex) == n, "Prediction columns are misaligned.")
        _require(np.isin(self.y, [0, 1]).all(), "y_approved must be binary.")
        p = np.asarray(self.p_approved, dtype=np.float64)
        _require(np.isfinite(p).all() and np.all((p >= 0.0) & (p <= 1.0)), "p_approved must be finite and within [0, 1].")
        _require(set(np.asarray(self.race).astype(str)) <= set(G6) | set(ri.COVERAGE_ONLY_GROUPS), "Race groups lie outside the contract.")
        validate_sex_labels(self.sex)


def validate_sex_labels(values: Sequence[Any]) -> None:
    """Require canonical RQ3 sex labels; reject raw codes, unknown, blank and missing values.

    This is an encoding check only: a valid label set may still lack a roster
    group or fail its denominator, which the EOD audit reports as undefined.
    """
    items = list(np.asarray(values, dtype=object))
    missing = [v for v in items if v is None or (isinstance(v, float) and math.isnan(v))]
    _require(not missing, f"Sex metadata contains {len(missing)} missing values.")
    _require(all(isinstance(v, str) for v in items), "Sex metadata must be canonical RQ3 label strings.")
    blank = [v for v in items if not v.strip() or v.strip() != v]
    _require(not blank, f"Sex metadata contains {len(blank)} blank or padded values.")
    raw = sorted({v for v in items if v in RQ3_SEX_CODES})
    _require(not raw, f"Sex metadata contains raw RQ3 applicant_sex codes {raw}; normalize with normalize_rq3_sex_codes first.")
    unknown = sorted(set(items) - set(SEX_LABELS))
    _require(not unknown, f"Sex metadata contains labels outside the RQ3 mapping: {unknown}.")


def assert_rq3_sex_mapping(rq3_protocol: Mapping[str, Any]) -> dict[str, str]:
    """The RQ3 applicant_sex mapping must equal the codes this module accepts (read-only check)."""
    sex = rq3_protocol["attributes"]["applicant_sex"]
    mapping = {str(code): str(label) for label, code in sex["primary_groups"].items()}
    mapping.update({str(code): str(label) for label, code in sex["coverage_only_groups"].items()})
    _require(mapping == RQ3_SEX_CODES, f"RQ3 applicant_sex mapping differs from RQ4b sex labels: {mapping}.")
    _require(tuple(sex["primary_groups"]) == SEX_ROSTER, "RQ3 applicant_sex primary groups differ from the sex EOD roster.")
    return mapping


def normalize_rq3_sex_codes(codes: Sequence[Any], rq3_protocol: Mapping[str, Any]) -> np.ndarray:
    """Explicitly map approved RQ3 applicant_sex codes to canonical labels; reject anything else."""
    mapping = assert_rq3_sex_mapping(rq3_protocol)
    items = list(np.asarray(codes, dtype=object))
    _require(all(isinstance(v, str) for v in items), "applicant_sex codes must be strings (missing values are rejected).")
    _require(all(v.strip() == v and v for v in items), "applicant_sex codes contain blank or padded values.")
    unknown = sorted(set(items) - set(mapping))
    _require(not unknown, f"applicant_sex codes outside the RQ3 mapping: {unknown}.")
    return np.asarray([mapping[v] for v in items], dtype=object)


def require_paired(control: PartitionPredictions, adversarial: PartitionPredictions) -> None:
    """Matched arms must cover identical ordered rows, labels and audit groups."""
    _require(control.record_ids == adversarial.record_ids, "Control and adversarial arms cover different ordered record IDs.")
    _require(np.array_equal(control.y, adversarial.y), "Control and adversarial labels differ.")
    _require(np.array_equal(np.asarray(control.race).astype(str), np.asarray(adversarial.race).astype(str)), "Arms carry different race metadata.")
    _require(np.array_equal(np.asarray(control.sex).astype(str), np.asarray(adversarial.sex).astype(str)), "Arms carry different sex metadata.")


# ---------------------------------------------------------------------------
# Global threshold calibration
# ---------------------------------------------------------------------------


def threshold_candidates() -> np.ndarray:
    grid = THRESHOLDS["candidates"]
    count = int(grid["count"])
    values = np.arange(1, count + 1, dtype=np.int64) / 1000.0
    _require(values[0] == grid["start"] and values[-1] == grid["stop"], "Threshold grid differs from the contract.")
    return values


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    balanced_accuracy: float
    tied_thresholds: tuple[float, ...]
    candidate_count: int


def calibrate_threshold(y: np.ndarray, p_approved: np.ndarray) -> ThresholdSelection:
    """One global threshold on the complete Validation partition: maximize BA (``p >= t``).

    Exact BA ties choose the minimum absolute distance to 0.5, then the lower threshold.
    """
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p_approved, dtype=np.float64)
    _require((y == 1).any() and (y == 0).any(), "Threshold calibration requires both classes.")
    candidates = threshold_candidates()
    scores = np.array([rq4a_metrics.balanced_accuracy(y, (p >= t).astype(np.int8)) for t in candidates], dtype=np.float64)
    best = scores.max()
    tied = candidates[scores == best]
    return ThresholdSelection(resolve_threshold_tie(tied), float(best), tuple(float(t) for t in tied), len(candidates))


def resolve_threshold_tie(tied: Sequence[float]) -> float:
    """Exact BA ties: minimum absolute distance to 0.5, then the lower threshold."""
    _require(len(tied) > 0, "No tied thresholds supplied.")
    return float(min(tied, key=lambda t: (abs(float(t) - 0.5), float(t))))


def decisions_at(p_approved: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(p_approved, dtype=np.float64) >= float(threshold)).astype(np.int8)


# ---------------------------------------------------------------------------
# Point metrics for one arm
# ---------------------------------------------------------------------------


def _eod(predictions: PartitionPredictions, decisions: np.ndarray, roster: Sequence[str], labels: np.ndarray) -> tuple[rq4a_metrics.EODEstimate, dict]:
    group_metrics = rq4a_metrics.compute_roster_group_metrics(predictions.y, predictions.p_approved, decisions, labels, roster, rules=RULES)
    return rq4a_metrics.compute_fixed_roster_eod(group_metrics, roster), group_metrics


def _metric_dict(estimate: MetricEstimate) -> dict[str, Any]:
    return {"value": estimate.value, "status": estimate.status, "denominator": estimate.denominator,
            "group_rows": estimate.group_rows, "actual_approved": estimate.actual_approved, "actual_denied": estimate.actual_denied}


def _eod_dict(estimate: rq4a_metrics.EODEstimate) -> dict[str, Any]:
    return {"value": estimate.value, "valid": estimate.valid, "tpr_range": estimate.tpr_range, "fpr_range": estimate.fpr_range,
            "roster": list(estimate.roster), "invalid_groups": list(estimate.invalid_groups), "reason_code": estimate.reason_code}


def _core(y: np.ndarray, p: np.ndarray, race: np.ndarray, threshold: float) -> dict[str, Any]:
    decisions = decisions_at(p, threshold)
    out = {"ba": rq4a_metrics.balanced_accuracy(y, decisions)}
    for name, roster in (("eod3", G3), ("eod6", G6)):
        group_metrics = rq4a_metrics.compute_roster_group_metrics(y, p, decisions, race, roster, rules=RULES)
        out[name] = rq4a_metrics.compute_fixed_roster_eod(group_metrics, roster)
    return out


def core_quantities(predictions: PartitionPredictions, threshold: float) -> dict[str, Any]:
    """Primary quantities (also used in every bootstrap replicate): full-partition BA, EOD3, EOD6."""
    return _core(np.asarray(predictions.y), np.asarray(predictions.p_approved, dtype=np.float64), np.asarray(predictions.race).astype(str), threshold)


def arm_point_metrics(predictions: PartitionPredictions, threshold: float) -> dict[str, Any]:
    """Primary metrics plus the required audits for one arm on a complete partition."""
    decisions = decisions_at(predictions.p_approved, threshold)
    race = np.asarray(predictions.race).astype(str)
    sex = np.asarray(predictions.sex).astype(str)
    eod3, g3_metrics = _eod(predictions, decisions, G3, race)
    eod6, g6_metrics = _eod(predictions, decisions, G6, race)
    all_groups = (*G6, *ri.COVERAGE_ONLY_GROUPS)
    coverage_metrics = rq4a_metrics.compute_roster_group_metrics(predictions.y, predictions.p_approved, decisions, race, all_groups, rules=RULES)
    sex_eod, sex_metrics = _eod(predictions, decisions, SEX_ROSTER, sex)
    g3_rows = np.isin(race, G3)
    return {
        "threshold": float(threshold),
        "rows": int(len(decisions)),
        "ba": rq4a_metrics.balanced_accuracy(predictions.y, decisions),
        "ba_g3_rows_audit_only": rq4a_metrics.balanced_accuracy(predictions.y[g3_rows], decisions[g3_rows]) if g3_rows.any() else None,
        "eod3": _eod_dict(eod3),
        "eod6": _eod_dict(eod6),
        "sex_eod_safety_audit": _eod_dict(sex_eod),
        **rq4a_metrics.denied_metrics(predictions.y, decisions),
        "per_group": {group: {name: _metric_dict(coverage_metrics[group][name]) for name in ("tpr_approved", "fpr_approved", "selection_rate")} for group in all_groups},
        "extrema_groups": {metric: rq4a_metrics.extremum_roles(coverage_metrics, metric) for metric in ("tpr_approved", "fpr_approved", "selection_rate")},
        "group_coverage": {group: int(np.sum(race == group)) for group in all_groups},
        "sex_groups": {group: _metric_dict(sex_metrics[group]["tpr_approved"]) for group in SEX_ROSTER},
        "sex_group_coverage": {group: int(np.sum(sex == group)) for group in SEX_LABELS},
    }


def _value(eod: rq4a_metrics.EODEstimate) -> float | None:
    return eod.value if eod.valid else None


def matched_deltas(control: Mapping[str, Any], adversarial: Mapping[str, Any]) -> dict[str, Any]:
    """adversarial - control for EOD3, EOD6 and full-partition BA; None with a reason when undefined."""
    out: dict[str, Any] = {}
    reasons: dict[str, str] = {}
    for name, key in (("delta_eod3", "eod3"), ("delta_eod6", "eod6")):
        out[name] = delta(_value(adversarial[key]), _value(control[key]))
        if out[name] is None:
            reasons[name] = REASON_GROUP_INVALID
    out["delta_ba"] = delta_ba(adversarial["ba"], control["ba"])
    if out["delta_ba"] is None:
        reasons["delta_ba"] = REASON_BA_UNDEFINED
    out["undefined_reasons"] = reasons
    return out


# ---------------------------------------------------------------------------
# Validation candidate evaluation and lambda selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateEvaluation:
    adversary_lambda: float
    model_seed: int
    threshold: ThresholdSelection
    ba: float | None
    eod3: float | None
    eod6: float | None
    eod3_invalid_groups: tuple[str, ...] = ()


def evaluate_candidate(validation: PartitionPredictions, *, adversary_lambda: float, model_seed: int) -> CandidateEvaluation:
    """Calibrate this candidate's global threshold and evaluate BA/EOD3/EOD6 on complete Validation."""
    _require((float(adversary_lambda), int(model_seed)) in DEVELOPMENT_UNITS, "Candidate lambda/seed is not an authorized amended development unit.")
    _ensure_compatible()
    selection = calibrate_threshold(validation.y, validation.p_approved)
    core = core_quantities(validation, selection.threshold)
    return CandidateEvaluation(float(adversary_lambda), int(model_seed), selection, core["ba"], _value(core["eod3"]), _value(core["eod6"]), core["eod3"].invalid_groups)


def select_lambda(evaluations: Sequence[CandidateEvaluation]) -> dict[str, Any]:
    """Apply the locked Validation-only sparse-roster selection policy."""
    cells = {(e.adversary_lambda, e.model_seed): e for e in evaluations}
    _require(len(cells) == len(evaluations), "Duplicate candidate evaluations.")
    expected = set(DEVELOPMENT_UNITS)
    _require(set(cells) == expected, f"Sparse development roster incomplete or unexpected: missing={sorted(expected - set(cells))}, extra={sorted(set(cells) - expected)}.")
    units = []
    for lam, seed in DEVELOPMENT_UNITS:
        candidate = cells[(lam, seed)]
        control = cells.get((0.0, seed))
        matched_delta = delta_ba(candidate.ba, control.ba) if control is not None else None
        units.append({
            "adversary_lambda": lam,
            "model_seed": seed,
            "threshold": candidate.threshold.threshold,
            "balanced_accuracy": candidate.ba,
            "eod3": candidate.eod3,
            "eod6_audit_only": candidate.eod6,
            "matched_delta_ba": matched_delta,
            "matched_control_available": control is not None,
        })
    selectable = tuple(float(x) for x in (0.0, 0.01))
    candidates = []
    for lam in selectable:
        rows = [row for row in units if row["adversary_lambda"] == lam]
        deltas = [row["matched_delta_ba"] for row in rows]
        eods = [row["eod3"] for row in rows]
        eligible = len(rows) == len(DEVELOPMENT_SEEDS) and all(v is not None for v in deltas) and float(np.mean(deltas)) >= ELIGIBILITY_MINIMUM
        candidates.append({"adversary_lambda": lam, "mean_matched_delta_ba": None if any(v is None for v in deltas) else float(np.mean(deltas)),
                           "mean_eod3": None if any(v is None for v in eods) else float(np.mean(eods)), "eligible": eligible,
                           "invalid_reason": None if eligible else "missing_metric_or_ba_eligibility_failure"})
    eligible = [row for row in candidates if row["eligible"] and row["mean_eod3"] is not None]
    selected = min(eligible, key=lambda row: (row["mean_eod3"], row["adversary_lambda"])) if eligible else None
    return {"selected_lambda": None if selected is None else selected["adversary_lambda"], "lambda_zero_selected": bool(selected and selected["adversary_lambda"] == 0.0),
            "status": "selected" if selected else "not_evaluable", "reason": None if selected else "no_selectable_candidate_meets_locked_eligibility",
            "arbitrary_two_of_three_rule_used": False, "eod6_role": "audit_only", "units": units, "candidates": candidates,
            "screening_only_lambdas": [0.03, 0.1, 0.3], "limitation": "threshold calibration and lambda selection share the complete Validation partition (optimism)"}


# ---------------------------------------------------------------------------
# Final arms: development artifact reuse and lambda=0 deduplication
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactRef:
    """Identity of one fitted refit artifact and its frozen Validation threshold."""

    adversary_lambda: float
    model_seed: int
    artifact_id: str
    model_sha256: str
    selection_record_sha256: str
    threshold: float


def plan_final_arms(selected_lambda: float, development: Mapping[tuple[float, int], ArtifactRef]) -> dict[int, dict[str, Any]]:
    """Per final seed: control (lambda=0) and adversarial (selected lambda) sources.

    Under the amendment, all three final seeds must reuse authorized development
    artifacts; no post-selection fit is allowed. This helper remains fail-closed
    until a three-seed selection policy is explicitly approved.
    """
    _require(SELECTION_POLICY_STATUS == "approved_evaluation_only_three_seed_validation_selection", "Final-arm planning requires the locked evaluation-only selection policy.")
    selected = float(selected_lambda)
    _require(selected in LAMBDA_GRID, "Selected lambda is not in the locked grid.")
    plan: dict[int, dict[str, Any]] = {}
    for seed in FINAL_SEEDS:
        arms: dict[str, Any] = {}
        for role, lam in (("control", 0.0), ("adversarial", selected)):
            if seed in DEVELOPMENT_SEEDS:
                ref = development.get((lam, seed))
                _require(ref is not None, f"Development artifact for lambda={lam}, seed={seed} is missing.")
                _require((ref.adversary_lambda, ref.model_seed) == (lam, seed), "Development artifact identity differs from its grid cell.")
                arms[role] = {"source": "reuse_development", "artifact": ref}
            else:
                arms[role] = {"source": "fit_required", "adversary_lambda": lam, "model_seed": seed}
        arms["deduplicated_lambda_zero"] = selected == 0.0
        if selected == 0.0:
            arms["adversarial"] = {"source": "same_as_control"}
        plan[seed] = arms
    return plan


def required_new_fits(plan: Mapping[int, Mapping[str, Any]]) -> list[tuple[float, int]]:
    fits = []
    for seed, arms in plan.items():
        for role in ("control", "adversarial"):
            arm = arms[role]
            if arm["source"] == "fit_required":
                fits.append((float(arm["adversary_lambda"]), int(seed)))
    return sorted(set(fits))


def validate_reuse_identity(final: ArtifactRef, development: ArtifactRef) -> None:
    """A reused final-seed artifact must be exactly the accepted development artifact."""
    _require(final == development, f"Reused artifact identity differs from development: {final} vs {development}.")


# ---------------------------------------------------------------------------
# Locked Test bootstrap
# ---------------------------------------------------------------------------


def validate_locked_indices(indices: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    """Locked RQ1 Y-stratified layout: (1000, n_test) int32; Denied columns first, then Approved."""
    y = np.asarray(y_test, dtype=np.int8)
    n_denied = int(np.sum(y == 0))
    _require(isinstance(indices, np.ndarray) and indices.dtype == np.int32, "Locked Test bootstrap indices must be int32.")
    _require(indices.shape == (REPLICATES, len(y)), f"Locked Test bootstrap indices must have shape ({REPLICATES}, {len(y)}).")
    _require(int(indices.min()) >= 0 and int(indices.max()) < len(y), "Locked Test bootstrap indices are out of range.")
    _require(np.all(y[indices[:, :n_denied]] == 0) and np.all(y[indices[:, n_denied:]] == 1), "Locked Test bootstrap Y-stratification differs.")
    return indices


def _ci(values: Sequence[float | None], invalid: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Percentile 95% CI only with REPLICATES/REPLICATES valid values; never topped up."""
    valid = [float(v) for v in values if v is not None]
    summary: dict[str, Any] = {"replicates": len(values), "valid_replicates": len(valid), "invalid_replicates": list(invalid),
                               "quantiles": list(QUANTILES), "quantile_method": QUANTILE_METHOD}
    if len(values) != REPLICATES or len(valid) != REPLICATES:
        summary.update(lower=None, upper=None, evaluable=False, reason=REASON_INSUFFICIENT)
        return summary
    lower, upper = np.quantile(np.asarray(valid, dtype=np.float64), list(QUANTILES), method=QUANTILE_METHOD)
    summary.update(lower=float(lower), upper=float(upper), evaluable=True, reason=None)
    return summary


def paired_bootstrap(
    control: PartitionPredictions,
    adversarial: PartitionPredictions,
    *,
    control_threshold: float,
    adversarial_threshold: float,
    indices: np.ndarray,
) -> dict[str, Any]:
    """Both arms on identical resampled Test rows per replicate; models and thresholds fixed."""
    require_paired(control, adversarial)
    validate_locked_indices(indices, control.y)
    y = np.asarray(control.y)
    race = np.asarray(control.race).astype(str)
    p_control = np.asarray(control.p_approved, dtype=np.float64)
    p_adversarial = np.asarray(adversarial.p_approved, dtype=np.float64)
    values = {name: [] for name in DELTA_QUANTITIES}
    invalid = {name: [] for name in DELTA_QUANTITIES}
    for replicate, rows in enumerate(indices):
        c = _core(y[rows], p_control[rows], race[rows], control_threshold)
        a = _core(y[rows], p_adversarial[rows], race[rows], adversarial_threshold)
        deltas = matched_deltas(c, a)
        for name in DELTA_QUANTITIES:
            values[name].append(deltas[name])
            if deltas[name] is None:
                invalid[name].append({"replicate": replicate, "reason": deltas["undefined_reasons"][name]})
    return {name: {**_ci(values[name], invalid[name]), "values": values[name]} for name in DELTA_QUANTITIES}


# ---------------------------------------------------------------------------
# Per-seed criteria, strong support and five-seed summaries
# ---------------------------------------------------------------------------


def point_criterion(deltas: Mapping[str, Any]) -> dict[str, Any]:
    """delta EOD3 < 0 AND full-Test delta BA >= -0.01 (non-strict); missing -> not_evaluable."""
    d_eod3, d_ba = deltas.get("delta_eod3"), deltas.get("delta_ba")
    if d_eod3 is None or d_ba is None:
        return {"status": "not_evaluable", "fairness_condition": None, "utility_condition": None}
    fairness, utility = d_eod3 < 0.0, d_ba >= -BA_BUDGET
    return {"status": "pass" if fairness and utility else "fail", "fairness_condition": fairness, "utility_condition": utility}


def strong_support_seed(bootstrap: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """upper CI(delta EOD3) < 0 AND lower CI(delta BA) > -0.01 (strict); an invalid CI cannot support."""
    eod3, ba = bootstrap["delta_eod3"], bootstrap["delta_ba"]
    if not (eod3.get("evaluable") and ba.get("evaluable")):
        return {"status": "not_evaluable", "supports": False, "fairness_condition": None, "utility_condition": None}
    fairness, utility = eod3["upper"] < 0.0, ba["lower"] > -BA_BUDGET
    return {"status": "evaluated", "supports": bool(fairness and utility), "fairness_condition": fairness, "utility_condition": utility}


def across_seed_summary(per_seed: Mapping[int, Mapping[str, Any]], quantity: str) -> dict[str, Any]:
    """Mean, sample SD (ddof=1), min, max, range over all three amended seeds."""
    _require(tuple(sorted(per_seed)) == tuple(sorted(FINAL_SEEDS)), "Across-seed summaries require exactly the three amended seeds.")
    values = [per_seed[seed]["deltas"].get(quantity) for seed in FINAL_SEEDS]
    missing = [seed for seed, value in zip(FINAL_SEEDS, values) if value is None]
    if missing:
        return {"status": "not_evaluable", "missing_seeds": missing, "seeds": list(FINAL_SEEDS)}
    array = np.asarray(values, dtype=np.float64)
    return {
        "status": "evaluated", "seeds": list(FINAL_SEEDS), "values": array.tolist(), "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=DDOF)), "ddof": DDOF, "min": float(array.min()), "max": float(array.max()),
        "range": float(array.max() - array.min()), "pooled_bootstrap_replicates": False,
    }


def evaluate_seed(
    control: PartitionPredictions,
    adversarial: PartitionPredictions,
    *,
    control_threshold: float,
    adversarial_threshold: float,
    indices: np.ndarray,
) -> dict[str, Any]:
    """Point metrics, audits, matched deltas, paired bootstrap and per-seed criteria for one final seed."""
    _ensure_compatible()
    require_paired(control, adversarial)
    c_core, a_core = core_quantities(control, control_threshold), core_quantities(adversarial, adversarial_threshold)
    deltas = matched_deltas(c_core, a_core)
    bootstrap = paired_bootstrap(control, adversarial, control_threshold=control_threshold, adversarial_threshold=adversarial_threshold, indices=indices)
    changes = rq4a_metrics.decision_changes(
        decisions_at(control.p_approved, control_threshold), decisions_at(adversarial.p_approved, adversarial_threshold),
        np.asarray(control.race).astype(str), (*G6, *ri.COVERAGE_ONLY_GROUPS),
    )
    return {
        "control": arm_point_metrics(control, control_threshold),
        "adversarial": arm_point_metrics(adversarial, adversarial_threshold),
        "deltas": deltas,
        "decision_changes": changes,
        "bootstrap": bootstrap,
        "point_criterion": point_criterion(deltas),
        "strong_support": strong_support_seed(bootstrap),
    }


def assess_final_seeds(per_seed: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """Descriptive assessment across three seeds; aggregate support remains unapproved."""
    _require(tuple(sorted(per_seed)) == tuple(sorted(FINAL_SEEDS)), "Assessment requires exactly the three amended seeds; none may be dropped.")
    supported = [seed for seed in FINAL_SEEDS if per_seed[seed]["strong_support"]["supports"]]
    not_evaluable = [seed for seed in FINAL_SEEDS if per_seed[seed]["strong_support"]["status"] == "not_evaluable"]
    return {
        "per_seed_point_criterion": {seed: per_seed[seed]["point_criterion"]["status"] for seed in FINAL_SEEDS},
        "strong_support": {
            "supported_seeds": supported,
            "not_evaluable_seeds": not_evaluable,
            "supported_count": len(supported),
            "required_count": STRONG_SUPPORT_REQUIRED,
            "final_seed_count": len(FINAL_SEEDS),
            "status": SELECTION_POLICY_STATUS,
            "arbitrary_two_of_three_rule_used": False,
            "joint_confidence_claim": False,
        },
        "across_seed": {quantity: across_seed_summary(per_seed, quantity) for quantity in DELTA_QUANTITIES},
        "stability_status": ASSESSMENT["stability_status"],
        "strong_support_implies_stable": False,
        "interpretation_boundary": "exploratory model-behaviour evidence; not causal, discrimination, legal-compliance, creditworthiness or default-risk evidence",
    }


def fairness_generalization_gap(test_eod: float | None, validation_eod: float | None) -> float | None:
    """Test EOD minus development (complete Validation) EOD; None unless both are defined."""
    if test_eod is None or validation_eod is None:
        return None
    return float(test_eod) - float(validation_eod)
