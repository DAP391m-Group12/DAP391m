"""Tests for RQ4b candidate selection and Test evaluation logic (synthetic fixtures only).

Every partition and bootstrap index matrix here is synthetic. No project
Validation/Test data, predictions or locked index file is read.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from src.dataset_b.mitigation import metrics as rq4a_metrics
from src.dataset_b.mitigation import rq4b_evaluation as ev
from src.dataset_b.mitigation import rq4b_inputs as ri
from src.utils.path import PROJECT_ROOT


ALL_GROUPS = (*ri.G6_GROUPS, *ri.COVERAGE_ONLY_GROUPS)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _partition(seed: int, *, per_group: int = 200, groups=ALL_GROUPS, shift: float = 0.0, denied_override: dict | None = None):
    """Each group has `per_group` rows (half approved) unless overridden; p depends on y and group."""
    rng = np.random.default_rng(seed)
    ys, races = [], []
    for index, group in enumerate(groups):
        n_denied = (denied_override or {}).get(group, per_group // 2)
        ys += [0] * n_denied + [1] * (per_group - n_denied)
        races += [group] * per_group
    y = np.asarray(ys, dtype=np.int8)
    race = np.asarray(races, dtype=object)
    offset = np.asarray([0.05 * (ALL_GROUPS.index(g) % 4) for g in race])
    p = np.clip(0.25 + 0.5 * y + offset + shift + rng.normal(0, 0.15, len(y)), 0.0, 1.0)
    sex = np.where(rng.random(len(y)) < 0.5, "Male", "Female").astype(object)
    ids = tuple(f"dataset_b-{i:09d}" for i in range(1, len(y) + 1))
    return ev.PartitionPredictions(ids, y, p, race, sex)


def _with_p(base: ev.PartitionPredictions, p: np.ndarray) -> ev.PartitionPredictions:
    return ev.PartitionPredictions(base.record_ids, base.y, p, base.race, base.sex)


def _indices(y: np.ndarray, seed: int = 0) -> np.ndarray:
    """Synthetic indices in the locked RQ1 layout: Denied columns first, then Approved."""
    rng = np.random.default_rng(seed)
    denied, approved = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    rows = [np.concatenate([rng.choice(denied, len(denied)), rng.choice(approved, len(approved))]) for _ in range(1000)]
    return np.asarray(rows, dtype=np.int32)


@pytest.fixture(scope="module")
def test_pair():
    control = _partition(1)
    rng = np.random.default_rng(2)
    adversarial = _with_p(control, np.clip(control.p_approved + rng.normal(0, 0.03, len(control.y)), 0, 1))
    return control, adversarial, _indices(control.y)


@pytest.fixture(scope="module")
def seed_result(test_pair):
    control, adversarial, indices = test_pair
    return ev.evaluate_seed(control, adversarial, control_threshold=0.5, adversarial_threshold=0.52, indices=indices)


# ---------------------------------------------------------------------------
# Compatibility, sign and threshold calibration
# ---------------------------------------------------------------------------


def test_reused_metric_semantics_are_compatible():
    assert all(ev.assert_metric_compatibility().values())


def test_delta_sign_is_adversarial_minus_control_not_rq4a_ba_cost():
    assert ev.delta_ba(0.70, 0.80) == pytest.approx(-0.10) and ev.delta_ba(0.80, 0.70) == pytest.approx(0.10)
    assert ev.delta_ba(0.70, 0.80) == -rq4a_metrics.ba_cost(0.80, 0.70)  # RQ4a: baseline - mitigated
    assert ev.delta(None, 0.5) is None and ev.delta(0.5, None) is None
    control = {"ba": 0.8, "eod3": rq4a_metrics.EODEstimate(0.20, True, 0.2, 0.1, ri.G3_GROUPS, (), "none"),
               "eod6": rq4a_metrics.EODEstimate(0.30, True, 0.3, 0.1, ri.G6_GROUPS, (), "none")}
    adversarial = {"ba": 0.79, "eod3": rq4a_metrics.EODEstimate(0.15, True, 0.15, 0.1, ri.G3_GROUPS, (), "none"),
                   "eod6": rq4a_metrics.EODEstimate(0.35, True, 0.35, 0.1, ri.G6_GROUPS, (), "none")}
    deltas = ev.matched_deltas(control, adversarial)
    assert deltas["delta_eod3"] == pytest.approx(-0.05) and deltas["delta_eod6"] == pytest.approx(0.05) and deltas["delta_ba"] == pytest.approx(-0.01)


def test_threshold_grid_and_ties():
    grid = ev.threshold_candidates()
    assert len(grid) == 999 and grid[0] == 0.001 and grid[-1] == 0.999 and grid[499] == 0.5
    y = np.array([0] * 50 + [1] * 50)
    plateau_with_half = ev.calibrate_threshold(y, np.array([0.2] * 50 + [0.8] * 50))
    assert plateau_with_half.threshold == 0.5 and plateau_with_half.balanced_accuracy == 1.0 and len(plateau_with_half.tied_thresholds) == 600
    above = ev.calibrate_threshold(y, np.array([0.6] * 50 + [0.9] * 50))
    assert above.threshold == 0.601  # nearest to 0.5 within the (0.6, 0.9] plateau
    below = ev.calibrate_threshold(y, np.array([0.1] * 50 + [0.3] * 50))
    assert below.threshold == 0.3  # p >= t: 0.3 itself still approves the positives
    assert ev.resolve_threshold_tie([0.4, 0.6]) == 0.4 and ev.resolve_threshold_tie([0.3, 0.7, 0.45, 0.55]) == 0.45
    assert np.array_equal(ev.decisions_at(np.array([0.7, 0.6999999]), 0.7), np.array([1, 0]))  # p >= t is approved
    with pytest.raises(ev.RQ4bEvaluationError):
        ev.calibrate_threshold(np.ones(5), np.full(5, 0.5))


# ---------------------------------------------------------------------------
# Candidate eligibility and lambda selection
# ---------------------------------------------------------------------------


def _cell(lam, seed, *, ba, eod3, eod6=0.5):
    return ev.CandidateEvaluation(float(lam), seed, ev.ThresholdSelection(0.5, ba if ba is not None else 0.0, (0.5,), 999), ba, eod3, eod6)


def _grid(ba=None, eod3=None, eod6=None, control_ba=0.01):
    """The exact sparse nine-unit roster; replicated adversarial BA is 0.0."""
    cells = []
    for lam, seed in ev.DEVELOPMENT_UNITS:
        if lam == 0.0:
            cells.append(_cell(lam, seed, ba=control_ba, eod3=(eod3 or {}).get(lam, 0.30), eod6=(eod6 or {}).get(lam, 0.5)))
        else:
            cells.append(_cell(lam, seed, ba=(ba or {}).get(lam, 0.0), eod3=(eod3 or {}).get(lam, 0.30 - lam / 10), eod6=(eod6 or {}).get(lam, 0.5)))
    return cells


def test_sparse_roster_is_recorded_but_selection_remains_blocked():
    result = ev.select_lambda(_grid())
    assert result["status"] == "blocked_pending_explicit_three_seed_methodological_approval"
    assert result["selected_lambda"] is None and result["arbitrary_two_of_three_rule_used"] is False
    assert [(row["adversary_lambda"], row["model_seed"]) for row in result["units"]] == list(ev.DEVELOPMENT_UNITS)
    replicated = {(row["adversary_lambda"], row["model_seed"]): row for row in result["units"]}
    assert replicated[(0.01, 43)]["matched_delta_ba"] == -0.01
    assert replicated[(0.03, 42)]["matched_control_available"] is True


def test_eod6_remains_audit_only_and_cannot_unlock_selection():
    result = ev.select_lambda(_grid(eod3={0.01: 0.01}, eod6={0.01: 0.0}))
    assert result["eod6_role"] == "audit_only"
    assert result["selected_lambda"] is None
    assert result["status"] == ev.SELECTION_POLICY_STATUS


def test_incomplete_or_cartesian_expanded_roster_is_rejected():
    with pytest.raises(ev.RQ4bEvaluationError, match="Sparse development roster incomplete"):
        ev.select_lambda(_grid()[:-1])
    expanded = [*_grid(), _cell(0.03, 43, ba=0.0, eod3=0.2)]
    with pytest.raises(ev.RQ4bEvaluationError, match="unexpected"):
        ev.select_lambda(expanded)


def test_final_arm_planning_is_blocked_until_three_seed_policy_is_approved():
    dev = {(lam, seed): ev.ArtifactRef(lam, seed, f"a{lam}_{seed}", "f" * 64, "e" * 64, 0.5) for lam, seed in ev.DEVELOPMENT_UNITS}
    with pytest.raises(ev.RQ4bEvaluationError, match="blocked pending"):
        ev.plan_final_arms(0.01, dev)


def test_development_reuse_identity_validation_remains_exact():
    dev = {(lam, seed): ev.ArtifactRef(lam, seed, f"a{lam}_{seed}", f"{seed:064d}", "e" * 64, 0.5) for lam, seed in ev.DEVELOPMENT_UNITS}
    ev.validate_reuse_identity(dev[(0.1, 42)], dev[(0.1, 42)])
    for changed in ({"model_sha256": "0" * 64}, {"threshold": 0.51}, {"artifact_id": "other"}, {"selection_record_sha256": "1" * 64}):
        import dataclasses

        with pytest.raises(ev.RQ4bEvaluationError):
            ev.validate_reuse_identity(dataclasses.replace(dev[(0.1, 42)], **changed), dev[(0.1, 42)])


# ---------------------------------------------------------------------------
# Test metrics, groups and denominators
# ---------------------------------------------------------------------------


def test_full_test_ba_includes_rows_outside_g3(test_pair):
    control, _, _ = test_pair
    outside = ~np.isin(control.race, ri.G3_GROUPS)
    p = control.p_approved.copy()
    p[outside & (control.y == 1)] = 0.0  # adversarial only worse outside G3
    adversarial = _with_p(control, p)
    c, a = ev.arm_point_metrics(control, 0.5), ev.arm_point_metrics(adversarial, 0.5)
    assert a["ba_g3_rows_audit_only"] == c["ba_g3_rows_audit_only"]
    deltas = ev.matched_deltas(ev.core_quantities(control, 0.5), ev.core_quantities(adversarial, 0.5))
    assert deltas["delta_ba"] < -0.1 and deltas["delta_ba"] == a["ba"] - c["ba"]
    assert deltas["delta_eod3"] == 0.0


def test_missing_roster_group_invalidates_without_shrinking(test_pair):
    groups = tuple(g for g in ALL_GROUPS if g != "Asian")
    control = _partition(3, groups=groups)
    adversarial = _with_p(control, np.clip(control.p_approved - 0.01, 0, 1))
    core = ev.core_quantities(control, 0.5)
    assert not core["eod3"].valid and core["eod3"].invalid_groups == ("Asian",) and core["eod3"].roster == ri.G3_GROUPS
    result = ev.evaluate_seed(control, adversarial, control_threshold=0.5, adversarial_threshold=0.5, indices=_indices(control.y))
    assert result["deltas"]["delta_eod3"] is None and result["deltas"]["undefined_reasons"]["delta_eod3"] == ev.REASON_GROUP_INVALID
    assert result["point_criterion"]["status"] == "not_evaluable"
    boot = result["bootstrap"]["delta_eod3"]
    assert boot["evaluable"] is False and boot["valid_replicates"] == 0 and len(boot["invalid_replicates"]) == 1000
    assert result["strong_support"]["status"] == "not_evaluable" and result["strong_support"]["supports"] is False
    assert result["control"]["group_coverage"]["Asian"] == 0


def test_group_denominator_failure_invalidates_only_the_affected_roster():
    control = _partition(4, denied_override={"2 or more minority races": 20})
    core = ev.core_quantities(control, 0.5)
    assert core["eod3"].valid and not core["eod6"].valid and core["eod6"].invalid_groups == ("2 or more minority races",)
    metrics = ev.arm_point_metrics(control, 0.5)
    fpr = metrics["per_group"]["2 or more minority races"]["fpr_approved"]
    assert fpr["value"] is None and fpr["status"] == "undefined_insufficient_actual_denied" and fpr["actual_denied"] == 20


def test_required_audits_are_reported(seed_result):
    arm = seed_result["control"]
    for key in ("denied_f1", "denied_precision", "denied_recall", "sex_eod_safety_audit", "extrema_groups", "group_coverage", "per_group"):
        assert key in arm
    assert set(arm["per_group"]) == set(ALL_GROUPS) and arm["sex_eod_safety_audit"]["roster"] == ["Male", "Female"]
    changes = seed_result["decision_changes"]
    assert changes[0]["scope"] == "overall" and [row["group"] for row in changes[1:]] == list(ALL_GROUPS)


# ---------------------------------------------------------------------------
# Locked-index bootstrap and CIs
# ---------------------------------------------------------------------------


def test_locked_index_validation(test_pair):
    control, _, indices = test_pair
    ev.validate_locked_indices(indices, control.y)
    for bad in (indices.astype(np.int64), indices[:999], indices[:, :-1], np.where(indices == 0, len(control.y), indices).astype(np.int32)):
        with pytest.raises(ev.RQ4bEvaluationError):
            ev.validate_locked_indices(bad, control.y)
    swapped = indices.copy()
    swapped[:, [0, -1]] = swapped[:, [-1, 0]]  # breaks the Denied-first stratified layout
    with pytest.raises(ev.RQ4bEvaluationError, match="stratification"):
        ev.validate_locked_indices(swapped, control.y)


def test_bootstrap_is_paired_on_identical_rows(test_pair):
    control, _, indices = test_pair
    same = ev.paired_bootstrap(control, control, control_threshold=0.5, adversarial_threshold=0.5, indices=indices)
    for name in ev.DELTA_QUANTITIES:
        assert same[name]["valid_replicates"] == 1000 and set(same[name]["values"]) == {0.0}
        assert (same[name]["lower"], same[name]["upper"]) == (0.0, 0.0)
    reordered = ev.PartitionPredictions(control.record_ids[::-1], control.y[::-1], control.p_approved[::-1], control.race[::-1], control.sex[::-1])
    with pytest.raises(ev.RQ4bEvaluationError, match="different ordered record IDs"):
        ev.paired_bootstrap(control, reordered, control_threshold=0.5, adversarial_threshold=0.5, indices=indices)


def test_bootstrap_ci_matches_linear_quantiles_and_point_values(test_pair, seed_result):
    control, adversarial, indices = test_pair
    for name in ev.DELTA_QUANTITIES:
        boot = seed_result["bootstrap"][name]
        assert boot["valid_replicates"] == 1000 and boot["evaluable"]
        lower, upper = np.quantile(np.asarray(boot["values"]), [0.025, 0.975], method="linear")
        assert (boot["lower"], boot["upper"]) == (lower, upper)
    rows = indices[17]
    c = ev._core(control.y[rows], control.p_approved[rows], control.race.astype(str)[rows], 0.5)
    a = ev._core(adversarial.y[rows], adversarial.p_approved[rows], adversarial.race.astype(str)[rows], 0.52)
    assert seed_result["bootstrap"]["delta_ba"]["values"][17] == a["ba"] - c["ba"]
    point = ev.matched_deltas(ev.core_quantities(control, 0.5), ev.core_quantities(adversarial, 0.52))
    assert seed_result["deltas"]["delta_eod3"] == point["delta_eod3"]


def test_invalid_replicates_make_ci_not_evaluable_without_replacement():
    control = _partition(5, denied_override={"Asian": 30})  # exactly at the gate: resampling drops some replicates below 30
    adversarial = _with_p(control, np.clip(control.p_approved + 0.01, 0, 1))
    boot = ev.paired_bootstrap(control, adversarial, control_threshold=0.5, adversarial_threshold=0.5, indices=_indices(control.y, seed=3))
    eod3 = boot["delta_eod3"]
    assert 0 < len(eod3["invalid_replicates"]) < 1000 and eod3["valid_replicates"] == 1000 - len(eod3["invalid_replicates"])
    assert eod3["evaluable"] is False and eod3["lower"] is None and eod3["reason"] == ev.REASON_INSUFFICIENT
    assert all(item["reason"] == ev.REASON_GROUP_INVALID for item in eod3["invalid_replicates"])
    assert len(eod3["values"]) == 1000 and sum(v is None for v in eod3["values"]) == len(eod3["invalid_replicates"])
    assert boot["delta_ba"]["evaluable"] is True  # validity is per quantity


# ---------------------------------------------------------------------------
# Criteria, strong support and three-seed summaries
# ---------------------------------------------------------------------------


def test_point_ba_is_non_strict_and_ci_ba_is_strict():
    assert ev.point_criterion({"delta_eod3": -0.001, "delta_ba": -0.01})["status"] == "pass"
    assert ev.point_criterion({"delta_eod3": 0.0, "delta_ba": 0.0})["status"] == "fail"  # strict fairness
    assert ev.point_criterion({"delta_eod3": -0.1, "delta_ba": -0.010000000000000002})["status"] == "fail"
    assert ev.point_criterion({"delta_eod3": None, "delta_ba": 0.0})["status"] == "not_evaluable"
    ok = {"evaluable": True, "upper": -1e-12, "lower": -1.0}
    assert ev.strong_support_seed({"delta_eod3": ok, "delta_ba": {"evaluable": True, "lower": -0.01, "upper": 0.0}})["supports"] is False
    assert ev.strong_support_seed({"delta_eod3": ok, "delta_ba": {"evaluable": True, "lower": -0.009999999999999998, "upper": 0.0}})["supports"] is True
    assert ev.strong_support_seed({"delta_eod3": {"evaluable": True, "upper": 0.0, "lower": -1.0}, "delta_ba": {"evaluable": True, "lower": 0.0}})["supports"] is False
    assert ev.strong_support_seed({"delta_eod3": {"evaluable": False}, "delta_ba": {"evaluable": True, "lower": 0.0}})["status"] == "not_evaluable"


def _seed_record(d_eod3, d_ba, supports, *, evaluable=True, boot_values=None):
    status = "evaluated" if evaluable else "not_evaluable"
    return {"deltas": {"delta_eod3": d_eod3, "delta_ba": d_ba, "delta_eod6": 0.0}, "point_criterion": ev.point_criterion({"delta_eod3": d_eod3, "delta_ba": d_ba}),
            "strong_support": {"status": status, "supports": supports}, "bootstrap": {"delta_eod3": {"values": boot_values or []}}}


def test_strong_support_stays_pending_without_an_approved_three_seed_rule():
    per_seed = {42: _seed_record(-0.05, 0.0, True), 43: _seed_record(-0.04, 0.0, True),
                44: _seed_record(-0.03, 0.0, False, evaluable=False)}
    result = ev.assess_final_seeds(per_seed)["strong_support"]
    assert result["status"] == ev.SELECTION_POLICY_STATUS
    assert result["supported_count"] == 2 and result["not_evaluable_seeds"] == [44]
    assert result["required_count"] is None and result["final_seed_count"] == 3
    assert result["joint_confidence_claim"] is False and result["arbitrary_two_of_three_rule_used"] is False
    del per_seed[44]
    with pytest.raises(ev.RQ4bEvaluationError, match="three amended seeds"):
        ev.assess_final_seeds(per_seed)


def test_across_seed_summary_uses_ddof1_without_pooling_or_dropping():
    values = [-0.05, -0.04, -0.03]
    per_seed = {seed: _seed_record(v, 0.0, True, boot_values=[v] * 1000) for seed, v in zip(ev.FINAL_SEEDS, values)}
    summary = ev.across_seed_summary(per_seed, "delta_eod3")
    assert summary["standard_deviation"] == float(np.std(values, ddof=1)) != float(np.std(values, ddof=0)) and summary["ddof"] == 1
    assert summary["mean"] == float(np.mean(values)) and summary["range"] == max(values) - min(values) and summary["pooled_bootstrap_replicates"] is False
    per_seed[44]["bootstrap"]["delta_eod3"]["values"] = [9.9] * 1000
    assert ev.across_seed_summary(per_seed, "delta_eod3")["mean"] == summary["mean"]  # bootstrap rows never pooled
    per_seed[43]["deltas"]["delta_eod3"] = None
    missing = ev.across_seed_summary(per_seed, "delta_eod3")
    assert missing["status"] == "not_evaluable" and missing["missing_seeds"] == [43]


def test_assessment_keeps_scientific_fields_separate(seed_result):
    per_seed = {seed: seed_result for seed in ev.FINAL_SEEDS}
    assessment = ev.assess_final_seeds(per_seed)
    assert assessment["stability_status"] == "not_assessed" and assessment["strong_support_implies_stable"] is False
    assert "execution_status" not in assessment and "not causal" in assessment["interpretation_boundary"]
    assert ev.fairness_generalization_gap(0.12, 0.10) == pytest.approx(0.02) and ev.fairness_generalization_gap(None, 0.1) is None


def test_inputs_are_validated():
    base = _partition(6, per_group=4)
    with pytest.raises(ev.RQ4bEvaluationError):
        ev.PartitionPredictions(base.record_ids, base.y, np.full(len(base.y), 1.5), base.race, base.sex)
    with pytest.raises(ev.RQ4bEvaluationError):
        ev.PartitionPredictions(base.record_ids, base.y, base.p_approved, np.full(len(base.y), "Martian", dtype=object), base.sex)
    with pytest.raises(ev.RQ4bEvaluationError):
        ev.evaluate_candidate(base, adversary_lambda=0.2, model_seed=42)
    with pytest.raises(ev.RQ4bEvaluationError):
        ev.evaluate_candidate(base, adversary_lambda=0.1, model_seed=45)


# ---------------------------------------------------------------------------
# Sex metadata representation (T4-R1)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rq3_protocol():
    return yaml.safe_load((PROJECT_ROOT / "config" / "dataset_b" / "rq3_protocol.yaml").read_text(encoding="utf-8"))


def _with_sex(base: ev.PartitionPredictions, sex) -> ev.PartitionPredictions:
    return ev.PartitionPredictions(base.record_ids, base.y, base.p_approved, base.race, np.asarray(sex, dtype=object))


def test_sex_representation_matches_rq3_mapping(rq3_protocol):
    assert ev.assert_rq3_sex_mapping(rq3_protocol) == {
        "1": "Male", "2": "Female", "3": "Information not provided", "4": "Not applicable", "6": "Applicant selected both male and female"}
    assert ev.SEX_ROSTER == ("Male", "Female") and len(ev.SEX_LABELS) == 5
    changed = yaml.safe_load(yaml.safe_dump(rq3_protocol))
    changed["attributes"]["applicant_sex"]["coverage_only_groups"]["Information not provided"] = "5"
    with pytest.raises(ev.RQ4bEvaluationError, match="mapping differs"):
        ev.assert_rq3_sex_mapping(changed)


@pytest.mark.parametrize("raw", [["1", "2"], ["2"], ["3"], ["6"]])
def test_raw_rq3_sex_codes_are_rejected(raw):
    base = _partition(7, per_group=4)
    sex = np.asarray(["Male"] * len(base.y), dtype=object)
    sex[: len(raw)] = raw
    with pytest.raises(ev.RQ4bEvaluationError, match="raw RQ3 applicant_sex codes"):
        _with_sex(base, sex)


def test_canonical_and_coverage_only_sex_labels_are_accepted(test_pair):
    control, _, _ = test_pair
    sex = np.asarray(control.sex, dtype=object).copy()
    sex[:30] = "Information not provided"
    sex[30:45] = "Not applicable"
    sex[45:60] = "Applicant selected both male and female"
    arm = ev.arm_point_metrics(_with_sex(control, sex), 0.5)
    assert arm["sex_group_coverage"]["Information not provided"] == 30 and arm["sex_group_coverage"]["Not applicable"] == 15
    assert arm["sex_group_coverage"]["Applicant selected both male and female"] == 15
    assert sum(arm["sex_group_coverage"].values()) == len(sex)  # every row keeps a category
    assert arm["sex_eod_safety_audit"]["roster"] == ["Male", "Female"] and arm["sex_eod_safety_audit"]["valid"]


@pytest.mark.parametrize(
    "bad, message",
    [("Unknown", "outside the RQ3 mapping"), ("male", "outside the RQ3 mapping"), ("", "blank"), (" ", "blank"), (" Male", "blank"),
     (None, "missing"), (float("nan"), "missing"), (1, "canonical RQ3 label strings")],
)
def test_unknown_blank_and_missing_sex_values_are_rejected(bad, message):
    base = _partition(8, per_group=4)
    sex = np.asarray(["Female"] * len(base.y), dtype=object)
    sex[3] = bad
    with pytest.raises(ev.RQ4bEvaluationError, match=message):
        _with_sex(base, sex)


def test_explicit_normalization_of_approved_codes(rq3_protocol):
    labels = ev.normalize_rq3_sex_codes(["1", "2", "3", "4", "6"], rq3_protocol)
    assert labels.tolist() == list(ev.SEX_LABELS)
    for bad in (["5"], ["1", ""], ["1 "], [None], [1]):
        with pytest.raises(ev.RQ4bEvaluationError):
            ev.normalize_rq3_sex_codes(bad, rq3_protocol)
    base = _partition(9, per_group=4)
    codes = np.where(np.arange(len(base.y)) % 2 == 0, "1", "2")
    normalized = _with_sex(base, ev.normalize_rq3_sex_codes(codes.tolist(), rq3_protocol))
    assert set(normalized.sex) == {"Male", "Female"}


def test_absent_or_insufficient_sex_groups_stay_distinct_from_invalid_encoding(test_pair):
    control, _, _ = test_pair
    only_male = _with_sex(control, ["Male"] * len(control.y))  # valid encoding, Female genuinely absent
    absent = ev.arm_point_metrics(only_male, 0.5)["sex_eod_safety_audit"]
    assert absent["valid"] is False and absent["invalid_groups"] == ["Female"] and absent["reason_code"] == ev.REASON_GROUP_INVALID
    sex = np.asarray(["Male"] * len(control.y), dtype=object)
    female_rows = np.flatnonzero(control.y == 0)[:20]  # 20 Female rows, all denied: 0 approved (< 30) and 20 denied (< 30)
    sex[female_rows] = "Female"
    insufficient = ev.arm_point_metrics(_with_sex(control, sex), 0.5)
    assert insufficient["sex_eod_safety_audit"]["valid"] is False and insufficient["sex_eod_safety_audit"]["invalid_groups"] == ["Female"]
    assert insufficient["sex_groups"]["Female"]["status"] == "undefined_insufficient_actual_approved"
    assert insufficient["eod3"]["valid"] and insufficient["ba"] is not None  # primary metrics unaffected by the sex audit
