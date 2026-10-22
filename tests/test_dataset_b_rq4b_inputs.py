"""Synthetic tests for the RQ4b production contract and input boundary.

No project Validation/Test data, frozen artifact, model fit or GPU is used.
Only the two YAML configuration files are read.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.dataset_b.mitigation import rq4b_inputs as ri
from src.dataset_b.preprocessing import split_inputs
from src.dataset_b.preprocessing.categorical_encoder import (
    CATEGORICAL_COLUMNS,
    CATEGORICAL_SCHEMA,
    NUMERIC_VALUE_COLUMNS,
    TWO_CHANNEL_COLUMNS,
)
from src.utils.path import PROJECT_ROOT


RQ3_PATH = PROJECT_ROOT / "config" / "dataset_b" / "rq3_protocol.yaml"


@pytest.fixture(scope="module")
def raw_config() -> dict:
    return yaml.safe_load(ri.RQ4B_CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rq3_protocol() -> dict:
    return yaml.safe_load(RQ3_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def contract() -> ri.RQ4bContract:
    return ri.load_contract()


def _ids(n: int, start: int = 1) -> list[str]:
    return [f"dataset_b-{index:09d}" for index in range(start, start + n)]


def _native_frame(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {}
    for column in TWO_CHANNEL_COLUMNS:
        if column in NUMERIC_VALUE_COLUMNS:
            data[column] = rng.normal(size=n).astype(np.float32)
        else:
            categories = list(CATEGORICAL_SCHEMA[column])
            data[column] = pd.Categorical.from_codes(rng.integers(0, len(categories), size=n), categories=categories)
    return pd.DataFrame(data, columns=list(TWO_CHANNEL_COLUMNS))


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_real_contract_validates_locked_fields(contract):
    assert contract.protocol_version == ri.PROTOCOL_VERSION
    assert contract.decision_revision == ri.DECISION_REVISION
    assert contract.track_a_source_run_id == ri.TRACK_A_SOURCE_RUN_ID == "rq4b3_adv_20261019_02"
    assert contract.track_b_child_run_id == ri.TRACK_B_CHILD_RUN_ID == "rq4b3_adv_20261021_01"
    assert contract.production_run_id == ri.SEED44_CONTINUATION_RUN_ID == "rq4b3_adv_20261022_01"
    assert contract.production_run_role == ri.SEED44_CONTINUATION_ROLE == "track_b_seed44_continuation"
    assert contract.g3 == ("White", "Black or African American", "Asian")
    assert contract.lambda_grid == (0.0, 0.01, 0.03, 0.1, 0.3)
    assert contract.development_seeds == (42, 43, 44)
    assert contract.final_seeds == (42, 43, 44)
    assert contract.retained_development_units == ri.RETAINED_DEVELOPMENT_UNITS
    assert contract.additional_development_units == ri.ADDITIONAL_DEVELOPMENT_UNITS
    assert contract.development_units == ri.DEVELOPMENT_UNITS
    assert contract.authorized_production_units == ri.SEED44_PRODUCTION_UNITS == ((0.0, 44), (0.01, 44))
    assert set(contract.source_run_evidence) == {"track_a_seed42", "track_b_seed43"}
    assert contract.selection_policy_status == "blocked_pending_explicit_three_seed_methodological_approval"
    assert (contract.cv_folds, contract.cv_random_state, contract.epochs_per_fold) == (5, 42, 256)
    assert (contract.batch_size, contract.predict_batch_size, contract.device) == (256, 1024, "cuda:0")
    assert contract.resources["frozen_probe_validation_predictions"] == PROJECT_ROOT / ri.PROBE_PREDICTION_PATH
    assert contract.resources["derived_race_source"] == PROJECT_ROOT / "data" / "raw" / "state_NY_2024.csv"


def _set(config: dict, path: tuple[str, ...], value) -> dict:
    mutated = copy.deepcopy(config)
    node = mutated
    for key in path[:-1]:
        node = node[key]
    if value is KeyError:
        del node[path[-1]]
    else:
        node[path[-1]] = value
    return mutated


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("protocol_version",), "rq4b_adversarial_realmlp_v1"),
        (("decision_revision",), KeyError),
        (("contract_amendment", "track_a_source_run_id"), "rq4b3_adv_20261021_01"),
        (("contract_amendment", "track_b_child_run_id"), "rq4b3_adv_20261019_02"),
        (("contract_amendment", "execution_lineage", "track_a_source_is_immutable"), False),
        (("contract_amendment", "execution_lineage", "cross_run_aggregation_status"), "approved"),
        (("contract_amendment", "execution_lineage", "prohibit_resume_across_code_change"), False),
        (("seed44_continuation", "run_id"), "rq4b3_adv_20261021_01"),
        (("seed44_continuation", "source_runs", "track_b_seed43", "manifest_sha256"), "0" * 64),
        (("seed44_continuation", "fitted_state_reuse"), "allowed"),
        (("seed44_continuation", "cross_run_aggregation_status"), "approved"),
        (("groups", "g3"), ["Asian", "White", "Black or African American"]),
        (("realmlp_recipe", "lambda_grid"), [0, 0.01, 0.03, 0.1, 0.3, 3.0]),
        (("realmlp_recipe", "development_seeds"), [42, 43]),
        (("development_units", "maximum_total"), 10),
        (("development_units", "additional_approved"), [{"lambda": 0, "seed": 43}]),
        (("development_units", "prohibit_cartesian_expansion"), False),
        (("realmlp_recipe", "predictor_features_exclude"), ["derived_race"]),
        (("realmlp_recipe", "preserve_native_loader_drop_last"), 1),
        (("realmlp_recipe", "adversary", "production_architecture", "hidden_sizes"), [64]),
        (("realmlp_recipe", "adversary", "optimizer", "learning_rate"), 0.01),
        (("realmlp_recipe", "adversary", "predictor_total_loss"), "L_prediction + lambda * L_adversary"),
        (("epoch_selection", "use_early_stopping"), True),
        (("epoch_selection", "exact_tie_break"), "latest_epoch"),
        (("epoch_selection", "splitter", "random_state"), 0),
        (("epoch_selection", "full_train_refit", "schedule_horizon", "compress_schedule_to_selected_epoch"), True),
        (("capability_assessment", "real_probe", "fitting", "patience"), 10),
        (("capability_assessment", "real_probe", "source_score_extraction", "probability_sum_check", "atol"), 1e-3),
        (("capability_assessment", "real_probe", "source_score_extraction", "prediction_path"), "artifacts/other.csv"),
        (("threshold_calibration", "candidates", "count"), 1000),
        (("runtime", "allow_silent_cpu_fallback"), True),
        (("runtime", "device"), "cuda"),
        (("fit_budget", "rq4b_3_maximum_cv_fold_fits_excluding_spike_and_retries"), 120),
        (("uncertainty", "within_seed", "allow_regenerate_indices"), True),
        (("configurations", "RQ4b-3", "group_roster"), "g6"),
    ],
)
def test_malformed_or_conflicting_config_is_rejected(raw_config, rq3_protocol, path, value):
    with pytest.raises(ri.RQ4bInputError, match="locked contract"):
        ri.validate_contract(_set(raw_config, path, value), rq3_protocol)


def test_non_mapping_config_is_rejected(rq3_protocol):
    with pytest.raises(ri.RQ4bInputError):
        ri.validate_contract(["not", "a", "mapping"], rq3_protocol)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("execution_governance", "production_launch_authorized"), False, "explicitly authorized"),
        (("execution_governance", "production_launch_scope", "run_id"), "rq4b3_adv_20261021_01", "continuation run"),
        (("execution_governance", "production_launch_scope", "role"), "track_b_child", "continuation role"),
        (("execution_governance", "production_launch_scope", "units"), [{"lambda": 0, "seed": 44}], "two authorized seed-44"),
        (("execution_governance", "production_launch_scope", "require_preflight_and_capability_pass"), False, "preflight and capability"),
        (("execution_governance", "production_launch_scope", "prohibit_automatic_retry"), False, "Automatic retry"),
        (("execution_governance", "production_launch_scope", "require_lambda_zero_artifact_verification_before_lambda_0_01"), False, "Lambda 0 artifacts"),
        (("execution_governance", "production_launch_scope", "selection_final_and_test_remain_closed"), False, "remain closed"),
    ],
)
def test_scoped_production_authorization_is_enforced(raw_config, rq3_protocol, path, value, message):
    with pytest.raises(ri.RQ4bInputError, match=message):
        ri.validate_contract(_set(raw_config, path, value), rq3_protocol)


def test_conflicting_rq3_race_groups_are_rejected(raw_config, rq3_protocol):
    mutated = copy.deepcopy(rq3_protocol)
    mutated["attributes"]["derived_race"]["secondary_comparisons"] = ["Joint"]
    with pytest.raises(ri.RQ4bInputError, match="G6"):
        ri.validate_contract(raw_config, mutated)


def test_conflicting_rq3_raw_source_is_rejected(raw_config, rq3_protocol):
    mutated = copy.deepcopy(rq3_protocol)
    mutated["audit_data"]["raw_source"] = "data/raw/other.csv"
    with pytest.raises(ri.RQ4bInputError, match="raw source"):
        ri.validate_contract(raw_config, mutated)


# C1 finding 1: locked decisions previously outside the validator, plus analogous rules.
C1_LOCKED_RULES = [
    (("scientific_assessment", "strong_support", "fairness_condition"), "upper_95_ci_delta_eod3_less_than_or_equal_to_zero"),
    (("scientific_assessment", "strong_support", "utility_condition"), "lower_95_ci_delta_ba_greater_than_or_equal_to_minus_0_01"),
    (("capability_assessment", "synthetic_correctness_gate", "lower_95_ci_gain"), "greater_than_or_equal_to_zero"),
    (("epoch_selection", "numerical_computation"), "clipped_probability_ce"),
    (("denominators", "minimum_actual_approved"), 20),
    # analogous execution-critical rules
    (("denominators", "minimum_actual_denied"), 20),
    (("denominators", "minimum_group_rows"), 50),
    (("denominators", "insufficient_result"), 0),
    (("denominators", "never_replace_undefined_with_zero"), False),
    (("scientific_assessment", "per_seed_point_criterion", "fairness"), "delta_eod3_less_than_or_equal_to_zero"),
    (("scientific_assessment", "per_seed_point_criterion", "utility"), "full_test_delta_balanced_accuracy_greater_than_minus_0_02"),
    (("scientific_assessment", "per_seed_point_criterion", "combination"), "either_required"),
    (("scientific_assessment", "strong_support", "same_seed_must_satisfy_both_ci_conditions"), False),
        (("scientific_assessment", "strong_support", "final_seed_count"), 5),
    (("capability_assessment", "synthetic_correctness_gate", "ce_gain_comparison"), "greater_than"),
    (("capability_assessment", "synthetic_correctness_gate", "fixture_source_and_generation", "stratify_by"), ["y", "group"]),
    (("capability_assessment", "probe_uncertainty", "strata"), ["y_approved"]),
    (("capability_assessment", "probe_uncertainty", "confidence_level"), 0.9),
    (("capability_assessment", "probe_uncertainty", "round_before_quantiles_or_endpoint_comparison"), True),
    (("capability_assessment", "real_probe", "probe_fitted_on"), "probe_train_and_validation"),
    (("capability_assessment", "real_probe", "gain_and_intervals_reported_on"), "probe_validation_only"),
    (("epoch_selection", "monitor"), "weighted_binary_prediction_cross_entropy"),
    (("epoch_selection", "probability_clipping"), "allowed"),
    (("epoch_selection", "fairness_metrics_role"), "selection"),
    (("realmlp_recipe", "predictor_population_semantics"), "every_row_every_epoch"),
    (("realmlp_recipe", "adversary", "gradient_reversal_equivalent_allowed"), False),
    (("metrics", "eod_definition"), "mean_fixed_roster_tpr_gap"),
    (("metrics", "ba_budget_primary"), 0.02),
    (("metrics", "primary_fairness"), ["eod6"]),
    (("uncertainty", "within_seed", "confidence_level"), 0.9),
    (("uncertainty", "within_seed", "stratification"), "race_y"),
        (("uncertainty", "across_seed", "required_seed_count"), 5),
    (("uncertainty", "select_seed_from_test"), True),
    (("candidate_selection", "eligibility_metric"), "mean_matched_delta_roc_auc"),
    (("runtime", "compatibility_baseline", "torch"), "2.15.0"),
    (("runtime", "preflight_required"), ["cuda_available"]),
    (("groups", "invalid_required_group_metric_invalidates_eod6_replicate"), False),
    (("internal_extension_and_naming_gate", "if_recipe_not_preserved"), "rename_optional"),
    (("status_contract", "primary_criterion"), ["pass", "fail"]),
    (("reporting_contract", "prohibit_best_seed_selection"), False),
    (("prohibitions", "prohibit_causal_claim"), False),
    (("prerequisites", "rq1_paired_bootstrap_task"), "optional"),
    (("manuscript_equivalent",), "RQ3a"),
]


@pytest.mark.parametrize(("path", "value"), C1_LOCKED_RULES)
def test_conflicting_locked_rule_is_rejected(raw_config, rq3_protocol, path, value):
    with pytest.raises(ri.RQ4bInputError, match=".".join(path)):
        ri.validate_contract(_set(raw_config, path, value), rq3_protocol)


@pytest.mark.parametrize(("path", "_value"), C1_LOCKED_RULES)
def test_missing_locked_rule_is_rejected(raw_config, rq3_protocol, path, _value):
    with pytest.raises(ri.RQ4bInputError, match=r"\(missing\)"):
        ri.validate_contract(_set(raw_config, path, KeyError), rq3_protocol)


@pytest.mark.parametrize(
    "path",
    [("scientific_assessment", "strong_support", "extra_condition"), ("epoch_selection", "extra_rule"), ("new_section",)],
)
def test_unexpected_field_in_locked_contract_is_rejected(raw_config, rq3_protocol, path):
    with pytest.raises(ri.RQ4bInputError, match=r"\(unexpected\)"):
        ri.validate_contract(_set(raw_config, path, "x"), rq3_protocol)


def test_type_aware_comparison_rejects_bool_number_and_string_number_swaps(raw_config, rq3_protocol):
    for path, value in [
        (("denominators", "minimum_actual_approved"), "30"),
        (("denominators", "minimum_actual_approved"), True),
        (("scientific_assessment", "strong_support", "invalid_ci_counts_as_support"), 0),
        (("groups", "g6"), list(reversed(ri.G6_GROUPS))),
    ]:
        with pytest.raises(ri.RQ4bInputError, match="locked contract"):
            ri.validate_contract(_set(raw_config, path, value), rq3_protocol)


def test_unlocked_status_fields_may_change_without_rejection(raw_config, rq3_protocol):
    original_status = copy.deepcopy(raw_config["status"])
    mutated = copy.deepcopy(raw_config)
    mutated["status"] = "production_implemented"
    mutated["prerequisites"]["implementation_in_this_task"] = True
    mutated["capability_assessment"]["synthetic_correctness_gate"]["fixture_source_and_generation"]["observed_gate_result"] = "pass"
    mutated["configurations"]["RQ4b-3"]["readiness"] = "launched"
    mutated["technical_spike"]["codex_acceptance"]["status"] = "superseded"
    ri.validate_contract(mutated, rq3_protocol)
    assert raw_config["status"] == original_status  # caller mapping not mutated


def test_approved_snapshot_matches_module_constants_and_current_yaml(raw_config):
    snapshot = ri.APPROVED_CONTRACT
    assert ri.locked_contract_conflicts(raw_config) == []
    assert tuple(snapshot["groups"]["g3"]) == ri.G3_GROUPS and tuple(snapshot["groups"]["g6"]) == ri.G6_GROUPS
    assert tuple(snapshot["realmlp_recipe"]["lambda_grid"]) == ri.LAMBDA_GRID
    assert tuple(snapshot["realmlp_recipe"]["final_seeds"]) == ri.FINAL_SEEDS
    assert snapshot["capability_assessment"]["real_probe"]["source_score_extraction"]["prediction_path"] == ri.PROBE_PREDICTION_PATH
    assert snapshot["scientific_assessment"]["strong_support"]["fairness_condition"] == "upper_95_ci_delta_eod3_strictly_less_than_zero"
    assert snapshot["denominators"]["minimum_actual_approved"] == 30


# ---------------------------------------------------------------------------
# IDs, labels and partitions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ids", "message"),
    [
        (["dataset_b-000000001", "dataset_b-000000001"], "unique"),
        (["dataset_b-000000001", ""], "nonblank"),
        (["dataset_b-000000001", " dataset_b-000000002"], "nonblank"),
        (["dataset_b-000000001", "SYN-2"], "pattern"),
        (["dataset_b-000000001", 2], "nonblank"),
    ],
)
def test_invalid_record_ids_are_rejected(ids, message):
    with pytest.raises(ri.RQ4bInputError, match=message):
        ri.validate_record_ids(ids, "fixture")


def test_join_rejects_missing_extra_and_duplicate_ids():
    expected = _ids(4)
    frame = pd.DataFrame({"record_id": expected, "value": range(4)})
    with pytest.raises(ri.RQ4bInputError, match="missing"):
        ri.align_by_record_id(expected, frame.iloc[:3], "fixture", require_same_order=False)
    extra = pd.concat([frame, pd.DataFrame({"record_id": _ids(1, 99), "value": [9]})])
    with pytest.raises(ri.RQ4bInputError, match="unexpected"):
        ri.align_by_record_id(expected, extra, "fixture", require_same_order=False)
    duplicate = pd.concat([frame, frame.iloc[[0]]])
    with pytest.raises(ri.RQ4bInputError, match="unique"):
        ri.align_by_record_id(expected, duplicate, "fixture", require_same_order=False)


def test_row_permutation_rejected_when_order_is_locked_and_realigned_otherwise():
    expected = _ids(5)
    frame = pd.DataFrame({"record_id": expected, "y_approved": [1, 0, 1, 1, 0]})
    permuted = frame.iloc[[3, 0, 4, 2, 1]]
    with pytest.raises(ri.RQ4bInputError, match="order"):
        ri.align_by_record_id(expected, permuted, "fixture", require_same_order=True)
    aligned = ri.align_by_record_id(expected, permuted, "fixture", require_same_order=False)
    assert aligned["record_id"].tolist() == expected
    assert aligned["y_approved"].tolist() == [1, 0, 1, 1, 0]


def test_label_mismatch_and_nonbinary_labels_are_rejected():
    ri.require_same_labels([1, 0, 1], ["1", "0", "1"], "fixture")
    with pytest.raises(ri.RQ4bInputError, match="differs"):
        ri.require_same_labels([1, 0, 1], [1, 1, 1], "fixture")
    with pytest.raises(ri.RQ4bInputError, match="binary"):
        ri.validate_binary_labels([0, 2], "fixture")
    with pytest.raises(ri.RQ4bInputError, match="binary"):
        ri.validate_binary_labels([0, None], "fixture")


def test_partition_overlap_count_and_name_checks():
    counts = ri.validate_partition_membership({"train": _ids(3), "validation": _ids(2, 10)}, expected_rows={"train": 3})
    assert counts == {"train": 3, "validation": 2}
    with pytest.raises(ri.RQ4bInputError, match="another partition"):
        ri.validate_partition_membership({"train": _ids(3), "test": _ids(2, 3)})
    with pytest.raises(ri.RQ4bInputError, match="expected"):
        ri.validate_partition_membership({"train": _ids(3)}, expected_rows={"train": 4})
    with pytest.raises(ri.RQ4bInputError, match="Unknown"):
        ri.validate_partition_membership({"holdout": _ids(1)})
    with pytest.raises(ri.RQ4bInputError, match="outside"):
        ri.require_ids_within(_ids(2, 50), _ids(10), "fold")


# ---------------------------------------------------------------------------
# Roster coding and derived_race join
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "roster",
    [
        ri.G6_GROUPS,
        ("Black or African American", "White", "Asian"),
        ("White", "Asian"),
        ("White", "Black or African American", "Asian", "Joint"),
    ],
)
def test_wrong_roster_is_rejected(contract, roster):
    with pytest.raises(ri.RQ4bInputError, match="roster"):
        ri.encode_roster(["White"], roster, contract)


def test_roster_codes_and_outside_mask(contract):
    race = ["White", "Asian", "Joint", "Black or African American", "Race Not Available"]
    codes = ri.encode_roster(race, contract.g3, contract)
    assert codes.tolist() == [0, 2, -1, 1, -1]
    assert ri.roster_mask(codes).tolist() == [True, True, False, True, False]
    with pytest.raises(ri.RQ4bInputError, match="outside the contract"):
        ri.encode_roster(["White", "white"], contract.g3, contract)
    with pytest.raises(ri.RQ4bInputError, match="nonblank"):
        ri.encode_roster(["White", ""], contract.g3, contract)
    with pytest.raises(ri.RQ4bInputError):
        ri.roster_mask(np.asarray([0, 3]))


def _raw_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "action_taken": ["1", "3", "2", "3", "1"],
            "derived_race": ["White", "Joint", "Asian", "Black or African American", "Race Not Available"],
            "lei": ["A", "B", "C", "D", "E"],
        },
        index=[0, 1, 2, 3, 4],
    )


def test_derived_race_join_uses_source_ordinal_and_reconciles(contract):
    ids = ["dataset_b-000000004", "dataset_b-000000001", "dataset_b-000000003"]
    race = ri.join_derived_race(ids, [0, 1, 1], _raw_rows(), contract, reconciliation=pd.DataFrame({"lei": ["D", "A", "C"]}))
    assert race.tolist() == ["Black or African American", "White", "Asian"]
    with pytest.raises(ri.RQ4bInputError, match="action_taken"):
        ri.join_derived_race(ids, [1, 1, 1], _raw_rows(), contract)
    with pytest.raises(ri.RQ4bInputError, match="reconcile"):
        ri.join_derived_race(ids, [0, 1, 1], _raw_rows(), contract, reconciliation=pd.DataFrame({"lei": ["D", "A", "X"]}))
    with pytest.raises(ri.RQ4bInputError, match="source row"):
        ri.join_derived_race(["dataset_b-000000009"], [1], _raw_rows(), contract)
    bad = _raw_rows().assign(derived_race=["White", "Joint", "Other", "Asian", "Asian"])
    with pytest.raises(ri.RQ4bInputError, match="outside the contract"):
        ri.join_derived_race(ids, [0, 1, 1], bad, contract)


def test_derived_race_reader_reads_only_requested_rows(raw_config, rq3_protocol, tmp_path):
    local = ri.validate_contract(raw_config, rq3_protocol, project_root=tmp_path)
    source = local.resources["derived_race_source"]
    source.parent.mkdir(parents=True)
    _raw_rows().to_csv(source, index=False)
    access = ri.StageAccess("cv_epoch_selection", local)
    ids = ["dataset_b-000000005", "dataset_b-000000002"]
    permitted = frozenset([*ids, "dataset_b-000000099"])
    rows = ri.read_derived_race_rows(access, ids, permitted_ids=permitted, reconciliation_fields=("lei",))
    assert rows.index.tolist() == [1, 4]
    assert rows["lei"].tolist() == ["B", "E"]
    race = ri.join_derived_race(ids, [1, 0], rows, local)
    assert race.tolist() == ["Race Not Available", "Joint"]
    with pytest.raises(ri.RQ4bInputError, match="source row"):
        ri.read_derived_race_rows(access, ["dataset_b-000000099"], permitted_ids=permitted)


class _Manifest:
    train_record_ids = tuple(_ids(3, 1))
    validation_record_ids = tuple(_ids(1, 4))
    test_record_ids = tuple(_ids(1, 5))


@pytest.mark.parametrize("stage", ri.TRAIN_ONLY_STAGES)
def test_train_stage_cannot_read_validation_or_test_rows_from_raw_source(contract, stage):
    access = ri.StageAccess(stage, contract)
    permitted = ri.stage_population_ids(access, manifest_loader=lambda path: _Manifest)
    assert permitted == frozenset(_Manifest.train_record_ids)
    for foreign in (_Manifest.validation_record_ids, _Manifest.test_record_ids):
        with pytest.raises(ri.RQ4bAccessError, match="outside its population"):
            ri.read_derived_race_rows(access, [*_Manifest.train_record_ids[:1], *foreign], permitted_ids=permitted)


def test_probe_population_is_its_frozen_validation_ids_only(contract):
    access = ri.StageAccess("capability_probe", contract)
    with pytest.raises(ri.RQ4bAccessError):
        ri.stage_population_ids(access)  # no manifest fallback for the probe
    validated = ri.validate_frozen_probe_predictions(_probe_frame(), expected_rows=6)
    permitted = ri.stage_population_ids(access, probe_predictions=validated)
    assert permitted == frozenset(_ids(6))
    with pytest.raises(ri.RQ4bAccessError, match="outside its population"):
        ri.read_derived_race_rows(access, _ids(1, 100), permitted_ids=permitted)
    with pytest.raises(ri.RQ4bAccessError):
        ri.stage_population_ids(ri.StageAccess("synthetic_capability_gate", contract))


# ---------------------------------------------------------------------------
# Predictor schema and feature leakage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("leaked", ["derived_race", "applicant_sex", "record_id", "y_approved", "applicant_race-1", "lei"])
def test_metadata_in_native_frame_is_rejected_as_leakage(leaked):
    frame = _native_frame(8)
    frame[leaked] = "x"
    with pytest.raises(ri.RQ4bInputError, match="protected or audit"):
        ri.validate_native_predictor_frame(frame)


def test_raw_frame_requires_final_15_fields_without_leakage():
    raw = pd.DataFrame({column: ["1"] for column in ri.RAW_PREDICTOR_COLUMNS})
    ri.validate_raw_predictor_frame(raw)
    with pytest.raises(ri.RQ4bInputError, match="protected or audit"):
        ri.validate_raw_predictor_frame(raw.assign(applicant_sex="1"))
    with pytest.raises(ri.RQ4bInputError, match="15"):
        ri.validate_raw_predictor_frame(raw.iloc[:, ::-1])
    with pytest.raises(ri.RQ4bInputError, match="15"):
        ri.validate_raw_predictor_frame(raw.drop(columns=["income"]))


def test_native_schema_violations_are_rejected():
    frame = _native_frame(8)
    ri.validate_native_predictor_frame(frame)
    with pytest.raises(ri.RQ4bInputError, match="19"):
        ri.validate_native_predictor_frame(frame.iloc[:, ::-1])
    with pytest.raises(ri.RQ4bInputError, match="19"):
        ri.validate_native_predictor_frame(frame.drop(columns=[NUMERIC_VALUE_COLUMNS[0]]))
    as_float64 = frame.copy()
    as_float64[NUMERIC_VALUE_COLUMNS[0]] = as_float64[NUMERIC_VALUE_COLUMNS[0]].astype(np.float64)
    with pytest.raises(ri.RQ4bInputError, match="float32"):
        ri.validate_native_predictor_frame(as_float64)
    nonfinite = frame.copy()
    nonfinite.loc[0, NUMERIC_VALUE_COLUMNS[1]] = np.float32(np.nan)
    with pytest.raises(ri.RQ4bInputError, match="finite"):
        ri.validate_native_predictor_frame(nonfinite)
    plain = frame.copy()
    plain[CATEGORICAL_COLUMNS[0]] = plain[CATEGORICAL_COLUMNS[0]].astype(str)
    with pytest.raises(ri.RQ4bInputError, match="category dtype"):
        ri.validate_native_predictor_frame(plain)
    outside = frame.copy()
    outside[CATEGORICAL_COLUMNS[0]] = pd.Categorical.from_codes([-1] * 8, categories=list(CATEGORICAL_SCHEMA[CATEGORICAL_COLUMNS[0]]))
    with pytest.raises(ri.RQ4bInputError, match="outside the schema"):
        ri.validate_native_predictor_frame(outside)


def test_predictor_input_keeps_group_metadata_outside_features(contract):
    n = 6
    race = ["White", "Joint", "Asian", "Black or African American", "Race Not Available", "White"]
    built = ri.build_predictor_input(_native_frame(n), [1, 0, 1, 0, 1, 1], _ids(n), race, contract)
    assert tuple(built.X.columns) == ri.NATIVE_PREDICTOR_COLUMNS
    assert not set(built.X.columns) & ri.FORBIDDEN_PREDICTOR_COLUMNS
    assert built.group_codes.tolist() == [0, -1, 2, 1, -1, 0]
    assert built.roster_mask.sum() == 4  # outside-roster rows stay in the predictor population
    assert len(built.y) == n
    with pytest.raises(ri.RQ4bInputError, match="misaligned"):
        ri.build_predictor_input(_native_frame(n), [1] * n, _ids(n), race[:-1], contract)


# ---------------------------------------------------------------------------
# Frozen probabilities and reconstructed margins
# ---------------------------------------------------------------------------


def test_margin_is_log_ratio_of_stored_pair():
    approved = np.asarray([0.25, 0.9, 0.5])
    margin = ri.reconstruct_margin(approved, 1.0 - approved)
    np.testing.assert_array_equal(margin, np.log(approved) - np.log(1.0 - approved))
    assert margin.dtype == np.float64


def test_p_approved_one_with_positive_p_denied_is_accepted_without_one_minus_p():
    margin = ri.reconstruct_margin([1.0], [1e-7])
    assert np.isfinite(margin).all()
    assert margin[0] == np.log(1.0) - np.log(1e-7)
    with np.errstate(divide="ignore"):
        assert not np.isfinite(np.log(1.0) - np.log(1.0 - 1.0))  # the prohibited 1-p route


@pytest.mark.parametrize(
    ("approved", "denied", "message"),
    [
        ([1.0], [0.0], r"p_denied must lie"),
        ([0.0], [1.0], r"p_approved must lie"),
        ([np.nan], [0.5], "present and finite"),
        ([0.5], [np.inf], "present and finite"),
        ([1.2], [-0.2], "p_approved must lie"),
        ([0.6], [0.5], "sum to 1"),
        ([0.5, 0.5], [0.5], "aligned"),
        ([], [], "nonempty"),
        (["a"], [0.5], "numeric"),
    ],
)
def test_invalid_probability_pairs_are_rejected(approved, denied, message):
    with pytest.raises(ri.RQ4bInputError, match=message):
        ri.reconstruct_margin(approved, denied)


def _probe_frame(n: int = 6) -> pd.DataFrame:
    approved = np.linspace(0.2, 0.8, n)
    return pd.DataFrame(
        {"record_id": _ids(n), "y_approved": [1, 0] * (n // 2), "p_denied": 1.0 - approved, "p_approved": approved}
    )


def test_frozen_probe_frame_validation_and_g3_restriction(contract):
    validated = ri.validate_frozen_probe_predictions(_probe_frame(), expected_rows=6)
    race = ["White", "Joint", "Asian", "Black or African American", "Race Not Available", "Asian"]
    population = ri.build_real_probe_population(validated, race, contract)
    assert population["record_id"].tolist() == [_ids(6)[i] for i in (0, 2, 3, 5)]
    assert population["group_code"].tolist() == [0, 2, 1, 2]
    np.testing.assert_array_equal(population["margin"], validated["margin"].to_numpy()[[0, 2, 3, 5]])


def test_frozen_probe_frame_failures_stop_without_row_drop():
    with pytest.raises(ri.RQ4bInputError, match="columns"):
        ri.validate_frozen_probe_predictions(_probe_frame().drop(columns=["p_denied"]), expected_rows=6)
    with pytest.raises(ri.RQ4bInputError, match="rows"):
        ri.validate_frozen_probe_predictions(_probe_frame(), expected_rows=7)
    missing = _probe_frame()
    missing.loc[2, "p_denied"] = np.nan
    with pytest.raises(ri.RQ4bInputError, match="present and finite"):
        ri.validate_frozen_probe_predictions(missing, expected_rows=6)
    zero = _probe_frame()
    zero.loc[4, ["p_approved", "p_denied"]] = [1.0, 0.0]
    with pytest.raises(ri.RQ4bInputError, match="p_denied must lie"):
        ri.validate_frozen_probe_predictions(zero, expected_rows=6)
    text = _probe_frame().astype({"p_approved": object})
    text.loc[1, "p_approved"] = "n/a"
    with pytest.raises(ri.RQ4bInputError, match="present and finite"):
        ri.validate_frozen_probe_predictions(text, expected_rows=6)


# ---------------------------------------------------------------------------
# Stage-aware access
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stage", ri.TRAIN_ONLY_STAGES)
@pytest.mark.parametrize("partition", ["validation", "test"])
def test_train_only_stages_cannot_open_validation_or_test(contract, stage, partition, monkeypatch):
    access = ri.StageAccess(stage, contract)
    assert access.require_partition("train") == contract.resources["split_train"]
    with pytest.raises(ri.RQ4bAccessError):
        access.require_partition(partition)
    with pytest.raises(ri.RQ4bAccessError):
        access.require_resource("frozen_probe_validation_predictions")
    with pytest.raises(ri.RQ4bAccessError):
        access.require_path(contract.resources[f"split_{partition}"])

    def forbidden_loader(*args, **kwargs):
        raise AssertionError("loader must not be constructed for a forbidden partition")

    monkeypatch.setattr(split_inputs.DatasetBSplitLoader, "__init__", forbidden_loader)
    with pytest.raises(ri.RQ4bAccessError):
        ri.load_partition(access, partition)


def test_probe_stage_opens_only_its_declared_frozen_source(contract):
    access = ri.StageAccess("capability_probe", contract)
    assert access.require_path(PROJECT_ROOT / ri.PROBE_PREDICTION_PATH) == "frozen_probe_validation_predictions"
    access.require_resource("derived_race_source")
    for partition in ("train", "validation", "test"):
        with pytest.raises(ri.RQ4bAccessError):
            access.require_partition(partition)
    for resource in ("split_manifest", "locked_test_bootstrap_indices"):
        with pytest.raises(ri.RQ4bAccessError):
            access.require_resource(resource)
    other_artifact = PROJECT_ROOT / "artifacts/dataset_b/realmlp/final" / ri.PROBE_FROZEN_RUN_ID / "test/test_predictions.csv"
    with pytest.raises(ri.RQ4bAccessError, match="not a declared"):
        access.require_path(other_artifact)


def test_forbidden_probe_load_raises_before_opening(raw_config, rq3_protocol, tmp_path):
    local = ri.validate_contract(raw_config, rq3_protocol, project_root=tmp_path)  # no files exist here
    with pytest.raises(ri.RQ4bAccessError):
        ri.load_frozen_probe_predictions(ri.StageAccess("full_train_refit", local))
    with pytest.raises(FileNotFoundError):
        ri.load_frozen_probe_predictions(ri.StageAccess("capability_probe", local))


def test_validation_and_synthetic_stage_boundaries(contract):
    validation = ri.StageAccess("validation_calibration_selection", contract)
    validation.require_partition("validation")
    for partition in ("train", "test"):
        with pytest.raises(ri.RQ4bAccessError):
            validation.require_partition(partition)
    with pytest.raises(ri.RQ4bAccessError):
        ri.make_fit_population_preprocessor(validation)
    synthetic = ri.StageAccess("synthetic_capability_gate", contract)
    for resource in contract.resources:
        with pytest.raises(ri.RQ4bAccessError):
            synthetic.require_resource(resource)
    with pytest.raises(ri.RQ4bAccessError, match="Unknown RQ4b stage"):
        ri.StageAccess("pilot", contract)


def test_fit_population_preprocessor_is_fresh_for_train_stage(contract):
    access = ri.StageAccess("full_train_refit", contract)
    first = ri.make_fit_population_preprocessor(access)
    second = ri.make_fit_population_preprocessor(access)
    assert first is not second
    assert [name for name, _ in first.steps] == ["two_channel", "log1p_numeric", "native"]


def test_outputs_must_stay_inside_rq4b_namespace(contract):
    inside = contract.artifact_namespace / "run_x" / "manifest.json"
    assert ri.require_rq4b_output_path(inside, contract) == inside.resolve()
    for outside in (
        PROJECT_ROOT / "artifacts/dataset_b/realmlp/final" / ri.PROBE_FROZEN_RUN_ID / "x.json",
        PROJECT_ROOT / "artifacts/dataset_b/rq4/rq4a/x.json",
        contract.artifact_namespace,
        contract.artifact_namespace / ".." / "rq4a" / "x.json",
    ):
        with pytest.raises(ri.RQ4bInputError, match="outside"):
            ri.require_rq4b_output_path(outside, contract)


# ---------------------------------------------------------------------------
# C1 finding 2: partition loading bound to the authorized resources
# ---------------------------------------------------------------------------


AUDIT_COLUMNS = ("lei", "county_code", *(f"applicant_race-{slot}" for slot in range(1, 6)), "applicant_sex", "applicant_age")
RAW_ROW = {
    "loan_type": "1", "loan_purpose": "1", "lien_status": "1", "reverse_mortgage": "2",
    "open-end_line_of_credit": "2", "business_or_commercial_purpose": "2", "income": "100",
    "debt_to_income_ratio": "<20%", "loan_amount": "200000", "loan_to_value_ratio": "80",
    "loan_term": "360", "occupancy_type": "1", "preapproval": "2", "interest_only_payment": "2",
    "balloon_payment": "2",
}
SYNTHETIC_SPLIT = {"train": _ids(4, 1), "validation": _ids(2, 5), "test": _ids(2, 7)}
SYNTHETIC_ACTIONS = {"train": ["1", "3", "2", "3"], "validation": ["1", "3"], "test": ["3", "1"]}


def _partition_rows(partition: str) -> list[dict]:
    rows = []
    for record_id, action in zip(SYNTHETIC_SPLIT[partition], SYNTHETIC_ACTIONS[partition]):
        rows.append(
            {"record_id": record_id, "action_taken": action, "y_approved": "0" if action == "3" else "1",
             **RAW_ROW, **{column: "1" for column in AUDIT_COLUMNS}}
        )
    return rows


def _write_synthetic_root(root: Path, *, overrides: dict | None = None) -> None:
    """Synthetic split_v1 layout under ``root``; no project data is copied."""
    overrides = overrides or {}
    (root / "config" / "dataset_b").mkdir(parents=True)
    shutil.copyfile(PROJECT_ROOT / ri.FEATURE_CONTRACT_PATH, root / ri.FEATURE_CONTRACT_PATH)
    manifest = {
        "dataset": "dataset_b", "split_unit": "record_id", "grouping_constraint": "none",
        "random_state": 42, "split_version": "split_v1",
        **{f"{name}_record_ids": list(ids) for name, ids in SYNTHETIC_SPLIT.items()},
    }
    (root / ri.SPLIT_MANIFEST_PATH).parent.mkdir(parents=True)
    (root / ri.SPLIT_MANIFEST_PATH).write_text(json.dumps(manifest), encoding="utf-8")
    (root / "data" / "dataset_b" / "split_v1").mkdir(parents=True)
    for partition in SYNTHETIC_SPLIT:
        frame = pd.DataFrame(overrides.get(partition) or _partition_rows(partition))
        frame.to_csv(root / ri.SPLIT_V1_PATHS[partition], index=False)


def _synthetic_contract(raw_config, rq3_protocol, root: Path) -> ri.RQ4bContract:
    local = ri.validate_contract(raw_config, rq3_protocol, project_root=root)
    return dataclasses.replace(local, split_rows={name: len(ids) for name, ids in SYNTHETIC_SPLIT.items()})


@pytest.fixture
def loader_spy(monkeypatch):
    """Record the paths the shared split loader is actually constructed with."""
    calls: list[dict] = []
    original = split_inputs.DatasetBSplitLoader.__init__

    def spy(self, *args, **kwargs):
        calls.append(kwargs)
        original(self, *args, **kwargs)

    monkeypatch.setattr(split_inputs.DatasetBSplitLoader, "__init__", spy)
    return calls


@pytest.mark.parametrize(
    ("stage", "partition"),
    [("cv_epoch_selection", "train"), ("full_train_refit", "train"), ("validation_calibration_selection", "validation")],
)
def test_authorized_resources_are_the_resources_loaded(raw_config, rq3_protocol, tmp_path, loader_spy, stage, partition):
    _write_synthetic_root(tmp_path)
    local = _synthetic_contract(raw_config, rq3_protocol, tmp_path)
    gated = ri.load_partition(ri.StageAccess(stage, local), partition)
    assert loader_spy == [
        {
            "manifest_path": local.resources["split_manifest"],
            "split_dir": local.resources[f"split_{partition}"].parent,
            "feature_contract_path": local.resources["feature_contract"],
        }
    ]
    for path in loader_spy[0].values():
        assert tmp_path in Path(path).parents
    assert gated.record_ids.tolist() == SYNTHETIC_SPLIT[partition]
    assert gated.y.tolist() == [0 if action == "3" else 1 for action in SYNTHETIC_ACTIONS[partition]]
    assert tuple(gated.X.columns) == ri.RAW_PREDICTOR_COLUMNS
    assert gated.quality_audit.status == "PASS"


def test_alternate_root_never_falls_back_to_repository_defaults(raw_config, rq3_protocol, tmp_path, loader_spy):
    local = _synthetic_contract(raw_config, rq3_protocol, tmp_path)  # empty root: no manifest or CSV
    with pytest.raises(split_inputs.SplitInputError, match="manifest"):
        ri.load_partition(ri.StageAccess("cv_epoch_selection", local), "train")
    assert loader_spy[0]["manifest_path"] == tmp_path / ri.SPLIT_MANIFEST_PATH
    assert PROJECT_ROOT / ri.SPLIT_MANIFEST_PATH != loader_spy[0]["manifest_path"]


@pytest.mark.parametrize(
    ("stage", "partition"),
    [("cv_epoch_selection", "validation"), ("full_train_refit", "test"), ("capability_probe", "validation"),
     ("validation_calibration_selection", "test"), ("synthetic_capability_gate", "train")],
)
def test_forbidden_load_fails_before_loader_construction_or_file_open(raw_config, rq3_protocol, tmp_path, monkeypatch, stage, partition):
    _write_synthetic_root(tmp_path)
    local = _synthetic_contract(raw_config, rq3_protocol, tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("must not construct a loader or open a file")

    monkeypatch.setattr(split_inputs.DatasetBSplitLoader, "__init__", forbidden)
    monkeypatch.setattr(split_inputs.pd, "read_csv", forbidden)
    with pytest.raises(ri.RQ4bAccessError):
        ri.load_partition(ri.StageAccess(stage, local), partition)


def _mutated_train(mutate) -> list[dict]:
    rows = _partition_rows("train")
    mutate(rows)
    return rows


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"train": _mutated_train(lambda rows: rows.__setitem__(slice(None), rows[::-1]))}, split_inputs.SplitInputError, "order"),
        ({"train": _mutated_train(lambda rows: rows[1].update(y_approved="1"))}, split_inputs.SplitInputError, "target"),
        ({"train": [{k: v for k, v in row.items() if k != "income"} for row in _partition_rows("train")]}, split_inputs.SplitInputError, "schema"),
        ({"train": _mutated_train(lambda rows: rows[2].update(loan_type=" "))}, split_inputs.InputQualityError, "rejected"),
        ({"train": _mutated_train(lambda rows: rows[2].update(loan_type="9"))}, split_inputs.InputQualityError, "loan_type"),
        ({"train": _partition_rows("train")[:3]}, split_inputs.SplitInputError, "order"),
    ],
)
def test_existing_row_schema_label_and_membership_checks_remain_active(raw_config, rq3_protocol, tmp_path, overrides, error, message):
    _write_synthetic_root(tmp_path, overrides=overrides)
    local = _synthetic_contract(raw_config, rq3_protocol, tmp_path)
    with pytest.raises(error, match=message):
        ri.load_partition(ri.StageAccess("cv_epoch_selection", local), "train")


def test_row_count_must_match_split_v1(raw_config, rq3_protocol, tmp_path):
    _write_synthetic_root(tmp_path)
    local = dataclasses.replace(_synthetic_contract(raw_config, rq3_protocol, tmp_path), split_rows=dict(ri.SPLIT_V1_ROWS))
    with pytest.raises(ri.RQ4bInputError, match="205310"):
        ri.load_partition(ri.StageAccess("cv_epoch_selection", local), "train")
