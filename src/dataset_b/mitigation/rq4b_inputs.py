"""RQ4b production contract reader and input-validation boundary.

This module is independent of the RQ4a contract object and of the disposable
RQ4b spike. Pure checks never read files; the few loaders verify stage access
before any file is opened. Nothing is loaded at import time.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from src.dataset_b.fairness.audit_data import RECORD_ID_PATTERN, source_ordinals
from src.dataset_b.preprocessing.categorical_encoder import (
    CATEGORICAL_COLUMNS,
    CATEGORICAL_SCHEMA,
    NUMERIC_VALUE_COLUMNS,
    TWO_CHANNEL_COLUMNS,
)
from src.dataset_b.preprocessing.raw_parser import FEATURE_COLUMNS
from src.utils.path import PROJECT_ROOT


RQ4B_CONFIG_PATH = PROJECT_ROOT / "config" / "dataset_b" / "rq4b_adversarial.yaml"
PROTOCOL_VERSION = "rq4b_adversarial_realmlp_v4_seed44_continuation"
DECISION_REVISION = "rq4b3_test_output_contract_20261024"
METHOD_CONTRACT_STATE = "user_selected_test_arms_metrics_lineage_and_schema_locked"
TRACK_A_SOURCE_RUN_ID = "rq4b3_adv_20261019_02"
TRACK_B_CHILD_RUN_ID = "rq4b3_adv_20261021_01"
SEED44_CONTINUATION_RUN_ID = "rq4b3_adv_20261022_01"
SEED44_CONTINUATION_ROLE = "track_b_seed44_continuation"
SEED44_PRODUCTION_UNITS = ((0.0, 44), (0.01, 44))
TEST_RUN_ID = "rq4b3_test_20261024_01"
TEST_PROTOCOL_VERSION = "rq4b_adversarial_cross_run_test_evaluation_v1"
GATE_C_SELECTION_SHA256 = "76af4d077ea9febbb8865008c6d566923ddb66e88f5026330494ffa8bdc8c615"
GATE_C_MANIFEST_SHA256 = "5d072d7ff094817b8a281d2ea5b6527682678f97d09fa538b88013aae218931b"
GATE_C_RECEIPT_SHA256 = "14f602bb0ef59d7cbfb22e8b5fbdaf6604e5a90c295370bac212b965e1448f13"
TEST_EVALUATION_UNITS = (
    (0.0, 42), (0.0, 43), (0.0, 44),
    (0.01, 42), (0.01, 43), (0.01, 44),
    (0.03, 42), (0.1, 42), (0.3, 42),
)
TEST_MATCHED_COMPARISONS = (
    (42, 0.01, 0.0), (42, 0.03, 0.0), (42, 0.1, 0.0),
    (42, 0.3, 0.0), (43, 0.01, 0.0), (44, 0.01, 0.0),
)

G3_GROUPS = ("White", "Black or African American", "Asian")
G6_GROUPS = (
    *G3_GROUPS,
    "American Indian or Alaska Native",
    "2 or more minority races",
    "Joint",
)
COVERAGE_ONLY_GROUPS = (
    "Native Hawaiian or Other Pacific Islander",
    "Free Form Text Only",
    "Race Not Available",
)
OUTSIDE_ROSTER_CODE = -1

RAW_PREDICTOR_COLUMNS = tuple(FEATURE_COLUMNS)
NATIVE_PREDICTOR_COLUMNS = tuple(TWO_CHANNEL_COLUMNS)
RAW_FEATURE_COUNT = 15
SEMANTIC_FEATURE_COUNT = 19
# Identifiers, target, protected attributes and audit-only fields; none may enter X.
FORBIDDEN_PREDICTOR_COLUMNS = frozenset(
    {
        "record_id",
        "action_taken",
        "y_approved",
        "derived_race",
        "applicant_sex",
        "applicant_age",
        "lei",
        "county_code",
        *(f"applicant_race-{slot}" for slot in range(1, 6)),
    }
)

SPLIT_V1_ROWS = {"train": 205_310, "validation": 43_995, "test": 43_996}
SPLIT_V1_PATHS = {
    partition: f"data/dataset_b/split_v1/{partition}.csv" for partition in SPLIT_V1_ROWS
}
SPLIT_MANIFEST_PATH = "artifacts/dataset_b/split_v1.json"
FEATURE_CONTRACT_PATH = "config/dataset_b/feature_set_v1.yaml"
DERIVED_RACE_SOURCE_PATH = "data/raw/state_NY_2024.csv"
DERIVED_RACE_JOIN_METHOD = "one_based_source_row_number_from_record_id_suffix"

PROBE_FROZEN_RUN_ID = "realmlp_final_15f_20261004_01"
PROBE_PREDICTION_PATH = (
    f"artifacts/dataset_b/realmlp/final/{PROBE_FROZEN_RUN_ID}/validation/validation_predictions.csv"
)
PROBE_PREDICTION_COLUMNS = ("record_id", "y_approved", "p_denied", "p_approved")
PROBABILITY_SUM_ATOL = 1e-6

LAMBDA_GRID = (0, 0.01, 0.03, 0.1, 0.3)
DEVELOPMENT_SEEDS = (42, 43, 44)
FINAL_SEEDS = (42, 43, 44)
RETAINED_DEVELOPMENT_UNITS = ((0.0, 42), (0.01, 42), (0.03, 42), (0.1, 42), (0.3, 42))
ADDITIONAL_DEVELOPMENT_UNITS = ((0.0, 43), (0.01, 43), (0.0, 44), (0.01, 44))
DEVELOPMENT_UNITS = (*RETAINED_DEVELOPMENT_UNITS, *ADDITIONAL_DEVELOPMENT_UNITS)
CV_FOLDS = 5
EPOCHS_PER_FOLD = 256

# Approved execution-critical contract. The v3 amendment below overlays the
# retained v2 method body so the original and amended protocols remain
# distinguishable while every active field is still compared in full. Only the
# paths in UNLOCKED_PATHS may differ: run/readiness status, the historical spike
# record and governance roles.
_V2_BASE_CONTRACT_YAML = """schema_version: 1
dataset: dataset_b
protocol_version: rq4b_adversarial_realmlp_v2_cv_epoch_selection
repository_question: RQ4b
manuscript_equivalent: RQ3b
decision_revision: final_probe_and_numerical_lock_20261018
method_contract_state: user_decisions_locked_ready_for_scoped_implementation
artifact_namespace: artifacts/dataset_b/rq4/rq4b
prohibit_rq3_artifact_namespace_reuse: true
prerequisites: {documentation_and_config_approval: required, rq1_paired_bootstrap_task: required_before_execution,
  internal_extension_technical_spike: required}
target: {label: y_approved, positive_outcome: Approved, join_key: record_id}
groups:
  attribute: derived_race
  interpretation: hmda_application_level_applicant_coapplicant_aggregation
  joint_is_applicant_race: false
  g3: [White, Black or African American, Asian]
  g6: [White, Black or African American, Asian, American Indian or Alaska Native, 2 or more minority races, Joint]
  coverage_only_outside_g6: [Native Hawaiian or Other Pacific Islander, Free Form Text Only, Race Not Available]
  fixed_roster_across_bootstrap: true
  allow_drop_merge_rename_or_dynamic_shrink: false
  invalid_required_group_metric_invalidates_eod6_replicate: true
denominators: {source_contract: config/dataset_b/rq3_protocol.yaml, minimum_group_rows: 100, minimum_actual_approved: 30,
  minimum_actual_denied: 30, insufficient_result: null, never_replace_undefined_with_zero: true}
development_governance:
  source_partition: validation
  use_complete_validation: true
  split_validation: false
  split_validation_scope: predictor_threshold_calibration_and_lambda_selection
  capability_probe_internal_split_is_separate: true
  threshold_calibration_partition: validation
  lambda_selection_partition: validation
  lambda_selection_primary_fairness_metric: eod3
  eod6_role: audit_only
  preserve_train_validation_and_test_membership: true
  same_partition_selection_calibration_limitation: explicitly_report
  required_checks: [complete_validation_coverage, stable_record_id, group_outcome_counts]
realmlp_recipe:
  base_config: config/dataset_b/realmlp_v1.yaml
  predictor_features_exclude: [derived_race, applicant_sex]
  predictor_loss_population: all_train_rows
  predictor_population_semantics: no_race_filter_not_every_row_every_epoch
  preserve_native_loader_drop_last: true
  adversarial_loss_population: configured_group_roster_only
  all_groups_receive_predictor_output: true
  decision_threshold: one_global_calibrated_threshold
  adversary:
    task: multiclass_group_prediction
    loss_primary: unweighted_multiclass_cross_entropy
    conditioned_on_true_y_during_training_only: true
    input_preference: approved_minus_denied_logit_margin
    fallback_input: historical_pre_spike_probability_not_production_selected
    final_input_requires_technical_spike_gate: true
    predictor_total_loss: L_prediction - lambda * L_adversary
    gradient_reversal_equivalent_allowed: true
    additive_positive_adversary_loss_prohibited: true
    optimizer: {name: adam, learning_rate: 0.001, weight_decay: 0.0}
    updates_per_batch: {adversary: 1, predictor: 1}
    update_order: detached_score_adversary_then_frozen_adversary_predictor
    empty_roster_batch: predictor_only_and_log_adversary_skip
    group_alignment: shuffled_row_metadata_removed_before_predictor_forward
    lambda_zero: bypass_adversary_and_isolate_adversary_rng
    production_architecture:
      status: user_locked
      input_features: [approved_minus_denied_logit_margin, true_y_training_only]
      input_dimensions: 2
      hidden_sizes: [32]
      activation: relu
      output_classes: 3
      dropout: 0.0
      class_weighting: false
      architecture_search: false
    seed_policy: {adversary_seed: same_as_model_seed, matched_initialization_across_lambda_within_seed_and_fold: true,
      isolate_rng_from_predictor: true}
  lambda_grid: [0, 0.01, 0.03, 0.1, 0.3, 1.0]
  development_seeds: [42, 43, 44]
  final_seeds: [42, 43, 44, 45, 46]
epoch_selection:
  policy: post_training_five_fold_mean_prediction_ce
  splitter: {name: StratifiedKFold, folds: 5, shuffle: true, random_state: 42}
  same_fold_membership_across_lambdas_and_model_seeds: true
  population: train_only
  unit: each_lambda_and_seed
  epochs_per_fold: 256
  use_early_stopping: false
  monitor: unweighted_binary_prediction_cross_entropy_approved_positive
  evaluation_mode: eval_no_grad
  numerical_computation: stable_binary_ce_from_approved_minus_denied_logits
  probability_clipping: prohibited_as_unapproved_alternative
  fold_aggregation: unweighted_arithmetic_mean
  required_finite_fold_epoch_values: 1280
  selection: minimum_unrounded_mean_ce
  exact_tie_break: earliest_epoch
  epoch_cap_hit: selected_epoch_equals_256
  epoch_cap_hit_action: report_only_no_automatic_budget_extension
  fairness_metrics_role: monitoring_only
  prohibit_adversarial_loss_or_project_validation_or_test_selection: true
  cv_selection_optimism_limitation_required: true
  full_train_refit:
    epochs: selected_epoch
    fresh_states: [preprocessing, predictor, adversary, optimizers, backend]
    stopping_holdout: false
    schedule_horizon:
      status: user_locked
      epochs: 256
      applies_to: [learning_rate, weight_decay, dropout]
      execution_budget: selected_epoch
      compress_schedule_to_selected_epoch: false
capability_assessment:
  required_components: [synthetic_correctness_gate, independent_probe_vs_y_only]
  architecture_search_on_inconclusive: false
  inconclusive_status: capability_inconclusive
  inconclusive_is_execution_failure: false
  inconclusive_proves_no_group_information: false
  real_probe:
    source: frozen_rq1_realmlp_validation_predictions
    frozen_run_id: realmlp_final_15f_20261004_01
    groups: g3
    architecture: fixed_2_32_relu_3_no_architecture_search
    split:
      fractions: {train: 0.6, validation: 0.2, holdout: 0.2}
      stratify_by: [derived_race, y_approved]
      random_state: 42
      changes_project_split_membership: false
    fitting:
      optimizer: adamw
      learning_rate: 0.001
      maximum_epochs: 100
      patience: 15
      weight_decay: 0.01
      betas: [0.9, 0.999]
      epsilon: 1.0e-08
      amsgrad: false
      maximize: false
      capturable: false
      differentiable: false
      foreach: false
      fused: false
      batch_size: 256
      device: cuda:0
      random_state: 42
      isolate_rng_from_predictor: true
      monitor: unweighted_group_prediction_ce_on_probe_validation
      monitor_mode: min
      evaluation_mode: eval_no_grad
      min_delta: 0.0
      improvement: strictly_lower_ce
      exact_tie_break: earliest_checkpoint
      stopping: 15_consecutive_completed_epochs_without_improvement
      restore_best_checkpoint_before_holdout: true
      nonfinite_loss_or_gradient: stop_attempt_and_write_failure_receipt
      skip_bad_batch_or_automatic_restart: false
    y_only_comparator: {definition: conditional_group_probability_given_true_y, estimator: laplace_smoothed_group_counts_within_y,
      alpha: 1.0, fitted_on: probe_train_only}
    probe_fitted_on: probe_train_only
    checkpoint_selected_on: probe_validation_only
    gain_and_intervals_reported_on: probe_holdout_only
    ce_gain: ce_y_only_minus_ce_probe
    production_pass_fail_role: none_descriptive_only
    prohibit_holdout_driven_architecture_or_protocol_changes: true
    source_score_extraction:
      prediction_path: artifacts/dataset_b/realmlp/final/realmlp_final_15f_20261004_01/validation/validation_predictions.csv
      columns: [p_approved, p_denied]
      dtype: float64
      formula: log_p_approved_minus_log_p_denied
      interpretation: reconstructed_margin_from_stored_probabilities_not_bitwise_original_logits
      require_finite_probabilities: true
      probability_bounds: strictly_positive_and_at_most_one
      probability_sum_check: {target: 1.0, rtol: 0.0, atol: 1.0e-06}
      require_finite_margin: true
      invalid_input_action: stop_attempt_no_clipping_no_row_drop_no_new_inference
  synthetic_correctness_gate:
    fixture_source_and_generation:
      type: fixed_synthetic_positive_control
      rows: 6000
      groups: g3
      rows_per_group: 2000
      rows_per_group_and_y: 1000
      margin: group_center_plus_zero_mean_gaussian_noise
      centers_in_g3_order: [-2.0, 0.0, 2.0]
      noise_standard_deviation: 0.25
      numpy_rng: default_rng
      numpy_seed: 42
      stable_synthetic_record_ids: true
      split: {train: 0.6, validation: 0.2, holdout: 0.2}
      stratify_by: [group, y]
      split_seed: 42
      probe_inputs: [synthetic_margin, true_y]
      group_is_target_not_input: true
      fit_and_comparator_policy: same_locked_probe_recipe
      prohibit_fixture_retuning_after_results: true
    ce_gain_minimum_nats: 0.05
    ce_gain_comparison: greater_than_or_equal
    lower_95_ci_gain: strictly_greater_than_zero
    distinct_from_real_probe_gain_requirement: true
  probe_uncertainty:
    method: stratified_paired_bootstrap
    strata: [derived_race, y_approved]
    population: held_out_probe_rows
    replicates: 1000
    interval: percentile
    confidence_level: 0.95
    quantiles: [0.025, 0.975]
    quantile_interpolation: linear
    random_state: 42
    round_before_quantiles_or_endpoint_comparison: false
    minimum_valid_replicates: 1000
    paired_quantities: [y_only_ce, probe_ce]
    refit_probe_or_comparator_in_bootstrap: false
    reuse_production_test_indices: false
    invalid_quantity_action: ci_not_evaluable_no_replacement_replicates
  use_test_to_design_or_select_probe: false
threshold_calibration:
  partition: validation
  candidates: {start: 0.001, stop: 0.999, step: 0.001, count: 999}
  objective: maximize_balanced_accuracy
  tie_break: [minimum_absolute_distance_to_0.5, lower_threshold]
  global_threshold_only: true
candidate_selection: {evaluation_partition: validation, matched_control: same_seed_lambda_zero, eligibility_metric: mean_matched_delta_balanced_accuracy,
  eligibility_minimum: -0.01, rq4b_3_objective: minimum_mean_eod3, eod6_may_select_lambda: false, exact_tie_break: smaller_lambda,
  lambda_zero_may_win: true, use_test_for_selection: false, same_partition_as_threshold_calibration: true, exploratory_optimism_limitation_required: true}
configurations:
  RQ4b-3: {role: primary, group_roster: g3}
  RQ4b-6:
    role: deferred_optional_sensitivity
    group_roster: g6
    requires_separate_future_authorization: true
    prerequisite: rq4b_3_gates_and_separate_future_diagnostic_acceptance
    small_cell_warning: 2 or more minority races
    optional_weighted_sensitivity: {enabled_by_default: false, weighting: train_only_inverse_frequency, weights_fixed_before_test: true,
      additional_unique_full_fits: 5}
fit_budget:
  train_lambda_zero_controls_for_all_final_seeds: true
  reuse_lambda_zero_controls_across_g3_g6: true
  reuse_selected_lambda_artifacts_for_seeds: [42, 43, 44]
  retrain_reused_selected_lambda_artifacts: false
  committed_rq4b_3_maximum_unique_full_fits_approx: 22
  rq4b_3_maximum_cv_fold_fits_excluding_spike_and_retries: 110
  rq4b_3_maximum_total_fits_excluding_spike_and_retries: 132
  capability_probe_fits_included_in_132: false
  budget_basis: 18_development_combinations_plus_up_to_4_final_seed_combinations
  future_rq4b_6_additional_primary_fits_approx: 17
  future_optional_weighted_rq4b_6_additional_fits: 5
  future_rq4b_6_counts_are_refit_only_cv_budget_pending: true
  rq4b_6_outside_committed_budget: true
runtime:
  device: cuda:0
  allow_silent_cpu_fallback: false
  batch_size: 256
  predict_batch_size: 1024
  preflight_required: [cuda_available, finite_operation]
  compatibility_baseline: {python: 3.13.14, torch: 2.14.0+cu130, pytabkit: 1.7.3, numpy: 2.5.2, pandas: 3.0.5, scikit_learn: 1.9.1}
  pytabkit_source_revision_pin_required: true
internal_extension_and_naming_gate:
  public_sklearn_wrapper_has_adversarial_hooks: false
  existing_history_adapter_proves_internal_extension_possible: true
  existing_history_adapter_proves_adversarial_recipe: false
  required_gates: [pin_pytabkit_source_revision, replace_or_subclass_training_step, locate_score_tensor_and_verify_dimensions,
    finite_one_batch_forward_backward, serialization_and_reload, lambda_zero_equivalence, roster_alignment_and_signed_gradient_flow,
    cuda_fail_closed_and_nonfinite_failure_paths, exact_seed_plumbing_and_short_smoke_repeatability, synthetic_epoch_selection_and_schedule_horizon_feasibility]
  preserve_realmlp_recipe_required: true
  if_recipe_not_preserved: do_not_name_adversarial_realmlp
metrics:
  primary_fairness: [eod3]
  audit_fairness: [eod6]
  eod_definition: max_fixed_roster_tpr_range_fpr_range
  primary_utility: full_partition_balanced_accuracy
  ba_budget_primary: 0.01
  ba_budget_sensitivity: [0.005, 0.02]
  fairness_generalization_gap:
    name: fairness_generalization_gap
    definition: test_eod_minus_development_eod
    report_for: [eod3, eod6]
    development_partition: validation
    same_partition_selection_calibration_limitation: true
    prohibited_name: calibration-versus-Test fairness gap
  additional: [denied_f1, denied_precision, denied_recall, per_group_tpr, per_group_fpr, per_group_selection_rate,
    extrema_groups, group_coverage_and_denominators, sex_eod_safety_audit, decision_changes_overall_and_by_group]
uncertainty:
  within_seed:
    method: paired_test_bootstrap_vs_matched_lambda_zero
    replicates: 1000
    confidence_level: 0.95
    interval: percentile
    quantiles: [0.025, 0.975]
    quantile_interpolation: linear
    round_before_quantiles_or_endpoint_comparison: false
    stratification: y_approved_as_in_locked_rq1_indices
    do_not_regenerate_race_y_test_indices: true
    minimum_valid_replicates_per_quantity: 1000
    invalid_quantity_action: ci_not_evaluable_no_replacement_replicates
    ci_invalidity_is_scientific_failure: false
    refit_or_recalibrate_in_bootstrap: false
    indices: artifacts/dataset_b/rq1_test_bootstrap_indices.npz
    reuse_locked_indices: true
    allow_regenerate_indices: false
  across_seed:
    summaries: [mean, standard_deviation, min, max, range]
    standard_deviation_ddof: 1
    required_seed_count: 5
    pool_bootstrap_replicates_across_seeds: false
  select_seed_from_test: false
  save_invalid_replicates_and_reasons: true
scientific_assessment:
  delta_direction: adversarial_minus_same_seed_lambda_zero
  per_seed_point_criterion: {fairness: delta_eod3_strictly_less_than_zero, utility: full_test_delta_balanced_accuracy_greater_than_or_equal_to_minus_0_01,
    combination: both_required, missing_required_quantity: not_evaluable}
  strong_support: {required_seed_count: 4, final_seed_count: 5, same_seed_must_satisfy_both_ci_conditions: true,
    ci_conditions_status: user_locked, fairness_condition: upper_95_ci_delta_eod3_strictly_less_than_zero, utility_condition: lower_95_ci_delta_ba_strictly_greater_than_minus_0_01,
    invalid_ci_counts_as_support: false}
  stability_status: not_assessed
  strong_support_implies_stable: false
  scientific_conclusion_separate_from_execution_status: true
reporting_contract:
  required_evidence: [cv_ce_curves, selected_epoch, epoch_cap_hit, configurations, seeds, provenance, diagnostics,
    per_seed_matched_control_deltas, per_seed_bootstrap_intervals, invalid_replicates_and_reasons, across_seed_summaries,
    capability_assessment]
  prohibit_best_seed_selection: true
  prohibit_unfavorable_seed_exclusion: true
  prohibit_test_driven_budget_extension: true
status_contract:
  execution_status: [completed, failed, infeasible]
  stability_status: [not_assessed]
  primary_criterion: [pass, fail, not_evaluable]
  primary_criterion_scope: per_seed_point_criterion
  strong_support_is_separate_field: true
  fields_are_independent: true
  reason_codes: [none, prerequisite_rq1_paired_bootstrap_missing, technical_spike_pending, pytabkit_revision_unpinned,
    adversarial_hook_unavailable, score_tensor_unresolved, nonfinite_preflight, serialization_reload_failed, lambda_zero_equivalence_failed,
    required_group_metric_invalid, insufficient_valid_bootstrap_replicates, ba_budget_exceeded, unstable_diagnostics,
    runtime_failure, capability_inconclusive]
prohibitions: {prohibit_race_or_sex_as_predictor: true, prohibit_group_specific_thresholds: true, prohibit_silent_cpu_fallback: true,
  prohibit_test_selection_of_lambda_threshold_seed_or_weighting: true, prohibit_best_seed_selection: true, prohibit_dynamic_group_roster: true,
  prohibit_locked_test_bootstrap_regeneration: true, prohibit_adversarial_realmlp_name_before_gate_passes: true,
  prohibit_legal_compliance_claim: true, prohibit_discrimination_claim: true, prohibit_causal_claim: true, prohibit_creditworthiness_or_default_risk_claim: true}
"""
APPROVED_CONTRACT: dict[str, Any] = yaml.safe_load(_V2_BASE_CONTRACT_YAML)
_V3_AMENDMENT: dict[str, Any] = yaml.safe_load(
    """
protocol_version: rq4b_adversarial_realmlp_v3_parallel_evaluation_training
decision_revision: rq4b3_parallel_evaluation_training_20261021
method_contract_state: user_approved_amendment_recorded_selection_policy_pending
contract_amendment:
  id: RQ4b-3_parallel_evaluation_and_training
  effective_date: '2026-10-21'
  track_a_source_run_id: rq4b3_adv_20261019_02
  track_b_child_run_id: rq4b3_adv_20261021_01
  execution_lineage:
    relationship: track_b_child_reads_no_track_a_artifacts_during_fit
    track_a_source_is_immutable: true
    track_a_source_protocol: rq4b_adversarial_realmlp_v2_cv_epoch_selection
    track_b_child_protocol: rq4b_adversarial_realmlp_v3_parallel_evaluation_training
    cross_run_aggregation_status: blocked_until_gate_c_lineage_validation
    prohibit_resume_across_code_change: true
  supersedes: [rq4b_adversarial_realmlp_v2_cv_epoch_selection_large_protocol, temporary_12_fit_priority_policy_20261020]
  original_and_amended_protocols_must_remain_distinguishable: true
  tracks:
    evaluation: {name: track_a, dependency_on_additional_training: none, reads: frozen_completed_seed_42_development_artifacts_only, writes: authorized_evaluation_outputs_only, prohibit_refit_or_reinference: true}
    additional_training: {name: track_b, dependency_on_track_a: none, maximum_units: 4, sequential_queue_if_capacity_requires: true}
  test_remains_closed_until_gate_c: true
development_units:
  maximum_total: 9
  retained_completed:
    - {lambda: 0, seed: 42}
    - {lambda: 0.01, seed: 42}
    - {lambda: 0.03, seed: 42}
    - {lambda: 0.1, seed: 42}
    - {lambda: 0.3, seed: 42}
  retained_completed_count: 5
  additional_approved:
    - {lambda: 0, seed: 43}
    - {lambda: 0.01, seed: 43}
    - {lambda: 0, seed: 44}
    - {lambda: 0.01, seed: 44}
  additional_approved_count: 4
  lambda_1_execution_status: removed_from_queue
  screening_only_at_seed_42: [0.03, 0.1, 0.3]
  replicated_across_seeds_42_43_44: [0, 0.01]
  prohibit_cartesian_expansion: true
  prohibit_additional_lambda_seed_architecture_or_method: true
execution_gates:
  gate_a_technical_readiness:
    status: PASS
    decision_date: '2026-10-21'
    scope: v3_sparse_nine_unit_alignment_and_track_b_child_run_lineage
    blocking_for: track_b
    eod3_evaluation_required: false
    required: [approved_amended_scope_and_budget, valid_configuration_and_implementation, correct_run_id_seed_lambda_assignment, nonconflicting_artifact_paths_and_write_ownership, completed_artifacts_protected_from_overwrite, no_unresolved_critical_implementation_or_lineage_issue]
  gate_b_development_execution_and_validation:
    blocking_for_each_new_unit: true
    required: [five_cv_folds, full_train_refit_under_existing_epoch_selection_policy, development_validation, integrity_and_prediction_count_checks, artifact_and_receipt_validation]
    incomplete_unit_may_be_marked_completed: false
  gate_c_final_evaluation_and_selection:
    blocking: true
    required: [evaluation_for_all_retained_completed_development_units, eod3, paired_same_seed_delta_eod3, balanced_accuracy_cost, eod6_safety_audit, training_seed_variability, training_stability_and_integrity, approved_three_seed_selection_policy, explicit_statistical_limitations]
    prohibit_test_before_pass: true
"""
)
APPROVED_CONTRACT.update(_V3_AMENDMENT)
_V4_CONTINUATION: dict[str, Any] = yaml.safe_load(
    """
protocol_version: rq4b_adversarial_realmlp_v4_seed44_continuation
decision_revision: rq4b3_seed44_continuation_20261022
seed44_continuation:
  effective_date: '2026-10-22'
  run_id: rq4b3_adv_20261022_01
  role: track_b_seed44_continuation
  protocol_version: rq4b_adversarial_realmlp_v4_seed44_continuation
  decision_revision: rq4b3_seed44_continuation_20261022
  authorized_units: [{lambda: 0, seed: 44}, {lambda: 0.01, seed: 44}]
  source_runs:
    track_a_seed42:
      run_id: rq4b3_adv_20261019_02
      protocol_version: rq4b_adversarial_realmlp_v2_cv_epoch_selection
      manifest_sha256: b84f5d018cd308a97d0918dd9ba336e8f449a8b5c16a7ca319c6207829c004bc
      receipt_sha256:
        stages/dev_l0_s42/receipt.json: 0734ace391cee86112039640a5b6cc6c7c63a98e0da63c697a931859b3435e1f
        stages/devval_l0_s42/receipt.json: 99f6d38f9cb05a095422f8ec4a0e4b612593e09f207ffe90587f33a266d625b6
        stages/dev_l0p01_s42/receipt.json: 2fbf721f352fef151c00e512e288223411d04b48f84a19cb11864323eaa001b3
        stages/devval_l0p01_s42/receipt.json: 90da727f27b88e42626a9719bd9f64abae8bc0895c12d3b62807cb88e56fd645
        stages/dev_l0p03_s42/receipt.json: 0fe741be9b855bd5ea013c720677cfd2359e6728604373606ed27e8e1d6c7843
        stages/devval_l0p03_s42/receipt.json: e6e7c55bd4b3921c8b8bdda9cef1c6a270e404479a0e224242ebaa48157ab5dd
        stages/dev_l0p1_s42/receipt.json: 54a26511089fc82d5fbce445246c6205678900216faaba38555e205c6990de11
        stages/devval_l0p1_s42/receipt.json: fed37b989d0e3ae900ac9ee4a3e988860e38730491938fe5718bfa62facbfb94
        stages/dev_l0p3_s42/receipt.json: 98e9aaa0de00f4f10469b12f24867432ef4a597d86a2459c45171a1a9b79e377
        stages/devval_l0p3_s42/receipt.json: 8ec1b31a442f8c3070a7f66cde94e78b0723559b0fcddd0131c526a00efbc6e4
    track_b_seed43:
      run_id: rq4b3_adv_20261021_01
      protocol_version: rq4b_adversarial_realmlp_v3_parallel_evaluation_training
      manifest_sha256: 315bd175e1239230992c459fb937f8de9ca4f995527a5e0a6aef63f9e3f11dfc
      receipt_sha256:
        stages/dev_l0_s43/receipt.json: 3525c6605e5be3f4d59472fa3f538c7c3daa94e48df742b7c7450918a4b43f84
        stages/devval_l0_s43/receipt.json: b34dd1b9e0a322a35a20748361e34517de666f8080b1f36d4538c64a60d423ca
        stages/dev_l0p01_s43/receipt.json: cb3cd1bf6c75825bd7d44716799b709f68b4fa4ea3a9199bf002a80e4d09724a
        stages/devval_l0p01_s43/receipt.json: 043e4fc03fa0e743dabce9d87be8deaac46658ca33aab1204a94f3a9d75192a5
      failure_receipt_sha256:
        failures/dev_l0_s43.ca0ec0704255402b9f2a7fb8779f57fa/FAILURE_RECEIPT.json: 1b8c3630e3e39b43e9c3121cec4bbe6a56ae5d604cdfa4288f2ba63398e74115
  source_runs_are_immutable: true
  fitted_state_reuse: prohibited
  migration_policy: no_migration_fresh_seed44_fit_state
  cross_run_aggregation_status: blocked_until_gate_c_lineage_validation
  source_digest_mismatch_action: stop_before_run_creation_or_open
  code_or_config_change_after_creation_action: preserve_run_and_require_new_run_id
  seed44_execution_order: [{lambda: 0, seed: 44}, verify_published_artifacts, {lambda: 0.01, seed: 44}]
  selection_final_test_and_package_remain_closed: true
"""
)
APPROVED_CONTRACT.update(_V4_CONTINUATION)
APPROVED_CONTRACT["method_contract_state"] = METHOD_CONTRACT_STATE
APPROVED_CONTRACT["execution_gates"]["gate_a_technical_readiness"].update(
    {"decision_date": "2026-10-22", "scope": "v4_seed44_continuation_and_pinned_source_lineage"}
)
APPROVED_CONTRACT["realmlp_recipe"].update(
    {"lambda_grid": list(LAMBDA_GRID), "development_seeds": list(DEVELOPMENT_SEEDS), "final_seeds": list(FINAL_SEEDS)}
)
APPROVED_CONTRACT["candidate_selection"] = {
    "policy_status": "approved_evaluation_only_three_seed_validation_selection",
    "evaluation_partition": "validation",
    "matched_control": "same_seed_lambda_zero",
    "eligibility_metric": "mean_matched_delta_balanced_accuracy",
    "eligibility_minimum": -0.01,
    "rq4b_3_objective": "minimum_mean_eod3",
    "eod6_may_select_lambda": False,
    "exact_tie_break": "smaller_lambda",
    "lambda_zero_may_win": True,
    "use_test_for_selection": False,
    "same_partition_as_threshold_calibration": True,
    "exploratory_optimism_limitation_required": True,
    "prior_four_of_five_acceptance_satisfied_by_three_seeds": False,
    "arbitrary_two_of_three_rule_prohibited": True,
}
APPROVED_CONTRACT["gate_c_aggregation"] = {
    "run_id": "rq4b3_gatec_20261024_01",
    "protocol_version": "rq4b_adversarial_gate_c_evaluation_only_v1",
    "artifact_namespace": "artifacts/dataset_b/rq4/rq4b/gate_c_aggregation",
    "source_run_ids": [TRACK_A_SOURCE_RUN_ID, TRACK_B_CHILD_RUN_ID, SEED44_CONTINUATION_RUN_ID],
    "evaluation_only": True, "prohibit_fit": True, "prohibit_inference": True,
    "test_status": "blocked_pending_separate_methodological_gate",
    "test_status_scope": "status_at_gate_c_publication",
    "current_test_contract_status": "authorized_pending_implementation",
    # Exact immutable evidence for all nine units (18 receipts: 45 CV fits and 9 refits).
    "validation_rows": 43995,
    "validation_record_id_sha256": "f5331875e449fc2f940de5a845483d5ea0830ae2b5a8501df98e9814a2fa5875",
    "source_pins": {
        TRACK_A_SOURCE_RUN_ID: {
            "manifest_sha256": "b84f5d018cd308a97d0918dd9ba336e8f449a8b5c16a7ca319c6207829c004bc",
            "receipt_sha256": copy.deepcopy(APPROVED_CONTRACT["seed44_continuation"]["source_runs"]["track_a_seed42"]["receipt_sha256"]),
        },
        TRACK_B_CHILD_RUN_ID: {
            "manifest_sha256": "315bd175e1239230992c459fb937f8de9ca4f995527a5e0a6aef63f9e3f11dfc",
            "receipt_sha256": copy.deepcopy(APPROVED_CONTRACT["seed44_continuation"]["source_runs"]["track_b_seed43"]["receipt_sha256"]),
        },
        SEED44_CONTINUATION_RUN_ID: {
            "manifest_sha256": "f6ebf429d4d0a5ec2b4abd08e247a6afa75b381da2da38f78cd5185b70a53249",
            "receipt_sha256": {
                "stages/dev_l0_s44/receipt.json": "5f5bf597283ffea13b05d42c147371b5bae92349cf0250a5db0238bf73a41200",
                "stages/devval_l0_s44/receipt.json": "82e34c7f7ade0926ac9841b6b0f32427248fec01cbba8fb9ea73e8c5dc35f648",
                "stages/dev_l0p01_s44/receipt.json": "e3c39216e7c1e09322fada1c5be5c1dd087830cd493a32463a1556e01e5ed021",
                "stages/devval_l0p01_s44/receipt.json": "6053990156cfdd8a5fb5da41f70121faae631c421db6706ea1fc78f1fced5f4a",
            },
        },
    },
}
APPROVED_CONTRACT["decision_revision"] = DECISION_REVISION
APPROVED_CONTRACT["method_contract_state"] = METHOD_CONTRACT_STATE
APPROVED_CONTRACT["uncertainty"]["within_seed"]["indices_sha256"] = (
    "2ee377b6fbae8a3a6453a7d8f2f4a0a3f1ce61374f1258ef882589b84230fee1"
)
APPROVED_CONTRACT["scientific_assessment"] = {
    "ownership": "user", "automated_research_conclusion": "prohibited",
    "automated_primary_criterion": "prohibited", "automated_strong_support_classification": "prohibited",
    "code_scope": "export_execution_status_lineage_and_neutral_metrics_only",
    "delta_direction": "lambda_0_01_minus_same_seed_lambda_zero", "final_seed_count": 3,
    "stability_status": "not_assessed", "scientific_conclusion_separate_from_execution_status": True,
    "interpretation_must_not_be_inferred_from_execution_success": True,
}
APPROVED_CONTRACT["reporting_contract"] = {
    "required_evidence": [
        "execution_status", "evaluation_arms", "source_and_gate_c_lineage", "test_bootstrap_index_lineage",
        "balanced_accuracy", "eod3", "eod6_audit", "group_tpr_fpr_and_counts",
        "matched_delta_ba_eod3_eod6", "decision_change_counts_and_rates", "paired_bootstrap_replicates",
        "confidence_interval_endpoints", "across_seed_summaries", "validation_test_generalization_gaps",
    ],
    "prohibit_best_seed_selection": True, "prohibit_unfavorable_seed_exclusion": True,
    "prohibit_test_driven_budget_extension": True, "prohibit_automated_research_interpretation": True,
}
APPROVED_CONTRACT["status_contract"] = {
    "execution_status": ["completed", "failed", "infeasible"], "stability_status": ["not_assessed"],
    "metric_values_are_separate_from_execution_status": True, "research_interpretation_owner": "user",
    "primary_criterion_output": "prohibited_for_rq4b3_test", "strong_support_output": "prohibited_for_rq4b3_test",
    "fields_are_independent": True,
    "reason_codes": ["none", "prerequisite_rq1_paired_bootstrap_missing", "technical_spike_pending",
                     "pytabkit_revision_unpinned", "adversarial_hook_unavailable", "score_tensor_unresolved",
                     "nonfinite_preflight", "serialization_reload_failed", "lambda_zero_equivalence_failed",
                     "required_group_metric_invalid", "insufficient_valid_bootstrap_replicates", "ba_budget_exceeded",
                     "unstable_diagnostics", "runtime_failure", "capability_inconclusive"],
}
APPROVED_CONTRACT["fit_budget"] = {
    "train_lambda_zero_controls_for_all_final_seeds": True,
    "reuse_lambda_zero_controls_across_g3_g6": True,
    "reuse_selected_lambda_artifacts_for_seeds": [42, 43, 44],
    "retrain_reused_selected_lambda_artifacts": False,
    "committed_rq4b_3_maximum_development_units": 9,
    "retained_completed_development_units": 5,
    "maximum_additional_development_units": 4,
    "rq4b_3_maximum_cv_fold_fits_excluding_spike_and_retries": 45,
    "rq4b_3_maximum_full_train_refits_excluding_spike_and_retries": 9,
    "rq4b_3_maximum_total_fits_excluding_spike_and_retries": 54,
    "capability_probe_fits_included_in_54": False,
    "budget_basis": "5_retained_seed42_units_plus_4_additional_control_and_lambda_0_01_replications",
    "superseded_budget_basis": "18_development_combinations_plus_up_to_4_final_seed_combinations",
    "superseded_maximum_total_fits": 132,
    "future_rq4b_6_additional_primary_fits_approx": 17,
    "future_optional_weighted_rq4b_6_additional_fits": 5,
    "future_rq4b_6_counts_are_refit_only_cv_budget_pending": True,
    "rq4b_6_outside_committed_budget": True,
}
APPROVED_CONTRACT["uncertainty"]["across_seed"]["required_seed_count"] = 3
UNLOCKED_PATHS: tuple[tuple[str, ...], ...] = (
    ("status",),
    ("technical_spike",),
    ("execution_governance",),
    ("prerequisites", "implementation_in_this_task"),
    ("capability_assessment", "synthetic_correctness_gate", "fixture_source_and_generation", "observed_gate_result"),
    ("configurations", "RQ4b-3", "readiness"),
    ("configurations", "RQ4b-6", "readiness"),
    # Explicitly replaced by the 2026-10-24 Test-output contract. These are
    # validated field-by-field below instead of against the historical snapshot.
    ("test_evaluation",),
)
LOCKED_FIELDS: dict[tuple[str, ...], Any] = {(key,): value for key, value in APPROVED_CONTRACT.items()}


class RQ4bInputError(RuntimeError):
    """Raised when an RQ4b contract or input boundary is violated."""


class RQ4bAccessError(RQ4bInputError):
    """Raised when a stage requests a partition or source it may not open."""


def _require(condition: bool, message: str, error: type[RQ4bInputError] = RQ4bInputError) -> None:
    if not condition:
        raise error(message)


def _strict_equal(actual: Any, expected: Any) -> bool:
    """Type-aware equality: booleans never equal numbers; lists keep order."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(actual) is bool and type(expected) is bool and actual == expected
    if isinstance(expected, (int, float)):
        return isinstance(actual, (int, float)) and math.isfinite(actual) and actual == expected
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(_strict_equal(actual[key], expected[key]) for key in expected)
        )
    if isinstance(expected, (list, tuple)):
        return (
            isinstance(actual, (list, tuple))
            and len(actual) == len(expected)
            and all(_strict_equal(a, e) for a, e in zip(actual, expected))
        )
    return type(actual) is type(expected) and actual == expected


def _lookup(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    node: Any = payload
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            return _MISSING
        node = node[key]
    return node


_MISSING = object()


def _leaf_conflicts(actual: Any, expected: Any, path: tuple[str, ...]) -> list[str]:
    """Dotted paths where ``actual`` is missing, extra or differs from ``expected``."""
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        found = []
        for key in expected:
            found += _leaf_conflicts(actual[key], expected[key], (*path, str(key))) if key in actual else [".".join((*path, str(key))) + " (missing)"]
        found += [".".join((*path, str(key))) + " (unexpected)" for key in actual if key not in expected]
        return found
    return [] if _strict_equal(actual, expected) else [".".join(path)]


def locked_contract_conflicts(config: Mapping[str, Any]) -> list[str]:
    """Compare every locked section in full; only UNLOCKED_PATHS are ignored."""
    candidate = copy.deepcopy(dict(config))
    expected = copy.deepcopy(APPROVED_CONTRACT)
    for path in UNLOCKED_PATHS:
        for payload in (candidate, expected):
            parent = _lookup(payload, path[:-1])
            if isinstance(parent, dict):
                parent.pop(path[-1], None)
    return _leaf_conflicts(candidate, expected, ())


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


STAGES = (
    "cv_epoch_selection",
    "full_train_refit",
    "validation_calibration_selection",
    "test_evaluation",
    "capability_probe",
    "synthetic_capability_gate",
)
TRAIN_ONLY_STAGES = ("cv_epoch_selection", "full_train_refit")
STAGE_RESOURCES: dict[str, frozenset[str]] = {
    "cv_epoch_selection": frozenset({"split_manifest", "feature_contract", "split_train", "derived_race_source"}),
    "full_train_refit": frozenset({"split_manifest", "feature_contract", "split_train", "derived_race_source"}),
    "validation_calibration_selection": frozenset(
        {"split_manifest", "feature_contract", "split_validation", "derived_race_source"}
    ),
    "test_evaluation": frozenset(
        {"split_manifest", "feature_contract", "split_test", "derived_race_source", "locked_test_bootstrap_indices"}
    ),
    "capability_probe": frozenset({"frozen_probe_validation_predictions", "derived_race_source"}),
    "synthetic_capability_gate": frozenset(),
}


@dataclass(frozen=True)
class RQ4bContract:
    """Validated execution-critical RQ4b-3 settings with resolved input paths."""

    protocol_version: str
    decision_revision: str
    track_a_source_run_id: str
    track_b_child_run_id: str
    production_run_id: str
    production_run_role: str
    source_run_evidence: Mapping[str, Any]
    aggregation_run_id: str
    aggregation_protocol_version: str
    aggregation_namespace: Path
    aggregation_source_run_ids: tuple[str, ...]
    aggregation_source_pins: Mapping[str, Any]
    aggregation_validation_rows: int
    aggregation_validation_record_id_sha256: str
    test_run_id: str
    test_protocol_version: str
    test_namespace: Path
    test_evaluation_arms: tuple[Mapping[str, Any], ...]
    test_evaluation_units: tuple[tuple[float, int], ...]
    test_matched_comparisons: tuple[tuple[int, float, float], ...]
    test_source_pins: Mapping[str, Any]
    test_gate_c_manifest_sha256: str
    test_gate_c_receipt_sha256: str
    test_gate_c_selection_sha256: str
    test_bootstrap_path: Path
    test_bootstrap_sha256: str
    test_bootstrap_array: str
    test_bootstrap_replicates: int
    test_required_output_files: Mapping[str, tuple[str, ...]]
    g3: tuple[str, ...]
    g6: tuple[str, ...]
    coverage_only: tuple[str, ...]
    lambda_grid: tuple[float, ...]
    development_seeds: tuple[int, ...]
    final_seeds: tuple[int, ...]
    retained_development_units: tuple[tuple[float, int], ...]
    additional_development_units: tuple[tuple[float, int], ...]
    development_units: tuple[tuple[float, int], ...]
    authorized_production_units: tuple[tuple[float, int], ...]
    selection_policy_status: str
    cv_folds: int
    cv_random_state: int
    epochs_per_fold: int
    batch_size: int
    predict_batch_size: int
    device: str
    probability_sum_atol: float
    artifact_namespace: Path
    resources: Mapping[str, Path] = field(repr=False)
    split_rows: Mapping[str, int] = field(default_factory=lambda: dict(SPLIT_V1_ROWS))

    @property
    def allowed_race_groups(self) -> frozenset[str]:
        return frozenset((*self.g6, *self.coverage_only))


def validate_contract(
    config: Mapping[str, Any],
    rq3_protocol: Mapping[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
) -> RQ4bContract:
    """Reject malformed, missing, or conflicting execution-critical settings."""

    _require(isinstance(config, Mapping), "RQ4b config must be a mapping.")
    _require(isinstance(rq3_protocol, Mapping), "RQ3 denominator/audit protocol must be a mapping.")
    conflicts = locked_contract_conflicts(config)
    _require(not conflicts, f"RQ4b contract fields differ from the locked contract: {conflicts}")

    amendment = config["contract_amendment"]
    _require(amendment["track_a_source_run_id"] == TRACK_A_SOURCE_RUN_ID, "Track A source run differs from the execution-lineage amendment.")
    _require(amendment["track_b_child_run_id"] == TRACK_B_CHILD_RUN_ID, "Track B child run differs from the execution-lineage amendment.")
    _require(amendment["track_a_source_run_id"] != amendment["track_b_child_run_id"], "Track A source and Track B child must be distinct runs.")
    lineage = amendment["execution_lineage"]
    _require(lineage["track_a_source_is_immutable"] is True, "Track A source run must remain immutable.")
    _require(lineage["track_a_source_protocol"] != lineage["track_b_child_protocol"], "Historical and amended protocols must remain distinguishable.")
    _require(lineage["track_b_child_protocol"] == "rq4b_adversarial_realmlp_v3_parallel_evaluation_training", "Track B child protocol differs from its published v3 lineage.")
    _require(lineage["cross_run_aggregation_status"] == "blocked_until_gate_c_lineage_validation", "Cross-run aggregation must remain blocked until Gate C lineage validation.")
    _require(lineage["prohibit_resume_across_code_change"] is True, "Run resumption across code changes must remain prohibited.")
    governance = config["execution_governance"]
    launch = governance["production_launch_scope"]
    continuation = config["seed44_continuation"]
    authorized_production_units = tuple((float(unit["lambda"]), int(unit["seed"])) for unit in launch["units"])
    continuation_units = tuple((float(unit["lambda"]), int(unit["seed"])) for unit in continuation["authorized_units"])
    _require(continuation["run_id"] == SEED44_CONTINUATION_RUN_ID, "Seed-44 continuation run identity differs from the approved assignment.")
    _require(continuation["role"] == SEED44_CONTINUATION_ROLE, "Seed-44 continuation role differs from the approved assignment.")
    _require(continuation["protocol_version"] == config["protocol_version"] == PROTOCOL_VERSION, "Seed-44 continuation protocol differs from the active protocol.")
    _require(config["decision_revision"] == DECISION_REVISION, "Active decision revision differs from the Test-output contract.")
    _require(continuation["decision_revision"] == "rq4b3_seed44_continuation_20261022",
             "Historical seed-44 continuation decision revision differs.")
    _require(continuation_units == SEED44_PRODUCTION_UNITS, "Seed-44 continuation must contain exactly the two approved units.")
    _require(continuation["source_runs_are_immutable"] is True, "Continuation source runs must remain immutable.")
    _require(continuation["fitted_state_reuse"] == "prohibited", "Continuation must prohibit fitted-state reuse.")
    _require(continuation["migration_policy"] == "no_migration_fresh_seed44_fit_state", "Continuation migration policy must require fresh seed-44 fit state.")
    _require(continuation["cross_run_aggregation_status"] == "blocked_until_gate_c_lineage_validation", "Continuation aggregation must remain blocked until Gate C.")
    source_runs = continuation["source_runs"]
    _require(source_runs["track_a_seed42"]["run_id"] == TRACK_A_SOURCE_RUN_ID, "Continuation Track A source identity differs.")
    _require(source_runs["track_b_seed43"]["run_id"] == TRACK_B_CHILD_RUN_ID, "Continuation seed-43 source identity differs.")
    _require(governance["production_launch_authorized"] is True, "The current scoped production launch must be explicitly authorized.")
    _require(launch["run_id"] == SEED44_CONTINUATION_RUN_ID, "Production launch scope differs from the seed-44 continuation run.")
    _require(launch["role"] == SEED44_CONTINUATION_ROLE, "Production launch role differs from the seed-44 continuation role.")
    _require(authorized_production_units == SEED44_PRODUCTION_UNITS == continuation_units, "Production launch scope must contain exactly the two authorized seed-44 units.")
    _require(launch["require_preflight_and_capability_pass"] is True, "Production launch must require preflight and capability PASS.")
    _require(launch["prohibit_automatic_retry"] is True, "Automatic retry must remain prohibited.")
    _require(launch["require_lambda_zero_artifact_verification_before_lambda_0_01"] is True, "Lambda 0 artifacts must be verified before lambda 0.01 is opened.")
    _require(launch["selection_final_and_test_remain_closed"] is True, "Selection, final and Test must remain closed.")

    groups = config["groups"]
    g3, g6 = tuple(groups["g3"]), tuple(groups["g6"])
    coverage = tuple(groups["coverage_only_outside_g6"])
    _require(len(set(g6)) == len(g6) and g6[: len(g3)] == g3, "G3 must be the ordered prefix of a unique G6.")
    _require(not set(g6) & set(coverage), "Coverage-only groups must lie outside G6.")

    recipe = config["realmlp_recipe"]
    lambdas = tuple(recipe["lambda_grid"])
    _require(lambdas[0] == 0 and all(b > a for a, b in zip(lambdas, lambdas[1:])), "Lambda grid must start at 0 and increase strictly.")
    dev, final = tuple(recipe["development_seeds"]), tuple(recipe["final_seeds"])
    _require(final[: len(dev)] == dev and len(set(final)) == len(final), "Development seeds must be the ordered prefix of unique final seeds.")
    _require(
        tuple(config["fit_budget"]["reuse_selected_lambda_artifacts_for_seeds"]) == dev,
        "Selected-lambda artifact reuse must cover exactly the development seeds.",
    )

    selection = config["epoch_selection"]
    folds, epochs = selection["splitter"]["folds"], selection["epochs_per_fold"]
    _require(selection["required_finite_fold_epoch_values"] == folds * epochs, "Required CV values must equal folds x epochs.")
    _require(
        selection["full_train_refit"]["schedule_horizon"]["epochs"] == epochs,
        "Refit schedule horizon must equal the CV epoch horizon.",
    )
    units = config["development_units"]
    retained = tuple((float(unit["lambda"]), int(unit["seed"])) for unit in units["retained_completed"])
    additional = tuple((float(unit["lambda"]), int(unit["seed"])) for unit in units["additional_approved"])
    development_units = (*retained, *additional)
    _require(retained == RETAINED_DEVELOPMENT_UNITS, "Retained development units differ from the amended contract.")
    _require(additional == ADDITIONAL_DEVELOPMENT_UNITS, "Additional development units differ from the amended contract.")
    _require(len(set(development_units)) == len(development_units), "Development units must be unique.")
    _require(
        units["maximum_total"] == len(development_units)
        and units["retained_completed_count"] == len(retained)
        and units["additional_approved_count"] == len(additional),
        "Development-unit counts conflict with the amended roster.",
    )
    _require(all(lam in lambdas and seed in dev for lam, seed in development_units), "Development unit lies outside the locked lambda/seed roster.")
    _require(1.0 not in lambdas and units["lambda_1_execution_status"] == "removed_from_queue", "Lambda 1 must remain outside the amended execution queue.")
    _require(units["prohibit_cartesian_expansion"] is True, "Sparse development design must prohibit Cartesian expansion.")

    budget = config["fit_budget"]
    _require(len(development_units) == budget["committed_rq4b_3_maximum_development_units"], "Development-unit budget conflicts with the sparse roster.")
    _require(len(retained) == budget["retained_completed_development_units"], "Retained-unit budget conflicts with the sparse roster.")
    _require(len(additional) == budget["maximum_additional_development_units"], "Additional-unit budget conflicts with the sparse roster.")
    _require(len(development_units) == budget["rq4b_3_maximum_full_train_refits_excluding_spike_and_retries"], "Refit budget conflicts with the sparse roster.")
    _require(len(development_units) * folds == budget["rq4b_3_maximum_cv_fold_fits_excluding_spike_and_retries"], "CV fit budget conflicts with the sparse roster.")
    _require(
        len(development_units) * (folds + 1) == budget["rq4b_3_maximum_total_fits_excluding_spike_and_retries"],
        "Total fit budget conflicts with CV plus refit budgets.",
    )
    policy_status = config["candidate_selection"]["policy_status"]
    _require(policy_status == "approved_evaluation_only_three_seed_validation_selection", "Gate C selection must use the approved evaluation-only three-seed Validation policy.")
    assessment = config["scientific_assessment"]
    _require(assessment["ownership"] == "user", "Research interpretation must remain user-owned.")
    _require(assessment["automated_research_conclusion"] == "prohibited", "Code must not assign a research conclusion.")
    _require(assessment["automated_primary_criterion"] == "prohibited", "Code must not assign a primary research criterion.")
    _require(assessment["automated_strong_support_classification"] == "prohibited", "Code must not assign strong support.")
    grid = config["threshold_calibration"]["candidates"]
    _require(round((grid["stop"] - grid["start"]) / grid["step"]) + 1 == grid["count"], "Threshold grid count conflicts with its bounds.")
    fractions = config["capability_assessment"]["real_probe"]["split"]["fractions"]
    _require(math.isclose(sum(fractions.values()), 1.0, rel_tol=0.0, abs_tol=1e-12), "Probe split fractions must sum to 1.")
    synthetic = config["capability_assessment"]["synthetic_correctness_gate"]["fixture_source_and_generation"]
    _require(
        synthetic["rows"] == synthetic["rows_per_group"] * len(g3) == synthetic["rows_per_group_and_y"] * 2 * len(g3)
        and len(synthetic["centers_in_g3_order"]) == len(g3),
        "Synthetic fixture counts conflict with the G3 roster.",
    )

    audit = rq3_protocol.get("audit_data")
    race = _lookup(rq3_protocol, ("attributes", "derived_race"))
    _require(isinstance(audit, Mapping) and isinstance(race, Mapping), "RQ3 protocol lacks audit_data or derived_race.")
    _require(audit.get("raw_source") == DERIVED_RACE_SOURCE_PATH, "RQ3 derived_race source differs from the locked raw source.")
    _require(audit.get("join_key") == "record_id" and audit.get("join_method") == DERIVED_RACE_JOIN_METHOD, "RQ3 derived_race join differs.")
    rq3_groups = {
        str(race.get("reference_group")),
        *map(str, race.get("primary_comparisons", ())),
        *map(str, race.get("secondary_comparisons", ())),
    }
    _require(rq3_groups == set(g6), "RQ3 compared race groups conflict with RQ4b G6.")
    _require(set(map(str, race.get("coverage_only_groups", ()))) == set(coverage), "RQ3 coverage-only race groups conflict with RQ4b.")

    root = Path(project_root)
    probe_path = config["capability_assessment"]["real_probe"]["source_score_extraction"]["prediction_path"]
    resources = {
        "split_manifest": root / SPLIT_MANIFEST_PATH,
        "feature_contract": root / FEATURE_CONTRACT_PATH,
        **{f"split_{partition}": root / path for partition, path in SPLIT_V1_PATHS.items()},
        "derived_race_source": root / DERIVED_RACE_SOURCE_PATH,
        "frozen_probe_validation_predictions": root / probe_path,
        "locked_test_bootstrap_indices": root / config["uncertainty"]["within_seed"]["indices"],
    }
    runtime = config["runtime"]
    aggregation = config["gate_c_aggregation"]
    _require(tuple(aggregation["source_run_ids"]) == (TRACK_A_SOURCE_RUN_ID, TRACK_B_CHILD_RUN_ID, SEED44_CONTINUATION_RUN_ID), "Gate C aggregation must pin exactly the three immutable source runs in order.")
    _require(aggregation["evaluation_only"] is True and aggregation["prohibit_fit"] is True and aggregation["prohibit_inference"] is True, "Gate C aggregation must remain evaluation-only.")
    _require(aggregation["test_status"] == "blocked_pending_separate_methodological_gate", "Gate C aggregation must keep Test blocked.")
    _require(aggregation["test_status_scope"] == "status_at_gate_c_publication", "Gate C Test status must remain historical publication state.")
    _require(aggregation["current_test_contract_status"] == "authorized_pending_implementation", "Current Test contract status differs.")
    _require(aggregation["validation_rows"] == SPLIT_V1_ROWS["validation"], "Gate C Validation population size differs from the locked split.")
    pins = aggregation["source_pins"]
    _require(tuple(pins) == tuple(aggregation["source_run_ids"]), "Gate C source pins must cover exactly the three source runs in order.")
    for label, run_id in (("track_a_seed42", TRACK_A_SOURCE_RUN_ID), ("track_b_seed43", TRACK_B_CHILD_RUN_ID)):
        evidence = source_runs[label]
        _require(pins[run_id]["manifest_sha256"] == evidence["manifest_sha256"] and pins[run_id]["receipt_sha256"] == evidence["receipt_sha256"],
                 f"Gate C pins for {run_id} differ from the continuation source evidence.")
    for run_id, seed in zip(aggregation["source_run_ids"], dev):
        expected = {f"stages/{prefix}_l{f'{lam:g}'.replace('.', 'p')}_s{seed}/receipt.json"
                    for lam, unit_seed in development_units if unit_seed == seed for prefix in ("dev", "devval")}
        _require(set(pins[run_id]["receipt_sha256"]) == expected, f"Gate C receipt pins for {run_id} do not cover exactly its seed-{seed} development units.")

    test = config["test_evaluation"]
    _require(test["authorization_status"] == "authorized_pending_implementation" and test["execution_status"] == "not_run",
             "Test contract must be authorized but not executed.")
    _require(test["run_id"] == TEST_RUN_ID and test["protocol_version"] == TEST_PROTOCOL_VERSION,
             "Test run identity or protocol differs from the locked contract.")
    _require(test["artifact_namespace"] == "artifacts/dataset_b/rq4/rq4b/test_evaluation"
             and test["run_path"] == f"artifacts/dataset_b/rq4/rq4b/test_evaluation/{TEST_RUN_ID}",
             "Test artifact namespace or run path differs from the locked contract.")
    _require(test["cross_run_evaluation"] is True, "Test evaluation must be cross-run.")
    for field_name in ("prohibit_fit", "prohibit_refit", "prohibit_cv", "prohibit_threshold_recalibration",
                       "prohibit_bootstrap_index_regeneration"):
        _require(test[field_name] is True, f"Test contract must keep {field_name} enabled.")
    test_units = tuple(
        (float(arm["adversary_lambda"]), int(seed))
        for arm in test["evaluation_arms"] for seed in arm["seeds"]
    )
    _require(test_units == TEST_EVALUATION_UNITS, "Test evaluation arms differ from the locked nine-arm roster.")
    _require(test["evaluation_arm_count"] == len(TEST_EVALUATION_UNITS) == test["max_test_inferences"],
             "Test arm or inference count differs from nine.")
    expected_arm_headers = (
        ("gate_c_selected_control", 0.0, (42, 43, 44), "gate_c_selected_lambda"),
        ("user_requested_comparator", 0.01, (42, 43, 44), "explicit_user_test_request"),
        ("user_requested_screening", 0.03, (42,), "explicit_user_test_request"),
        ("user_requested_screening", 0.1, (42,), "explicit_user_test_request"),
        ("user_requested_screening", 0.3, (42,), "explicit_user_test_request"),
    )
    actual_arm_headers = tuple((str(arm["role"]), float(arm["adversary_lambda"]), tuple(arm["seeds"]), str(arm["provenance"]))
                               for arm in test["evaluation_arms"])
    _require(actual_arm_headers == expected_arm_headers, "Test arm roles or provenance differ from the locked contract.")
    comparisons = tuple(
        (int(row["seed"]), float(row["comparator_lambda"]), float(row["control_lambda"]))
        for row in test["matched_comparisons"]
    )
    _require(comparisons == TEST_MATCHED_COMPARISONS and test["matched_comparison_count"] == len(comparisons),
             "Test matched comparisons differ from the locked six-comparison roster.")
    lineage = test["lineage"]
    _require(lineage["gate_c_run_id"] == aggregation["run_id"], "Test lineage points to the wrong Gate C run.")
    _require(lineage["gate_c_manifest_sha256"] == GATE_C_MANIFEST_SHA256
             and lineage["gate_c_receipt_sha256"] == GATE_C_RECEIPT_SHA256,
             "Test Gate C manifest or receipt pin differs from the locked production evidence.")
    _require(tuple(lineage["source_pins"]) == tuple(aggregation["source_run_ids"]), "Test source pins must cover the three source runs in order.")
    for run_id in aggregation["source_run_ids"]:
        _require(lineage["source_pins"][run_id]["manifest_sha256"] == pins[run_id]["manifest_sha256"],
                 f"Test manifest pin for {run_id} differs from Gate C.")
        _require(lineage["source_pins"][run_id]["evaluated_arm_receipt_sha256"] == pins[run_id]["receipt_sha256"],
                 f"Test evaluated-arm receipt pins for {run_id} differ from Gate C.")
    bootstrap = lineage["locked_test_bootstrap_indices"]
    _require(bootstrap["path"] == config["uncertainty"]["within_seed"]["indices"], "Test bootstrap path differs from the uncertainty contract.")
    _require(bootstrap["sha256"] == config["uncertainty"]["within_seed"]["indices_sha256"], "Test bootstrap digest differs from the uncertainty contract.")
    _require(bootstrap["replicates"] == 1000 and bootstrap["matched_comparisons"] == 6
             and bootstrap["expected_total_replicate_rows"] == 6000 and bootstrap["regeneration_allowed"] is False,
             "Test bootstrap dimensions or regeneration policy differ.")
    boundary = test["interpretation_boundary"]
    _require(all(boundary[name] is True for name in (
        "execution_status_is_not_research_conclusion", "metric_values_are_neutral_outputs",
        "code_must_not_assign_research_pass_fail", "code_must_not_assign_effectiveness_or_support_conclusion")),
        "Test interpretation boundary is incomplete.")
    _require(boundary["research_interpretation_owner"] == "user"
             and boundary["machine_generated_research_interpretation_file"] == "prohibited",
             "Test interpretation ownership differs.")
    _require(test["delta_direction"] == "each_nonzero_lambda_minus_same_seed_lambda_0"
             and test["generalization_gap_direction"] == "test_minus_validation",
             "Test delta direction differs from the locked contract.")
    _require(test["group_metric_rosters"] == ["g3", "g6"]
             and test["decision_change_rosters"] == ["overall", "g3", "g6"]
             and test["overall_row_encoding"] == {"roster": "overall", "group": "__all__"},
             "Test group or decision-change roster differs from the locked contract.")
    _require(test["undefined_numeric_encoding"] is None and test["invalid_reason_encoding"] == "explicit_reason_code",
             "Test undefined-value encoding differs from the locked contract.")
    _require(test["test_read_during_contract_task"] is False
             and test["production_execution_requires_separate_explicit_launch"] is True,
             "Test access authorization boundary differs.")
    expected_output_columns = {
        "run_manifest.json": ("run_id", "protocol_version", "mode", "config_sha256", "code_sha256", "gate_c_pins", "source_pins", "bootstrap_index_pin", "evaluation_arms"),
        "stages/test_evaluation/receipt.json": ("status", "inputs", "outputs", "counts", "started_utc", "finished_utc"),
        "stages/test_evaluation/arm_metrics.csv": ("seed", "adversary_lambda", "role", "request_provenance", "source_run_id", "model_receipt_sha256", "validation_receipt_sha256", "threshold", "test_rows", "test_record_id_sha256", "balanced_accuracy", "eod3", "eod6_audit"),
        "stages/test_evaluation/group_metrics.csv": ("seed", "adversary_lambda", "roster", "group", "row_count", "actual_approved_count", "actual_denied_count", "predicted_approved_count", "predicted_denied_count", "tpr_approved", "fpr_approved", "tpr_valid", "fpr_valid", "invalid_reason"),
        "stages/test_evaluation/matched_deltas.csv": ("seed", "comparator_lambda", "control_lambda", "delta_balanced_accuracy", "delta_eod3", "delta_eod6"),
        "stages/test_evaluation/decision_changes.csv": ("seed", "comparator_lambda", "control_lambda", "roster", "group", "row_count", "unchanged_count", "changed_count", "denied_to_approved_count", "approved_to_denied_count", "change_rate"),
        "stages/test_evaluation/bootstrap_replicates.csv": ("seed", "comparator_lambda", "control_lambda", "replicate_id", "sample_size", "delta_balanced_accuracy", "delta_eod3", "delta_eod6", "balanced_accuracy_valid", "eod3_valid", "eod6_valid", "invalid_reasons"),
        "stages/test_evaluation/bootstrap_summary.csv": ("seed", "comparator_lambda", "control_lambda", "quantity", "valid_replicates", "invalid_replicates", "confidence_level", "lower", "upper", "quantile_method"),
        "stages/test_evaluation/across_seed_summary.csv": ("scope", "adversary_lambda", "comparator_lambda", "control_lambda", "quantity", "seed_count", "mean", "standard_deviation", "ddof", "min", "max", "range"),
        "stages/test_evaluation/generalization_gaps.csv": ("seed", "adversary_lambda", "quantity", "validation_value", "test_value", "test_minus_validation"),
        "stages/test_evaluation/execution_status.json": ("execution_status", "reason_codes", "test_rows", "inference_count", "fit_count", "refit_count", "cv_fit_count"),
    }
    actual_output_columns = {name: tuple(columns) for name, columns in test["required_output_files"].items()}
    _require(actual_output_columns == expected_output_columns, "Test required output schema differs from the locked contract.")
    _require(test["required_point_quantities"] == {
        "per_arm": ["balanced_accuracy", "eod3", "eod6_audit"],
        "per_group": ["tpr_approved", "fpr_approved", "row_count", "actual_approved_count", "actual_denied_count", "predicted_approved_count", "predicted_denied_count"],
        "matched_deltas": ["delta_balanced_accuracy", "delta_eod3", "delta_eod6"],
        "decision_changes": ["unchanged_count", "changed_count", "denied_to_approved_count", "approved_to_denied_count", "change_rate"],
        "bootstrap_deltas": ["delta_balanced_accuracy", "delta_eod3", "delta_eod6"],
        "across_seed_summaries": ["mean", "standard_deviation", "min", "max", "range"],
        "generalization_gaps": ["balanced_accuracy", "eod3", "eod6"],
    }, "Test required metric quantities differ from the locked contract.")
    _require(test["screening_single_seed_reporting"]["lambdas"] == [0.03, 0.1, 0.3]
             and test["screening_single_seed_reporting"]["seed"] == 42
             and test["screening_single_seed_reporting"]["across_seed_summary"] == "not_applicable_single_seed",
             "Test screening-arm reporting contract differs.")
    return RQ4bContract(
        protocol_version=config["protocol_version"],
        decision_revision=config["decision_revision"],
        track_a_source_run_id=amendment["track_a_source_run_id"],
        track_b_child_run_id=amendment["track_b_child_run_id"],
        production_run_id=continuation["run_id"],
        production_run_role=continuation["role"],
        source_run_evidence=copy.deepcopy(source_runs),
        aggregation_run_id=aggregation["run_id"],
        aggregation_protocol_version=aggregation["protocol_version"],
        aggregation_namespace=root / aggregation["artifact_namespace"],
        aggregation_source_run_ids=tuple(aggregation["source_run_ids"]),
        aggregation_source_pins=copy.deepcopy(pins),
        aggregation_validation_rows=aggregation["validation_rows"],
        aggregation_validation_record_id_sha256=aggregation["validation_record_id_sha256"],
        test_run_id=test["run_id"],
        test_protocol_version=test["protocol_version"],
        test_namespace=root / test["artifact_namespace"],
        test_evaluation_arms=tuple(copy.deepcopy(test["evaluation_arms"])),
        test_evaluation_units=test_units,
        test_matched_comparisons=comparisons,
        test_source_pins=copy.deepcopy(lineage["source_pins"]),
        test_gate_c_manifest_sha256=lineage["gate_c_manifest_sha256"],
        test_gate_c_receipt_sha256=lineage["gate_c_receipt_sha256"],
        test_gate_c_selection_sha256=GATE_C_SELECTION_SHA256,
        test_bootstrap_path=root / bootstrap["path"],
        test_bootstrap_sha256=bootstrap["sha256"],
        test_bootstrap_array=bootstrap["array"],
        test_bootstrap_replicates=bootstrap["replicates"],
        test_required_output_files={key: tuple(value) for key, value in test["required_output_files"].items()},
        g3=g3,
        g6=g6,
        coverage_only=coverage,
        lambda_grid=tuple(float(value) for value in lambdas),
        development_seeds=dev,
        final_seeds=final,
        retained_development_units=retained,
        additional_development_units=additional,
        development_units=development_units,
        authorized_production_units=authorized_production_units,
        selection_policy_status=policy_status,
        cv_folds=folds,
        cv_random_state=selection["splitter"]["random_state"],
        epochs_per_fold=epochs,
        batch_size=runtime["batch_size"],
        predict_batch_size=runtime["predict_batch_size"],
        device=runtime["device"],
        probability_sum_atol=PROBABILITY_SUM_ATOL,
        artifact_namespace=root / config["artifact_namespace"],
        resources=resources,
    )


def load_contract(path: Path = RQ4B_CONFIG_PATH, *, project_root: Path = PROJECT_ROOT) -> RQ4bContract:
    """Read the RQ4b YAML and its declared RQ3 source contract, then validate both."""

    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(config, Mapping), "RQ4b config must be a mapping.")
    source = config.get("denominators", {}).get("source_contract") if isinstance(config.get("denominators"), Mapping) else None
    _require(source == "config/dataset_b/rq3_protocol.yaml", "RQ4b denominator source contract is not the locked RQ3 protocol.")
    rq3_protocol = yaml.safe_load((Path(project_root) / source).read_text(encoding="utf-8"))
    return validate_contract(config, rq3_protocol, project_root=project_root)


# ---------------------------------------------------------------------------
# Stage-aware access
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageAccess:
    """Explicit allow-list of the partitions and sources one stage may open."""

    stage: str
    contract: RQ4bContract

    def __post_init__(self) -> None:
        _require(self.stage in STAGES, f"Unknown RQ4b stage {self.stage!r}.", RQ4bAccessError)

    @property
    def allowed_resources(self) -> frozenset[str]:
        return STAGE_RESOURCES[self.stage]

    def require_resource(self, resource: str) -> Path:
        _require(resource in self.contract.resources, f"Unknown RQ4b resource {resource!r}.", RQ4bAccessError)
        _require(
            resource in self.allowed_resources,
            f"RQ4b stage {self.stage!r} may not open {resource!r}.",
            RQ4bAccessError,
        )
        return self.contract.resources[resource]

    def require_partition(self, partition: str) -> Path:
        _require(partition in SPLIT_V1_ROWS, f"Unknown project partition {partition!r}.", RQ4bAccessError)
        return self.require_resource(f"split_{partition}")

    def require_path(self, path: str | Path) -> str:
        """Map a concrete path to a declared resource and check permission."""
        resolved = Path(path).resolve()
        matches = [name for name, declared in self.contract.resources.items() if declared.resolve() == resolved]
        _require(bool(matches), f"Path {path} is not a declared RQ4b input.", RQ4bAccessError)
        self.require_resource(matches[0])
        return matches[0]


def require_rq4b_output_path(path: str | Path, contract: RQ4bContract) -> Path:
    """Outputs may only go below the RQ4b artifact namespace (never baseline/RQ3/RQ4a)."""

    resolved = Path(path).resolve()
    namespace = contract.artifact_namespace.resolve()
    _require(
        resolved != namespace and namespace in resolved.parents,
        f"RQ4b output {path} is outside {namespace}.",
    )
    return resolved


# ---------------------------------------------------------------------------
# Identifiers, labels, partitions and rosters (pure checks)
# ---------------------------------------------------------------------------


def validate_record_ids(
    record_ids: Sequence[Any],
    label: str,
    *,
    pattern: re.Pattern[str] | None = RECORD_ID_PATTERN,
) -> tuple[str, ...]:
    values = list(record_ids)
    _require(all(isinstance(value, str) and value.strip() == value and value for value in values), f"{label} record_id values must be nonblank strings.")
    _require(len(set(values)) == len(values), f"{label} record_id values must be unique.")
    if pattern is not None:
        _require(all(pattern.fullmatch(value) for value in values), f"{label} record_id values do not match the stable ID pattern.")
    return tuple(values)


def validate_binary_labels(y: Sequence[Any], label: str, *, rows: int | None = None) -> np.ndarray:
    values = pd.to_numeric(pd.Series(list(y), dtype=object), errors="coerce")
    _require(values.notna().all() and values.isin([0, 1]).all(), f"{label} y_approved must be binary 0/1.")
    array = values.to_numpy(dtype=np.int8)
    if rows is not None:
        _require(len(array) == rows, f"{label} label count differs from its rows.")
    return array


def require_same_labels(expected: Sequence[Any], observed: Sequence[Any], label: str) -> None:
    left = validate_binary_labels(expected, f"{label} expected")
    right = validate_binary_labels(observed, f"{label} observed")
    _require(left.shape == right.shape and np.array_equal(left, right), f"{label} y_approved differs from the reference.")


def align_by_record_id(
    expected_ids: Sequence[str],
    frame: pd.DataFrame,
    label: str,
    *,
    require_same_order: bool,
    pattern: re.Pattern[str] | None = RECORD_ID_PATTERN,
) -> pd.DataFrame:
    """Exact one-to-one join on ``record_id``: no missing, extra, or duplicate IDs.

    With ``require_same_order`` a permuted frame is rejected; otherwise it is
    reordered to ``expected_ids`` and the original index is discarded.
    """

    expected = validate_record_ids(expected_ids, f"{label} expected", pattern=pattern)
    _require("record_id" in frame.columns, f"{label} lacks record_id.")
    observed = validate_record_ids(frame["record_id"].tolist(), label, pattern=pattern)
    missing = set(expected) - set(observed)
    extra = set(observed) - set(expected)
    _require(not missing, f"{label} is missing {len(missing)} expected record_id values.")
    _require(not extra, f"{label} has {len(extra)} unexpected record_id values.")
    if require_same_order:
        _require(observed == expected, f"{label} record_id order differs from the reference order.")
        return frame.reset_index(drop=True)
    positions = pd.Index(observed).get_indexer(list(expected))
    return frame.iloc[positions].reset_index(drop=True)


def validate_partition_membership(
    partitions: Mapping[str, Sequence[str]],
    *,
    expected_rows: Mapping[str, int] | None = None,
    pattern: re.Pattern[str] | None = RECORD_ID_PATTERN,
) -> dict[str, int]:
    """Partitions must be known, internally unique, pairwise disjoint and sized."""

    _require(set(partitions) <= set(SPLIT_V1_ROWS), f"Unknown partitions: {sorted(set(partitions) - set(SPLIT_V1_ROWS))}.")
    seen: dict[str, str] = {}
    counts: dict[str, int] = {}
    for name, ids in partitions.items():
        values = validate_record_ids(ids, name, pattern=pattern)
        overlap = [value for value in values if value in seen]
        _require(not overlap, f"{len(overlap)} record_id values of {name} also appear in another partition.")
        seen.update({value: name for value in values})
        counts[name] = len(values)
        if expected_rows is not None and name in expected_rows:
            _require(len(values) == expected_rows[name], f"{name} has {len(values)} rows, expected {expected_rows[name]}.")
    return counts


def require_ids_within(ids: Sequence[str], allowed_ids: Sequence[str], label: str) -> None:
    outside = set(ids) - set(allowed_ids)
    _require(not outside, f"{label} contains {len(outside)} record_id values outside its permitted population.")


def require_locked_roster(roster: Sequence[str], contract: RQ4bContract) -> tuple[str, ...]:
    """RQ4b-3 uses exactly the ordered G3 roster; G6 and any other roster are rejected."""

    _require(tuple(roster) == contract.g3, f"RQ4b-3 roster must be exactly {list(contract.g3)}; got {list(roster)}.")
    return contract.g3


def encode_roster(derived_race: Sequence[Any], roster: Sequence[str], contract: RQ4bContract) -> np.ndarray:
    """Code roster members by fixed position; known non-roster groups get -1."""

    fixed = require_locked_roster(roster, contract)
    values = list(derived_race)
    _require(all(isinstance(value, str) and value for value in values), "derived_race must be nonblank strings.")
    unknown = sorted(set(values) - contract.allowed_race_groups)
    _require(not unknown, f"derived_race contains groups outside the contract: {unknown}.")
    lookup = {group: code for code, group in enumerate(fixed)}
    return np.asarray([lookup.get(value, OUTSIDE_ROSTER_CODE) for value in values], dtype=np.int64)


def roster_mask(group_codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(group_codes)
    _require(codes.ndim == 1 and np.issubdtype(codes.dtype, np.integer), "Group codes must be a 1-D integer array.")
    _require(bool(np.isin(codes, [OUTSIDE_ROSTER_CODE, 0, 1, 2]).all()), "Group codes must be -1 or a G3 position.")
    return codes != OUTSIDE_ROSTER_CODE


_RAW_ACTION_TO_Y = {"1": 1, "2": 1, "3": 0}


def join_derived_race(
    record_ids: Sequence[str],
    y: Sequence[Any],
    raw_rows: pd.DataFrame,
    contract: RQ4bContract,
    *,
    reconciliation: pd.DataFrame | None = None,
) -> np.ndarray:
    """Join ``derived_race`` by the record_id source-row ordinal.

    ``raw_rows`` is indexed by zero-based source ordinal. Raw ``action_taken``
    must reproduce ``y``; optional reconciliation columns must match exactly.
    """

    ids = validate_record_ids(record_ids, "derived_race join")
    labels = validate_binary_labels(y, "derived_race join", rows=len(ids))
    ordinals = source_ordinals(pd.Series(ids, dtype=object))
    _require({"action_taken", "derived_race"} <= set(raw_rows.columns), "Raw metadata lacks action_taken or derived_race.")
    _require(raw_rows.index.is_unique and bool(pd.Index(ordinals).isin(raw_rows.index).all()), "Raw metadata lacks a requested source row.")
    selected = raw_rows.loc[ordinals].reset_index(drop=True)
    raw_y = selected["action_taken"].astype(str).map(_RAW_ACTION_TO_Y)
    _require(raw_y.notna().all() and np.array_equal(raw_y.to_numpy(dtype=np.int8), labels), "Raw action_taken does not reproduce y_approved.")
    if reconciliation is not None:
        _require(len(reconciliation) == len(ids), "Reconciliation rows differ from record_id rows.")
        for column in reconciliation.columns:
            _require(column in selected.columns, f"Raw metadata lacks reconciliation field {column}.")
            _require(
                selected[column].astype(str).tolist() == reconciliation[column].astype(str).tolist(),
                f"Raw {column} does not reconcile with the partition view.",
            )
    race = selected["derived_race"]
    _require(race.map(lambda value: isinstance(value, str) and bool(value)).all(), "Raw derived_race is missing for a requested row.")
    unknown = sorted(set(race) - contract.allowed_race_groups)
    _require(not unknown, f"Raw derived_race contains groups outside the contract: {unknown}.")
    return race.to_numpy(dtype=object)


# ---------------------------------------------------------------------------
# Predictor inputs (predictor/audit separation)
# ---------------------------------------------------------------------------


def _reject_forbidden_columns(columns: Sequence[str], label: str) -> None:
    leaked = sorted(set(columns) & FORBIDDEN_PREDICTOR_COLUMNS)
    _require(not leaked, f"{label} contains identifier, target, protected or audit columns: {leaked}.")


def validate_raw_predictor_frame(X: pd.DataFrame) -> None:
    _require(isinstance(X, pd.DataFrame), "Raw predictor input must be a DataFrame.")
    _reject_forbidden_columns(list(X.columns), "Raw predictor input")
    _require(
        tuple(X.columns) == RAW_PREDICTOR_COLUMNS and len(X.columns) == RAW_FEATURE_COUNT,
        "Raw predictor input must be the ordered final 15 fields.",
    )


def validate_native_predictor_frame(X: pd.DataFrame) -> None:
    _require(isinstance(X, pd.DataFrame), "Native predictor input must be a DataFrame.")
    _reject_forbidden_columns(list(X.columns), "Native predictor input")
    _require(
        tuple(X.columns) == NATIVE_PREDICTOR_COLUMNS and len(X.columns) == SEMANTIC_FEATURE_COUNT,
        "Native predictor input must be the ordered 19 semantic columns.",
    )
    for column in NUMERIC_VALUE_COLUMNS:
        _require(X[column].dtype == np.dtype("float32"), f"Native {column} must be float32.")
        _require(bool(np.isfinite(X[column].to_numpy()).all()), f"Native {column} must be finite.")
    for column in CATEGORICAL_COLUMNS:
        dtype = X[column].dtype
        _require(isinstance(dtype, pd.CategoricalDtype), f"Native {column} must be category dtype.")
        _require(tuple(dtype.categories) == tuple(CATEGORICAL_SCHEMA[column]), f"Native {column} categories differ from the schema.")
        _require(X[column].notna().all(), f"Native {column} contains values outside the schema.")


@dataclass(frozen=True)
class RQ4bPredictorInput:
    """Native predictor frame with row-aligned metadata kept outside ``X``."""

    X: pd.DataFrame
    y: np.ndarray
    record_ids: tuple[str, ...]
    group_codes: np.ndarray

    @property
    def roster_mask(self) -> np.ndarray:
        return roster_mask(self.group_codes)


def build_predictor_input(
    X_native: pd.DataFrame,
    y: Sequence[Any],
    record_ids: Sequence[str],
    derived_race: Sequence[Any],
    contract: RQ4bContract,
    *,
    pattern: re.Pattern[str] | None = RECORD_ID_PATTERN,
) -> RQ4bPredictorInput:
    validate_native_predictor_frame(X_native)
    ids = validate_record_ids(record_ids, "Predictor input", pattern=pattern)
    rows = len(X_native)
    _require(len(ids) == rows and len(derived_race) == rows, "Predictor features, IDs and derived_race are misaligned.")
    labels = validate_binary_labels(y, "Predictor input", rows=rows)
    codes = encode_roster(derived_race, contract.g3, contract)
    return RQ4bPredictorInput(X_native.reset_index(drop=True), labels, ids, codes)


def make_fit_population_preprocessor(access: StageAccess):
    """Fresh native preprocessing; fitting is permitted only in Train-only stages."""

    _require(access.stage in TRAIN_ONLY_STAGES, f"Stage {access.stage!r} may not fit preprocessing.", RQ4bAccessError)
    from src.dataset_b.preprocessing import make_realmlp_native_preprocessing_pipeline

    return make_realmlp_native_preprocessing_pipeline()


# ---------------------------------------------------------------------------
# Frozen probe probabilities
# ---------------------------------------------------------------------------


def reconstruct_margin(p_approved: Any, p_denied: Any, *, atol: float = PROBABILITY_SUM_ATOL) -> np.ndarray:
    """float64 ``log(p_approved) - log(p_denied)``; never ``1 - p_approved``, no clipping."""

    try:
        approved = np.asarray(p_approved, dtype=np.float64)
        denied = np.asarray(p_denied, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RQ4bInputError("Probability pairs must be numeric.") from exc
    _require(approved.ndim == 1 and approved.shape == denied.shape and approved.size > 0, "Probability pairs must be nonempty aligned 1-D arrays.")
    _require(bool(np.isfinite(approved).all() and np.isfinite(denied).all()), "Probability pairs must be present and finite.")
    _require(bool(((approved > 0.0) & (approved <= 1.0)).all()), "p_approved must lie in (0, 1].")
    _require(bool(((denied > 0.0) & (denied <= 1.0)).all()), "p_denied must lie in (0, 1].")
    _require(
        bool(np.isclose(approved + denied, 1.0, rtol=0.0, atol=atol).all()),
        f"Probability pairs must sum to 1 within atol={atol}.",
    )
    margin = np.log(approved) - np.log(denied)
    _require(bool(np.isfinite(margin).all()), "Reconstructed margin must be finite.")
    return margin


def validate_frozen_probe_predictions(
    frame: pd.DataFrame,
    *,
    expected_rows: int = SPLIT_V1_ROWS["validation"],
    pattern: re.Pattern[str] | None = RECORD_ID_PATTERN,
) -> pd.DataFrame:
    """Validate every stored row; any invalid pair stops the attempt (no row drop)."""

    _require(tuple(frame.columns) == PROBE_PREDICTION_COLUMNS, f"Frozen probe predictions must have columns {list(PROBE_PREDICTION_COLUMNS)}.")
    _require(len(frame) == expected_rows, f"Frozen probe predictions have {len(frame)} rows, expected {expected_rows}.")
    ids = validate_record_ids(frame["record_id"].tolist(), "Frozen probe predictions", pattern=pattern)
    y = validate_binary_labels(frame["y_approved"].tolist(), "Frozen probe predictions")
    approved = pd.to_numeric(frame["p_approved"], errors="coerce").to_numpy(dtype=np.float64)
    denied = pd.to_numeric(frame["p_denied"], errors="coerce").to_numpy(dtype=np.float64)
    margin = reconstruct_margin(approved, denied)
    return pd.DataFrame(
        {"record_id": ids, "y_approved": y, "p_approved": approved, "p_denied": denied, "margin": margin}
    )


def build_real_probe_population(
    predictions: pd.DataFrame,
    derived_race: Sequence[Any],
    contract: RQ4bContract,
) -> pd.DataFrame:
    """Restrict validated frozen predictions to G3 rows, keeping stored order."""

    _require({"record_id", "y_approved", "margin"} <= set(predictions.columns), "Probe predictions are not validated.")
    _require(len(derived_race) == len(predictions), "derived_race is misaligned with probe predictions.")
    codes = encode_roster(derived_race, contract.g3, contract)
    keep = roster_mask(codes)
    population = pd.DataFrame(
        {
            "record_id": predictions["record_id"].to_numpy()[keep],
            "y_approved": predictions["y_approved"].to_numpy(dtype=np.int8)[keep],
            "derived_race": np.asarray(derived_race, dtype=object)[keep],
            "group_code": codes[keep],
            "margin": predictions["margin"].to_numpy(dtype=np.float64)[keep],
        }
    )
    _require(set(population["group_code"]) == {0, 1, 2}, "Probe population must contain every G3 group.")
    return population


# ---------------------------------------------------------------------------
# Loaders (each checks stage access before opening a file)
# ---------------------------------------------------------------------------


def load_frozen_probe_predictions(access: StageAccess) -> pd.DataFrame:
    path = access.require_resource("frozen_probe_validation_predictions")
    frame = pd.read_csv(path, dtype={"record_id": str}, float_precision="round_trip", keep_default_na=True)
    return validate_frozen_probe_predictions(frame)


STAGE_PARTITION = {
    "cv_epoch_selection": "train",
    "full_train_refit": "train",
    "validation_calibration_selection": "validation",
    "test_evaluation": "test",
}


def stage_population_ids(
    access: StageAccess,
    *,
    probe_predictions: pd.DataFrame | None = None,
    manifest_loader: Callable[[Path], Any] | None = None,
) -> frozenset[str]:
    """Record IDs a stage may request metadata for.

    Split stages get their own split_v1 partition from the manifest; the probe
    gets the IDs of its validated frozen Validation predictions only.
    """

    if access.stage == "capability_probe":
        _require(probe_predictions is not None and "margin" in probe_predictions.columns, "Probe population requires validated frozen predictions.", RQ4bAccessError)
        return frozenset(validate_record_ids(probe_predictions["record_id"].tolist(), "Probe population"))
    _require(access.stage in STAGE_PARTITION, f"Stage {access.stage!r} has no project population.", RQ4bAccessError)
    path = access.require_resource("split_manifest")
    if manifest_loader is None:
        from src.dataset_b.splits import DatasetBSplitter

        manifest_loader = DatasetBSplitter.load_manifest
    manifest = manifest_loader(path)
    return frozenset(getattr(manifest, f"{STAGE_PARTITION[access.stage]}_record_ids"))


def read_derived_race_rows(
    access: StageAccess,
    record_ids: Sequence[str],
    *,
    permitted_ids: frozenset[str],
    reconciliation_fields: Sequence[str] = (),
) -> pd.DataFrame:
    """Read only the requested source rows of the raw metadata, indexed by ordinal.

    ``permitted_ids`` comes from ``stage_population_ids``; IDs outside the
    stage's own population are refused before the source is opened.
    """

    path = access.require_resource("derived_race_source")
    ids = validate_record_ids(record_ids, "derived_race request")
    outside = set(ids) - set(permitted_ids)
    _require(not outside, f"Stage {access.stage!r} requested {len(outside)} rows outside its population.", RQ4bAccessError)
    ordinals = source_ordinals(pd.Series(ids, dtype=object))
    wanted = {int(value) + 1 for value in ordinals}  # file line numbers, header is line 0
    columns = ["action_taken", "derived_race", *reconciliation_fields]
    frame = pd.read_csv(
        path,
        usecols=columns,
        dtype=str,
        keep_default_na=False,
        skiprows=lambda line: line != 0 and line not in wanted,
    )
    _require(len(frame) == len(wanted), "Raw metadata lacks a requested source row.")
    frame.index = pd.Index(sorted(int(value) for value in ordinals))
    return frame


def load_partition(access: StageAccess, partition: str):
    """Quality-gated split_v1 partition read only from the contract's resources.

    The shared split loader is constructed with the authorized manifest, split
    directory and feature contract, so an alternate project root can never fall
    back to repository-default data. Access is checked before construction.
    """

    split_path = access.require_partition(partition)
    manifest_path = access.require_resource("split_manifest")
    feature_contract_path = access.require_resource("feature_contract")
    from src.dataset_b.preprocessing import DatasetBInputQualityGate, DatasetBSplitLoader

    loader = DatasetBSplitLoader(
        manifest_path=manifest_path,
        split_dir=split_path.parent,
        feature_contract_path=feature_contract_path,
    )
    gated = DatasetBInputQualityGate().apply(loader.load(partition))
    expected = access.contract.split_rows[partition]
    _require(
        len(gated.X) == expected and len(gated.record_ids) == expected,
        f"RQ4b {partition} row count {len(gated.X)} differs from split_v1 ({expected}).",
    )
    _require(tuple(gated.X.columns) == tuple(loader.feature_columns), "Quality-gated raw schema is not the final 15 fields.")
    validate_raw_predictor_frame(gated.X)
    return gated
