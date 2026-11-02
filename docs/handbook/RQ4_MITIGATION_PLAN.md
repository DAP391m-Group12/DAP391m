# Dataset B RQ4 Equalized-Odds mitigation plan

## 1. Status, scope, and terminology

This document locks governance, methodology, machine-readable configuration,
and accepted closure evidence for Equalized-Odds mitigation. RQ4a
post-processing is implemented, executed, reported, and closed after C8
`PASS`. RQ4b-3 adversarial RealMLP is executed, package-validated, reported and
closed (2026-10-24; `STATUS.md` owns the evidence). The optional RQ4b-6
sensitivity remains deferred. The repository names are:

- **RQ4a**: Equalized-Odds post-processing; manuscript equivalent **RQ3a**.
- **RQ4b**: Equalized-Odds adversarial RealMLP; manuscript equivalent **RQ3b**.

Existing RQ3 remains the completed descriptive fairness audit. RQ4 must use a
new `rq4` artifact namespace and must never write mitigation outputs under the
accepted `artifacts/dataset_b/rq3/` namespace.

The four predeclared configurations are:

| Configuration | Role | Current readiness |
|---|---|---|
| RQ4a-3 | Primary post-processing analysis on G3 | `COMPLETE`; accepted run `rq4a3_postproc_20261017_01`; C6 review `PASS`; both models `completed / stable / pass`; reported in Notebook 11 and accepted by C8 `PASS` |
| RQ4a-6 | Required post-processing sensitivity on G6 | `COMPLETE`; accepted run `rq4a6_postproc_20261017_01`; C7 review `PASS`; both models `completed / unstable / fail` as the predeclared sensitivity outcome; reported in Notebook 11 and accepted by C8 `PASS` |
| RQ4b-3 | Primary adversarial RealMLP analysis on G3 | `COMPLETE` and closed (2026-10-24). The nine development units are complete with Gate B `PASS`. Gate C run `rq4b3_gatec_20261024_01` stores `selected_lambda = 0.0`. Test run `rq4b3_test_20261024_01` executed the nine user-selected arms once (receipt `PASS`), and its package validation is `PASS`. Notebook 12 reports the neutral outputs. Research interpretation is user-owned and not recorded |
| RQ4b-6 | Optional adversarial RealMLP sensitivity on G6 | Deferred, optional and not committed. RQ4b-3 is closed; RQ4b-6 may start only as a new stream under a separate user authorization |

The separate RQ1 paired-bootstrap prerequisite (T1) is `COMPLETE` and locked.
It compares all 15 pairs of the six frozen final models over the ten-metric
RQ1 contract using the shared locked Test bootstrap rows; its evidence is the
`RQ1 paired-bootstrap cross-model comparison` row of `STATUS.md` and
`notebooks/dataset_b/final/research_questions/07_model_total_review.ipynb`. RQ4 does not reopen it.

## 2. Evidence boundary and baseline anchors

RQ4a uses stored, ordered `record_id`, `y_approved`, and `p_approved` values
from the frozen XGBoost and RealMLP Validation/Test packages. It performs no
base-model retraining or reinference. The verified anchors are:

| Model | Frozen run | Validation threshold | Validation BA | Test BA | Test ROC-AUC | Test Brier |
|---|---|---:|---:|---:|---:|---:|
| XGBoost | `xgb_final_15f_20261003_01` | 0.525 | 0.7868670517 | 0.7829818366 | 0.8641458255 | 0.1313877649 |
| RealMLP | `realmlp_final_15f_20261004_01` | 0.726 | 0.7865608646 | 0.7827860581 | 0.8632032833 | 0.1065088230 |

These values are read from frozen threshold and metric artifacts. They are
baseline anchors, not computed RQ4 results and not evidence that mitigation
improves any metric. RQ4a ROC-AUC and Brier are inherited invariant diagnostics
from frozen scores; post-processing must never claim improvements in them.

### RQ4a input preflight (PASS, read-only, 2026-10-16)

A read-only preflight over `data/` and `artifacts/` passed before contract
closure. It wrote no project file and performed no fit, inference, or
bootstrap. It established:

- XGBoost and RealMLP Validation packages each contain 43,995 rows and Test
  packages each contain 43,996 rows, with unique `record_id`, finite
  `p_approved` in `[0, 1]`, and `y_approved` in `{0, 1}`;
- both models share identical ordered `record_id` and `y_approved` per
  partition, and that order matches `data/dataset_b/split_v1/{validation,test}.csv`;
- stored thresholds are `0.525` (XGBoost) and `0.726` (RealMLP);
- `artifacts/dataset_b/rq1_test_bootstrap_indices.npz` holds `indices` with
  shape `(1000, 43996)`, `int32`, range `0..43995`;
- the `record_id`-suffix join to `data/raw/state_NY_2024.csv` is complete for
  both partitions with no null `derived_race`.

A follow-up read-only check reconciled `lei`, `action_taken`, and
`applicant_sex` between the controlled `split_v1` Validation/Test views and
the joined raw rows with zero mismatches, found no null `applicant_sex`, and
found only codes mapped by the RQ3 sex mapping (`1`, `2`, `3`, `4`, `6`). The
C2 review reported its own direct runs of raw/view reconciliation and
`applicant_sex` completeness as `PASS`, and an independent verifier re-ran the
reconciliation with exact RQ3-style string equality and zero mismatches. The preflight evidence is the console
output of session-local read-only scripts and is not stored as a repository
artifact; implementation must re-run these checks as runtime validation.

Approved/Denied counts on the complete partitions are:

| `derived_race` | Validation | Test |
|---|---:|---:|
| White | 21,623 / 6,246 | 21,421 / 6,370 |
| Black or African American | 2,065 / 1,342 | 2,093 / 1,363 |
| Asian | 3,002 / 989 | 3,014 / 962 |
| American Indian or Alaska Native | 92 / 89 | 112 / 77 |
| 2 or more minority races | 49 / 57 | 53 / 59 |
| Joint | 593 / 161 | 618 / 147 |

Rows outside G6 are 7,687 (Validation) and 7,707 (Test); rows outside G3 are
8,728 and 8,773. Every G6 group meets the denominator gates on the complete
partitions. Stratified Validation resampling preserves these counts. The
locked Test indices are not race-stratified; a read-only count over all 1,000
locked rows found that no replicate fails a G6 TPR/FPR gate (minimum
Approved/Denied: `2 or more minority races` 31/39, `American Indian or Alaska
Native` 84/49), so EOD6 and EOD3 are expected to be valid in every Test
replicate for both models, which share the same ordered labels. The minimum
`2 or more minority races` replicate size is 74 rows, below the 100-row
selection-rate gate, so that group's selection rate may be null in some
replicates. The strict invalidity rule still applies if a gate fails.

RQ4b is a new RealMLP retraining experiment. It preserves the Dataset B cohort,
`split_v1`, final 15 raw predictors/19 semantic columns, target orientation,
and predictor exclusion of race and sex. It does not reuse the frozen RealMLP
as a fitted RQ4b model.

## 3. Protected-group contract

The protected attribute is `derived_race`, the HMDA application-level
applicant/co-applicant aggregation joined by stable `record_id`. It is not an
applicant-only race field. In particular, `Joint` is an application-level
derived category and must not be described as an applicant race.

G3, used by both primary configurations, is the fixed ordered roster:

1. `White`
2. `Black or African American`
3. `Asian`

G6, taken from the accepted RQ3 global-race scope rather than RQ2, is the fixed
ordered roster:

1. `White`
2. `Black or African American`
3. `Asian`
4. `American Indian or Alaska Native`
5. `2 or more minority races`
6. `Joint`

The coverage-only groups outside G6 are:

- `Native Hawaiian or Other Pacific Islander`
- `Free Form Text Only`
- `Race Not Available`

The configured roster is fixed across bootstrap replicates. A runner must
never silently drop, merge, rename, or dynamically shrink G6. If any required
G6 TPR/FPR component is invalid under the locked denominator rules, that
replicate's EOD6 is invalid and must be saved with a reason code. The accepted
RQ3 Test counts show especially small cells for `2 or more minority races`
(112 rows: 53 Approved, 59 Denied), so RQ4a-6 and any later RQ4b-6 analysis must
surface instability rather than conceal it.

RQ4 retains the accepted RQ3 denominator gates: at least 100 group rows for
selection-rate reporting and at least 30 actual Approved and 30 actual Denied
observations for TPR/FPR/EOD components. An undefined component remains null;
it is never replaced by zero or recomputed over a smaller group roster.

## 4. Development governance and leakage boundary

Train, Validation, and Test membership remain unchanged. Validation is not
subdivided. This avoids making the fixed G6 denominator gate mathematically
unattainable for `2 or more minority races`, which has only 49 Approved and 57
Denied observations in the complete Validation partition.

- RQ4a fits `ThresholdOptimizer` on the complete Validation partition under a
  fully predeclared configuration; it performs no model or policy search.
- RQ4b calibrates each candidate's one global threshold and selects lambda on
  the complete Validation partition. RQ4b-3 selection uses EOD3 and the matched
  BA budget. EOD6 is an audit metric and never selects lambda.
- Test is opened only after the applicable configuration, group scope,
  threshold/postprocessor, lambda, and seeds are locked. Test never chooses a
  seed, lambda, threshold, weighting policy, or mitigation scope.

Using the same Validation observations for RQ4b threshold calibration and
lambda selection can produce optimistic development estimates. This is an
explicit exploratory-study limitation, not evidence of independent model
selection and calibration. The limitation must be reported, and no Test result
may be used to revise the selected lambda or threshold.

## 5. RQ4a post-processing protocol

RQ4a applies Fairlearn `ThresholdOptimizer` to each frozen XGBoost and RealMLP
score stream with:

- `constraints: equalized_odds`
- `objective: balanced_accuracy_score`
- `grid_size: 1000`
- `flip: false`
- `prefit: true`
- `predict_method: predict_proba`

The primary randomized-decision seed is 42. Required sensitivity seeds are the
fixed sequence 42 through 71 inclusive. Every seed is reported; no seed is
selected as best.

RQ4a-3 fits/applies mitigation for G3 rows and uses each frozen model's baseline
hard prediction outside G3. RQ4a-6 fits/applies mitigation for G6 rows and uses
the frozen baseline hard prediction outside G6. Outside-scope rows are not
dropped and do not receive a new group-specific rule.

### RQ4a cached-probability architecture

The base model is represented only by a small `CachedProbabilityEstimator`
adapter over the stored frozen `p_approved` values. It:

- exposes `predict_proba`, returning ordered columns `[1 - p_approved,
  p_approved]` looked up by `record_id`;
- loads no model bundle, checkpoint, preprocessor, or feature matrix, and
  performs no training or inference;
- satisfies the `prefit: true` fitted-state contract without fitting;
- requires a unique `record_id` lookup (a duplicate lookup key fails) and
  fails on any unknown or missing query `record_id` rather than imputing or
  reordering silently;
- accepts repeated query `record_id` values from with-replacement bootstrap
  resampling and returns one row per query occurrence, preserving multiplicity
  and order; synthetic row keys such as `record_id#position` are not used
  (C4 clarification).

Baseline lineage is validated per model and partition: the frozen hard class
is `p_approved >= frozen_threshold`; a stored `predicted_approved` must match
it, a stored `predicted_denied` must equal `1 - predicted_approved`, an
artifact without a stored hard class derives it from the rule, and the
validated frame always carries `predicted_approved` (C4 clarification).

`ThresholdOptimizer` therefore receives frozen scores only; its own fitted
state is the post-processing policy.

### RQ4a Validation/Test audit join

The accepted RQ3 audit join is generalized from Test to both Validation and
Test: the one-based source-row number is parsed from the `record_id` suffix and
joined one-to-one to the immutable `data/raw/state_NY_2024.csv` to obtain
`derived_race`. Before `derived_race` is accepted, the reconciliation fields
`lei`, `action_taken`, and `applicant_sex` must match exactly between the
controlled split row and the raw-source row. As in RQ3, `applicant_sex` is
taken from the controlled split view and mapped to `applicant_sex_group` with
the RQ3 sex mapping; an unmapped code fails the join. The controlled split
views are named explicitly in the config (`data/dataset_b/split_v1/validation.csv`
and `data/dataset_b/split_v1/test.csv`) and are not derived from the RQ3
`test_view` location (C4 clarification). The join must be complete
with no null `derived_race` and no change in row order. Protected attributes are used only
as `sensitive_features` and audit columns; they never enter the base score.

### RQ4a fixed-roster EOD

`compute_fixed_roster_eod` computes EOD over the exact configured G3 or G6
roster using the accepted RQ3 `compute_group_metrics` and denominator gates
(`src/dataset_b/fairness/metrics.py`).
EOD is the maximum of the roster-wide TPR range and FPR range. If any required
TPR or FPR in the roster is undefined, that EOD is invalid (null with a reason
code) for that evaluation; it is never recomputed over a smaller roster. RQ3
`compute_global_race_ranges` is not reused directly because it may omit
undefined groups. Validity is metric-specific: an invalid EOD3 or EOD6 does
not invalidate Balanced Accuracy or the other EOD in the same replicate.
Because the denominator gates depend only on true labels and groups, baseline
and mitigated EOD share the same validity on the same rows.

### RQ4a uncertainty

RQ4a uses 1,000 replicates. In each replicate it:

1. refits the postprocessor on a predeclared conditional bootstrap of the
   complete Validation partition, stratified by (`derived_race`,
   `y_approved`);
2. reuses the existing locked paired Test indices for Test row lookup;
3. draws randomized decisions from an independent decision stream;
4. performs no base-model fit or inference; and
5. saves both valid and invalid replicates with reason codes.

The root seed is `10000 + replicate_id`. Independent deterministic child
streams are required for Validation resampling and randomized decisions.
Existing locked Test indices determine Test row selection and therefore do not
consume a newly generated Test-resampling stream. The locked indices must have
dtype exactly `int32`. Within a Test replicate, every occurrence of a row
receives its own independent randomized draw, taken in locked-index order from
the decision child stream; the same seed and the same order reproduce the
decisions bit for bit (C4 clarification). Child-stream identity must
be persisted.

Each replicate is paired: baseline and mitigated decisions are evaluated on
the same locked Test rows. Confidence intervals are two-sided 95% percentile
intervals over the valid values of each quantity, using quantile method
`linear`. Invalid replicates are retained in `bootstrap_replicates.csv` with
reason codes and are excluded only from the quantity they invalidate:

- the delta-EOD interval uses target-EOD-valid replicates;
- the BA-cost interval uses BA-valid replicates, including replicates whose
  EOD is invalid;
- audit-EOD intervals use their own valid replicates and never gate the
  primary criterion.

A replicate is BA-valid when both actual classes are present among its rows,
so Balanced Accuracy is defined for baseline and mitigated decisions. Every
criterion component requires at least 950 valid values out of 1,000.

### RQ4a primary criterion

Signs are locked as:

- `delta_eod = mitigated_eod - baseline_eod`;
- `ba_cost = baseline_ba - mitigated_ba`.

The target EOD is EOD3 for RQ4a-3 (EOD6 reported as audit) and EOD6 for
RQ4a-6 (EOD3 reported as audit). The criterion is evaluated separately for
XGBoost and RealMLP; models are not pooled and no model is selected. Point
estimates use the complete Test partition, the policy fitted on the complete
Validation partition, and randomized-decision seed 42. A configuration-model
pair has `primary_criterion: pass` only when all four hold:

1. point `delta_eod < 0` (target EOD decreases);
2. upper 95% CI of `delta_eod < 0`;
3. point `ba_cost <= 0.01`;
4. upper 95% CI of `ba_cost <= 0.01`.

It is `not_evaluable` when the point target EOD is invalid
(`required_group_metric_invalid`) or when the delta-EOD
or BA-cost component has fewer than 950 valid values
(`insufficient_valid_bootstrap_replicates`); these keep their existing reason
codes. Otherwise it is `fail`, recording
`target_eod_reduction_not_demonstrated` when point `delta_eod >= 0` or the
upper 95% CI of `delta_eod >= 0`, and `ba_budget_exceeded` when condition 3 or
4 fails; when both fail, both reason codes are stored. `unstable_diagnostics`
never substitutes for a primary-criterion failure reason. Budgets 0.005 and 0.02 are
reporting sensitivities only and never change `primary_criterion`.

### RQ4a randomization sensitivity and stability status

Seeds 42 through 71 re-draw randomized decisions on the complete Test
partition with the complete-Validation policy. They quantify randomization
sensitivity only, are reported in `randomization_sensitivity.csv`, and are
never pooled into the bootstrap intervals or used to choose a seed.

`stability_status` is assessed separately for each model and configuration
over all 30 seeds 42-71 (decision locked in C2 review):

- `not_assessed`: execution is incomplete, a seed is missing, or a required
  metric (target EOD or BA) is invalid for any seed;
- `stable`: all 30 seeds are valid and, for every seed, both indicators
  `delta_target_eod < 0` and `ba_cost <= 0.01` equal their seed-42 values;
- `unstable`: all 30 seeds are valid but at least one seed changes either
  indicator relative to seed 42; `unstable_diagnostics` is then added to the
  reason codes.

Stability measures consistency under decision randomization, not whether the
outcome is favourable: a result may be `stable` with `primary_criterion: fail`.
The minimum, maximum, and range of the sensitivity target EOD, delta target
EOD, mitigated BA, and BA cost are always reported so that `stable` is not read
as numerical invariance.

### RQ4a run identity, outputs, and reporting boundary

RQ4a-3 and RQ4a-6 each receive one immutable run ID with the patterns
`rq4a3_postproc_<YYYYMMDD>_<NN>` and `rq4a6_postproc_<YYYYMMDD>_<NN>`.
`<YYYYMMDD>` is the production execution date and `<NN>` starts at `01`;
concrete IDs are assigned at production execution, not in this contract. An
existing completed-run directory or failure receipt is never overwritten; a
rerun increments `<NN>`. A published failure receipt consumes its run ID. Each
completed run contains both models under:

```text
artifacts/dataset_b/rq4/rq4a/<run_id>/
  validation/fitted_policies/
  validation/decisions.csv
  test_evaluation/decisions.csv
  test_evaluation/bootstrap_replicates.csv
  test_evaluation/status.json
  reporting/model_summary.json
  reporting/experiment_metrics.csv
  reporting/group_metrics.csv
  reporting/decision_changes.csv
  reporting/randomization_sensitivity.csv
  reporting/bootstrap_summary.csv
  reporting/runtime.csv
  reporting/artifact_index.json
```

The completed-run layout above is published only after both model executions
and all artifact validation finish. It is never used for partial results. After
the runner contract, config, prerequisites, run ID, output collisions,
production inputs, and runner inputs have all passed their read-only gates, the
execution attempt begins. An exception after that boundary removes the
temporary completed-run directory and atomically writes one immutable receipt:

```text
artifacts/dataset_b/rq4/rq4a/_failures/<run_id>.json
```

With an `output_root` override, the receipt is written to
`<output_root>/_failures/<run_id>.json`. The receipt records the run and
configuration identity, attempt-level execution status and reason codes, failed
stage and model when known, UTC start/failure times, producer runtime
provenance, exception type, and a single-line error summary limited to 1,000
characters. The exception is re-raised after the receipt is published. Receipt
publication is atomic, never overwrites an existing receipt, prevents later use
of the same run ID, and leaves no partial receipt if receipt writing itself
fails.

Failures before the attempt boundary—including contract/config, prerequisite,
run-ID, collision, production-input-preflight, or runner-input failures—publish
neither a completed-run directory nor a failure receipt. Generic exceptions
after the boundary produce `execution_status: failed` with
`runtime_failure`. Only the explicit `RQ4AInfeasibleError` domain exception may
produce `execution_status: infeasible` with `postprocessing_infeasible`; the
runner must never infer infeasibility from an arbitrary Fairlearn or Python
exception.

Required file names are locked. The column lists under `outputs` in
`config/dataset_b/rq4a_postprocessing.yaml` were agent-authored, amended by
the C2 review, and are locked after C2 `PASS`. Lineage requirements from C2:

- `validation/decisions.csv` and `test_evaluation/decisions.csv` each carry
  `p_approved`, `derived_race`, `applicant_sex_group`, `in_target_roster`,
  baseline and seed-42 mitigated decisions, and `decision_changed`, so
  Validation/Test EOD, `fairness_generalization_gap`, inherited ROC-AUC/Brier,
  and the sex-EOD safety audit can be recomputed without rerunning Fairlearn;
- `group_metrics.csv` carries an `attribute` column (`derived_race` or
  `applicant_sex`) alongside `group`, and per-metric extremum roles
  `tpr_extremum_role`, `fpr_extremum_role`, and
  `selection_rate_extremum_role` with values `min`, `max`, `none`, or
  `not_evaluable`;
- `status.json` holds independent per-model `model_statuses` entries, each with
  `execution_status`, `stability_status`, `primary_criterion`, and
  `reason_codes`; an optional run-level status never replaces them;
- `model_summary.json` reports `fairness_generalization_gap_eod3` and
  `fairness_generalization_gap_eod6` as separate fields;
- `experiment_metrics.csv` carries `sex_eod_valid`, `sex_eod_tpr_range`, and
  `sex_eod_fpr_range`, so an undefined sex EOD (`sex_eod_valid = false`) is
  distinguishable from a missing output;
- `randomization_sensitivity.csv` carries `invalid_reason_codes`, so a
  `not_assessed` stability status caused by an invalid metric is traceable to
  its reason codes; a missing seed is detected by the absence of its row among
  the 30 expected seeds.

Python code produces and validates artifacts; the reporting module writes the
normalized `reporting/` files atomically. Notebook 11 follows the results
presentation pattern of Notebook 09: it reads `reporting/` only and must not
fit, predict, bootstrap, import Fairlearn, or read model, prediction, or
raw-data artifacts.

### RQ4a accepted production closure

RQ4a-3 production run `rq4a3_postproc_20261017_01` passed C6. XGBoost and
RealMLP are each `completed / stable / pass`. RQ4a-6 production run
`rq4a6_postproc_20261017_01` passed C7. XGBoost and RealMLP are each
`completed / unstable / fail` with the predeclared reason codes
`target_eod_reduction_not_demonstrated`, `ba_budget_exceeded`, and
`unstable_diagnostics`; these are study outcomes, not execution failures.

Each run published 16 validated files, wrote no failure receipt, retained
1,000/1,000 valid replicates for target EOD, audit EOD, and BA cost, and used
the locked two-sided 95% percentile intervals with `linear` quantiles. Notebook
11 reads the two `reporting/` packages only and passed C8. RQ4a and its
manuscript equivalent RQ3a are closed; the accepted numerical results and
interpretation limits are registered in `STATUS.md` and Notebook 11.

### RQ4a runtime

RQ4a runs on CPU in the dedicated environment `.venv-rq4a-cpu`, specified by
`requirements-rq4a-cpu.txt`. It matches the core version subset of the project
package baseline needed for RQ4a: Python 3.13.14, NumPy 2.5.2, pandas 3.0.5,
SciPy 1.18.1, scikit-learn 1.9.1, PyYAML 6.0.3, and joblib 1.6.0. The Python,
NumPy, pandas, and scikit-learn versions are recorded in
`artifacts/dataset_b/realmlp/final/realmlp_final_15f_20261004_01/runtime/manifest.json`;
the SciPy, PyYAML, and joblib versions follow the project `requirements.txt`. PyTorch and PyTabKit are not
installed there; the frozen RealMLP environment is not mutated.

The CPU environment/Fairlearn compatibility smoke (T3) is `COMPLETE` and the C3
independent review is `PASS`. Fairlearn is pinned at `0.15.0`. The accepted
smoke run is `rq4a_cpu_env_smoke_20261016_02`
(`artifacts/dataset_b/rq4/environment/rq4a_cpu_env_smoke_20261016_02/`), which
supersedes `rq4a_cpu_env_smoke_20261016_01`. It recorded the actual
interpreter and imported versions on synthetic toy data only, and locked policy
persistence as `joblib.dump(compress=3)` with the cached-probability adapter in
a stable importable module plus the `policy_interpolation.json` audit export.
CUDA is not a gate for RQ4a; CUDA/PyTabKit verification is deferred to the RQ4b
technical spike.

## 6. RQ4b adversarial RealMLP protocol

### Protocol ownership and decision boundary (2026-10-18)

The RQ4b-specific machine-readable contract is
`config/dataset_b/rq4b_adversarial.yaml`, protocol
`rq4b_adversarial_realmlp_v3_parallel_evaluation_training`. Its predecessor was
`rq4b_adversarial_realmlp_v2_cv_epoch_selection`. This section supersedes the
earlier RQ4b suggestions of fixed epoch 50, an inner-holdout selector, and
patience-based stopping. It does not change the frozen RealMLP baseline,
`realmlp_v1.yaml`, PlainMLP, or accepted RQ4a evidence. C-spike acceptance is
registered in STATUS with its limited scope; no production execution is implied.

Codex now owns implementation, documentation/config synchronization, tests,
review and reporting under chat-approved task scopes. A separate self-review
pass is not independent review. Historically Claude implemented the scoped
spike and T1-T5 work, and Codex independently reviewed those implementations;
retain that evidence with its original authorship. Under the user's current
chat decision, Codex implements, tests, fixes bugs, performs a separate
self-review/validation pass and reports technical acceptance. The user executes
all real fits: capability probe, CV and full-Train refit. Codex hands over
stage-specific commands and checks saved artifacts after each user run; Codex
does not launch or retry real fits. Claude has no current implementation,
review or verifier dependency. Real execution still requires its own scoped
task; documentation approval is not a launch. Spike PASS does
not itself authorize production execution. The user has locked the production
architecture, seed policy, refit horizon, probe population/split/budget and
scientific CI endpoint rules below. The final probe optimizer, stopping,
stored-score extraction, synthetic fixture and percentile interpolation are
also user-locked below. Consult STATUS for implementation acceptance and its
synthetic evidence limits; acceptance is not real-data runtime verification
or permission to train.

### RQ4b-3 parallel evaluation and training amendment (2026-10-21)

The user-approved amendment replaces both the original 18-development-unit
protocol and the temporary 12-fit priority policy. The original protocol
remains identifiable as
`rq4b_adversarial_realmlp_v2_cv_epoch_selection`; the amended machine-readable
protocol is `rq4b_adversarial_realmlp_v3_parallel_evaluation_training`. The
amendment does not change the predictor/adversary architectures, optimization,
five-fold epoch-selection and full-Train refit method, G3/G6 taxonomy, metric
definitions, denominator rules, frozen thresholds, Validation/Test separation,
or bootstrap definitions.

Two independent tracks are authorized by contract. Track A reads the five
frozen completed seed-42 development units and writes only authorized evaluation
outputs for EOD3, paired same-seed delta EOD3 where defined, Balanced Accuracy
cost, available diagnostics, fairness--utility evidence and training stability.
It must not refit, rerun inference, regenerate or overwrite their checkpoints,
predictions, epochs, receipts or manifests. Track B comprises only four new
development units: `(lambda=0, seed=43)`, `(lambda=0.01, seed=43)`,
`(lambda=0, seed=44)` and `(lambda=0.01, seed=44)`. Track B does not depend on
Track A completion or outcome. The units may run concurrently only when paths,
write ownership and resources are safe; otherwise queue the four units
sequentially. An unfavorable preliminary EOD3 result does not cancel Track B.

Gate A blocks Track B until the amended scope/budget, implementation and
configuration are valid; run ID, seeds and lambdas are correct; artifact paths
do not conflict; completed artifacts are protected; and no critical correctness
or lineage issue remains. EOD3 evaluation is not part of Gate A. Gate B requires
each new unit to complete five CV folds, one full-Train refit under the existing
epoch-selection policy, development-validation, integrity and prediction-count
checks, and artifact/receipt validation. An incomplete unit is never completed.
Gate C blocks final arm/lambda selection and Test until all retained development
units have evaluation evidence, EOD3, paired same-seed delta EOD3 and BA cost
are available, EOD6 is audited, seed variability and training stability/integrity
are assessed, statistical limitations are explicit, and a three-seed selection
policy receives separate methodological approval.

### Seed-44 lineage-preserving continuation (2026-10-22)

Seed 44 must not resume or modify the completed v3 child run. Its execution
identity is the v4 continuation `rq4b3_adv_20261022_01`, protocol
`rq4b_adversarial_realmlp_v4_seed44_continuation`, decision revision
`rq4b3_seed44_continuation_20261022`, and role
`track_b_seed44_continuation`. This run may contain only `(lambda=0, seed=44)`
and `(lambda=0.01, seed=44)`. The method, model architecture, data, split,
epoch-selection policy and fit budget per unit are unchanged from v3.

The continuation is a fresh run with no artifact migration and no fitted-state
reuse. Its contract pins the SHA-256 of both source manifests, all published
development/development-validation receipts from the five seed-42 and two
seed-43 units, and the retained seed-43 failure receipt. Production create/open must verify those source digests before
accepting the v4 run, then lock the current v4 config and code hashes in the
new manifest. A source mismatch stops before create/open. A later v4 code or
config change blocks resume; preserve the existing attempted run and use a new
identity only under a separately approved revision. Never rewrite a source
manifest or receipt to make validation pass.

Seed 44 completed sequentially under its later authorized execution task:
`(lambda=0, seed=44)` completed and its published development and
development-validation artifacts were verified before `(lambda=0.01,
seed=44)` started. Each unit has intact upstream receipts, five CV folds plus
one refit, and 43,995 Validation predictions. Preserve both units without
rerun or automatic retry. Gate C lineage validation must explicitly map the
seven source units and two continuation units in a new evaluation-only run
before cross-run aggregation. At that point selection, final-fit, Test and
package validation remained closed. Gate C aggregation, Test evaluation and
package validation later ran under separate authorizations (section 9). The
Test reused the development full-Train refits, so no final-fit stage was run.

For every RQ4b candidate, predictor loss uses all Train rows. Adversarial loss
is masked to the configured G3 or G6 roster. Race and sex never enter predictor
features. All rows, including rows outside the adversary mask, receive a
prediction from the new predictor and one global calibrated threshold.

The adversary is multiclass and conditioned on true `y_approved` during
training only. Its preferred score is the Approved-minus-Denied logit margin,
concatenated with true `y_approved`; positive-class probability was the
pre-spike fallback. The accepted spike exposed the margin
and the user selected it for production; do not silently substitute the fallback.
Production input dimensions are two: margin and true label during training only.

The production G3 adversary is fixed at `2 -> 32 ReLU -> 3 logits`, no dropout,
no class weighting and no architecture search. This is an approved fixed
choice, not evidence that the architecture is optimal. Set adversary seed to
the model seed with matching initialization across lambdas within the same
seed/fold, and isolate adversary RNG from predictor state. Assess capability
separately using a synthetic correctness gate and an independent probe compared
with Y-only under the capability protocol below. Neither project Test nor
architecture search may override the locked probe choices.
If capability is not established, report `capability_inconclusive`: it does not
automatically mean execution failure or prove absence of group information.
Do not automatically increase capacity, search architectures or drop seeds.

### Independent capability probe (user-locked decisions)

Use saved Validation predictions from frozen RQ1 RealMLP run
`realmlp_final_15f_20261004_01`, restricted to the three G3 groups. Do not retrain,
overwrite or rerun inference on the frozen model as part of this probe. Read
`validation/validation_predictions.csv` from that frozen run and reconstruct
the score in float64 as `log(p_approved) - log(p_denied)`. Require both saved
probabilities finite, in `(0,1]`, and their sum equal to 1 within `rtol=0`,
`atol=1e-6`; require the reconstructed margin finite. Stop on invalid input,
without clipping, dropping rows or replacing the source with new inference.
This is a reconstructed stored-probability margin, not a claim of bitwise
identity to original logits. Do not use `log(p_approved/(1-p_approved))`:
inspection of all 43,995 saved Validation rows found 1,383 Approved
probabilities equal to 1 while both saved class probabilities remained positive;
the paired log ratio was finite. These counts describe the entire Validation
package, not the G3 subset, and do not establish probe capability.

Within this probe population create a reproducible 60% training / 20% validation /
20% holdout split, stratified by `(derived_race, y_approved)`, seed 42. This is a
probe-local split of the existing project Validation records; it does not change
`split_v1`. The complete project Validation partition still calibrates predictor
thresholds and selects lambda. The no-subdivision rule in Section 4 applies to
those predictor stages, not to this independent probe. The probe is diagnostic;
do not use its holdout to change architecture or redesign the protocol.

Fit the probe only on probe-training rows, with AdamW, learning rate 0.001,
maximum 100 epochs and patience 15. Probe-validation selects the checkpoint;
probe-holdout reports CE gain and its interval. Lock weight decay 0.01,
betas `(0.9,0.999)`, epsilon `1e-8`, and AMSGrad/maximize/capturable/
differentiable/foreach/fused all false. Use batch size 256, `cuda:0`, probe seed
42 with RNG isolated from the predictor. Monitor unweighted group-prediction
CE on probe-validation in eval/no-gradient mode, minimizing it with min-delta
0. Only strictly lower CE improves; exact ties retain the earliest checkpoint.
Stop after 15 consecutive completed epochs without improvement or at epoch
100, and restore the best checkpoint before holdout evaluation. Nonfinite
loss or gradient stops the attempt with a failure receipt; do not skip the
batch or restart automatically. These probe settings do not change predictor training:
CV still runs all 256 epochs and the training adversary still uses Adam.

The synthetic positive control has 6,000 rows: 2,000 per G3 group, with 1,000
per `(group,Y)` cell. Generate the margin as the group's center plus independent
zero-mean Gaussian noise with SD 0.25; centers in fixed G3 order are `[-2,0,2]`.
Use NumPy `default_rng(42)`, stable synthetic IDs and a 60/20/20 stratified
`(group,Y)` split with seed 42. Probe inputs are only synthetic margin and Y;
group is the target, never a predictor input. Use the locked architecture,
fitting/stopping, comparator and bootstrap recipe. This fixture deliberately
contains score information about groups. Do not retune it after seeing results.
This paragraph originally specified the synthetic gate before execution. The
gate was subsequently executed and accepted as C-spike `PASS` in its limited
synthetic two-epoch scope; STATUS owns that execution evidence.

The Y-only comparator estimates `P(A|Y)` on probe-training only, using Laplace
smoothing alpha=1 over the three groups. For each Y value and group a, the
estimate is `(n[A=a,Y]+1)/(n[Y]+3)`. Do not fit comparator counts on validation
or holdout rows. Define `CE gain = CE_Y-only - CE_probe`, in nats. Both losses
are evaluated on the same probe-holdout rows. The synthetic correctness gate
requires point CE gain >= 0.05 nats AND lower endpoint of its 95% CI > 0.
The real probe reports gain and CI descriptively and is not a production
PASS/FAIL gate. Failure to establish real capability is
`capability_inconclusive`, not proof that group information is absent.

For probe gain use 1,000 paired bootstrap replicates of held-out probe rows,
stratified by `(race,Y)`, and percentile 95% CI; require 1,000/1,000 valid
replicates. Keep the fitted probe and Y-only comparator fixed in each replicate.
Do not reuse project Test bootstrap indices for probe rows. Use probe bootstrap
seed 42 and `np.quantile(values, [0.025,0.975], method="linear")`; do not round
before computing quantiles or comparing endpoints. Keep the probe
fits separate from the 132 predictor-fit budget; no architecture search,
unapproved retries or favorable-result selection is implied.

The predictor-side adversarial objective is explicitly
`L_predictor_total = L_prediction - lambda * L_adversary`. A gradient-reversal
layer is permitted only when it is mathematically equivalent to this signed
objective. The minus sign is part of the locked contract: predictor training
must maximize adversary loss while minimizing prediction loss; an additive
`+ lambda * L_adversary` predictor objective is not equivalent and is
prohibited.

Use a separate Adam adversary optimizer (`lr=0.001`, `weight_decay=0`), one
adversary update followed by one predictor update per nonempty-roster batch.
First minimize adversary CE using detached predictor scores; then freeze
adversary parameters and minimize the signed predictor objective. Preserve
the predictor forward graph when feasible so an extra dropout draw does not
silently alter training. The predictor's existing optimizer, regularization,
callbacks and schedules must remain traceable; do not add adversary parameters
to its optimizer implicitly. With an empty roster mask, update only the
predictor and record the skipped adversary update. Keep row/group metadata
aligned through shuffle and remove it before predictor forward. For lambda=0,
bypass the adversary and isolate its RNG from predictor initialization,
dropout and loader shuffle.

Here, all Train rows means the eligible predictor-loss population is not
filtered by race. Preserve the native loader's `drop_last=True` and batch size
256 in the spike, disclose the incomplete final batch omission, and do not
claim every eligible row is necessarily used in every epoch.

### Train-only CV epoch selection and fresh full-Train refit

For each lambda/seed combination, initialize five independent fold models with
fresh fold-local preprocessing and backend state. Use the contract's
`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` membership, shared
across lambdas and model seeds. Each fold trains for all 256 epochs, with no
per-fold or shared patience stopping. After each epoch evaluate prediction CE
on the complete corresponding fold-validation set in eval/no-gradient mode.
Fold-validation does not update fitted parameters or preprocessing state.

For Approved=1 and p=P(Approved), CE is the unweighted row mean of
`-(y*log(p) + (1-y)*log(1-p))`. Use the numerically stable equivalent from the
two-class logit margin; do not silently introduce probability clipping or
class weighting. Aggregate five fold CE values with an unweighted arithmetic
mean at each common epoch. Select the minimum unrounded mean over epochs
1..256; exact arithmetic-value ties select the **earliest** epoch. This replaces
the previously discussed backend latest-tie rule for this selector.
All 1,280 fold/epoch CE values must be finite and present. Missing/nonfinite
values invalidate the selection; do not shorten the horizon or shrink the
fold roster. Save the complete curves and selected-epoch evidence.

CV fairness metrics are monitoring only. Neither adversarial loss, EOD,
project Validation nor Test selects an epoch. CE selection protects prediction
quality but does not optimize the fairness/utility balance. These CV curves
are development evidence used for selection, not an independent post-selection
performance estimate. This is post-training epoch selection, not early stopping.

Initialize fresh preprocessing, predictor, adversary, optimizers and backend
state on the complete Train partition and refit exactly to the selected epoch;
do not reuse CV fitted state or create a stopping holdout. Record
`epoch_cap_hit=true` when the selected epoch equals 256. This is a boundary
diagnostic, not proof of nonconvergence; it does not authorize budget extension.

The user has locked the refit schedule horizon at **256 epochs** for learning
rate, weight decay and dropout, while execution stops after exactly the selected
number of epochs. Do not compress the schedule to the refit budget. The spike
provides short feasibility evidence for separating these quantities; production
implementation and validators must still prove the selected mechanism obeys
the policy. This is not evidence of a completed 256-epoch production fit.

The amended retained lambda roster is `[0, 0.01, 0.03, 0.1, 0.3]`; `lambda=1`
is removed from the execution queue. The five retained seed-42 development
units cover that roster. `lambda=0.03`, `0.1` and `0.3` remain screening
evidence at seed 42 only. Control (`lambda=0`) and `lambda=0.01` are replicated
over seeds 42, 43 and 44. This is an explicit nine-unit sparse design, not the
Cartesian product of five lambdas and three seeds.

For each development unit authorized by this sparse design:

1. complete Train-only five-fold epoch selection and fresh full-Train refit above;
2. calibrate one global threshold on the complete Validation partition using the frozen
   RQ1 grid `0.001..0.999` by `0.001` and tie rule (closest to 0.5, then lower);
3. evaluate matched BA and EOD3 on the same complete Validation partition;
4. compute matched delta BA against the same-seed lambda=0 control.

The earlier five-seed selection rule is superseded for this amended experiment,
but no replacement aggregate rule had been approved at that time. The BA budget remains
`-0.01`, EOD3 remains the primary fairness objective, EOD6 remains an audit and
cannot select lambda, and Test cannot select a lambda, seed, threshold or
architecture. Gate C was therefore blocked until a selection rule suitable for
the three replicated seeds was explicitly justified and approved. That rule was
later approved as the evaluation-only `candidate_selection` policy in
`rq4b_adversarial.yaml`. It compares only the replicated lambdas 0 and 0.01 on
Validation. A lambda is eligible when its mean matched delta BA is at least
-0.01. The policy then selects the minimum mean EOD3, with exact ties going to
the smaller lambda. EOD6 cannot select. Gate C ran under this policy in
`rq4b3_gatec_20261024_01` (see STATUS). No two-of-three rule was substituted.
RQ4b-6 remains outside the committed scope and requires separate authorization.

The amended budget contains at most nine development units: five retained
seed-42 units and the four additional Track B units. Each unit retains five CV
fold fits and one full-Train refit, giving an upper bound of 45 CV fold fits and
nine full-Train refits (54 fits total), excluding the capability probe, retries
and failed attempts. Existing completed units are reused without retraining.
No lambda, seed, architecture search or experimental method may be added.

The primary adversary uses unweighted multiclass cross-entropy. If RQ4b-6 is
separately authorized later, its optional weighted sensitivity may use
inverse-frequency class weights calculated from Train and fixed before Test.
Test information may never decide weighting.

### RealMLP internal-extension and naming gate

The public sklearn wrapper exposes no adversarial-training hook. The existing
history adapter proves that the pinned PyTabKit internals can be extended by a
project subclass/interface, but it does not prove an adversarial recipe. Before
RQ4b implementation is named or executed as adversarial RealMLP, a technical
spike must pass all of these gates:

1. pin the exact PyTabKit source revision in addition to version 1.7.3;
2. identify and replace/subclass the correct `training_step` path;
3. locate the correct predictor logit/score tensor and verify its dimensions;
4. run a finite one-batch forward/backward test for predictor and adversary;
5. serialize and reload the extended model without changing predictions; and
6. demonstrate lambda=0 equivalence to the preserved RealMLP recipe under a
   predeclared numerical policy.

If the preserved RealMLP recipe cannot be maintained, the experiment must be
marked infeasible or renamed; it must not be called adversarial RealMLP.

RQ4b requires `cuda:0` with no silent CPU fallback, `batch_size: 256`, and
`predict_batch_size: 1024`. CUDA availability and a finite-operation preflight
must pass before execution. The compatibility baseline is Python 3.13.14,
PyTorch 2.14.0+cu130, PyTabKit 1.7.3, NumPy 2.5.2, pandas 3.0.5, and
scikit-learn 1.9.1.

### Disposable technical spike and independent acceptance

The spike write scope consists only of new
`src/dataset_b/mitigation/rq4b_spike.py`,
`tests/test_dataset_b_rq4b_spike.py`, and unique attempt directories beneath
`scratch/rq4b_spike/`. No existing source, documentation/configuration, frozen
model, RQ4a package, notebook, or production artifact may be modified by the
spike implementer. Use synthetic native-schema fixtures; no project Validation
or Test data, production grid, or 256-epoch production fit. Each disposable
smoke fit is limited to at most two epochs with configured batch size 256.

The spike used one 32-unit ReLU hidden layer, three G3 output logits,
no dropout and unweighted CE. Its use alone did not lock production; the later
explicit user decision above independently locks the same architecture.
Pin the actual upstream source revision and record installed internal-file
hashes; package version alone or hashes alone do not satisfy revision lineage.
If provenance cannot be established, report the gate unresolved, never invent
a revision. Package modifications or dependency installation are outside scope.

Compare IDs, labels, masks, batch order and configuration exactly. Compare
lambda=0 predictor parameters and probabilities against the preserved recipe
under identical seed/input/step/schedule/device controls using `rtol=1e-5`,
`atol=1e-6`, reporting maximum absolute differences. Baseline here means a fresh
short smoke of the preserved recipe, not reusing frozen fitted state or epoch
selection policy. Serialization/reload requires exactly equal eval-mode
probabilities on identical inputs in the same environment. Repeat the same-seed
short smoke at least twice; do not generalize this to full-run determinism or
relax tolerances after observing results.

Test signed gradients, parameter updates, metadata alignment, non-G3 predictor
loss participation, empty masks, CUDA-unavailable rejection, nonfinite values,
reload failure, and synthetic epoch-selection ties/missing curves/cap flags.
The adversarial-loss gradient with respect to scores outside G3 must be zero; shared
predictor parameter updates can still change their predictions. Probe schedule
horizon feasibility without running long CV. Record the exact device of models
and tensors; CUDA availability alone does not prove cuda:0 training.

Save provenance, commands, seed/configuration, tensor shapes, diagnostics,
gate-by-gate evidence and naming conclusion in the unique disposable root.
Stop on a mandatory gate failure and write an immutable attempt failure receipt
there, distinguishing environment failure, implementation defect and an
unsupported/infeasible design. Do not emit fabricated production metrics or
label missing evidence PASS. Codex independently reviews the diff/scope and
reproduces focused gradient, equivalence and reload checks before C-spike PASS.
Schedule-policy support is reported separately from user approval. C-spike
`PASS` is limited to the synthetic technical scope and is registered in STATUS;
The method decisions for RQ4b-3 are locked. Implementation acceptance is
registered in STATUS; real-data runtime validation and explicit user launch
remain separate requirements.
RQ4b-6 remains separately deferred.

## 7. Metrics, utility budget, and generalization

Primary fairness metrics are EOD3 and EOD6 over their exact fixed rosters,
where each EOD is the maximum of the roster-wide TPR range and FPR range.
Primary utility is full-partition Balanced Accuracy, including rows outside the
mitigated/adversary roster. The primary BA budget is 0.01; sensitivity budgets
are 0.005 and 0.02.

`fairness_generalization_gap` is Test EOD minus Validation EOD, reported
separately for EOD3 and EOD6. It must not be called a
"calibration-versus-Test fairness gap". For RQ4a and RQ4b, development means
the predeclared evaluation on the complete Validation partition. For RQ4b this
is not an independently held-out estimate because the same partition also
calibrates thresholds and selects lambda.

Denied metrics treat Denied as the positive class: precision is null without
predicted Denied, recall is null without actual Denied, and
`F1 = 2 * true_denied / (actual_denied + predicted_denied)`, which is 0 when the
denominator is positive and `true_denied = 0` and null only when the denominator
is 0. An empty group's decision `change_rate` is null (C4 clarification).

Every applicable result also records Denied F1/precision/recall; per-group TPR,
FPR, and selection rate; extrema groups; group coverage and denominators; sex
EOD as a safety audit; and decision-change counts/rates overall and by group.
RQ4a additionally records inherited ROC-AUC/Brier without interpreting them as
post-processing gains.

The additional list above is the RQ4a reporting set. For RQ4b-3, the
applicable outputs are defined completely by the user-selected Test output
contract of 2026-10-24 (section 8 and the locked `required_output_files` in
`rq4b_adversarial.yaml`). Denied F1/precision/recall, race selection rates and
extrema groups are not RQ4b-3 outputs. The RQ4b-3 sex EOD safety audit is
provided separately by the evaluation-only supplementary run
`rq4b3_sex_supplementary_20261024_02` (see STATUS).

## 8. Uncertainty and reporting

RQ4b uses within-seed paired Test bootstrap comparisons against the matched
same-seed lambda=0 control, using the locked Test indices. It separately reports
across-seed mean, standard deviation, minimum, maximum and range. These are
distinct uncertainty views and must not be pooled. No seed is selected from Test.

### RQ4b-3 user-selected Test output contract (2026-10-24)

The authorized Test identity is `rq4b3_test_20261024_01`, published only under
`artifacts/dataset_b/rq4/rq4b/test_evaluation/`. Contract authorization does
not constitute implementation or execution: this documentation task reads no
Test row, performs no fit, CV, refit or inference, and publishes no Test run.
Production execution requires a separate explicit launch after implementation
and validation pass.

The exact user-selected Test roster is all nine saved development model
artifacts: lambdas 0, 0.01, 0.03, 0.1 and 0.3 at seed 42, plus lambdas 0 and
0.01 at seeds 43 and 44. Lambda 0 is identified as the Gate C selected control;
lambda 0.01 is the replicated comparator; and the three screening lambdas are
explicitly included at their available seed 42 because the user prioritizes
complete numerical reporting. No additional lambda, seed or arm may be inferred
from Gate C eligibility, convenience or a code-side policy.

The Test manifest must pin the Gate C manifest
`5d072d7f...218931b` and receipt `14f602bb...448f13`, the three source
manifests, and the model/Validation receipt pair for each of the nine arms. The
full exact hashes are machine-readable in `rq4b_adversarial.yaml`. It must also
pin `artifacts/dataset_b/rq1_test_bootstrap_indices.npz` at SHA-256
`2ee377b6fbae8a3a6453a7d8f2f4a0a3f1ce61374f1258ef882589b84230fee1`.
The indices are reused exactly and never regenerated.

For every one of the nine arms, export full-Test Balanced Accuracy, EOD3, audit EOD6,
and each fixed-roster group's TPR, FPR and counts (rows, actual Approved/Denied,
predicted Approved/Denied). Compare each nonzero-lambda arm against lambda 0 at
the same seed: four comparisons at seed 42 and one each at seeds 43 and 44.
For all six comparisons, export matched delta BA, delta EOD3 and delta EOD6
plus overall and per-group decision-change counts and rates, including both
change directions. Run exactly 1,000 paired bootstrap replicates per matched
comparison on the locked rows (6,000 replicate rows total) and export every
replicate, validity flags/reasons, valid/invalid counts, and the lower/upper 95%
percentile CI endpoints for all three matched deltas. Use quantiles 0.025/0.975
with method `linear`, without replacement replicates or rounding before the
quantile calculation.

Across the three seeds, export mean, sample SD (`ddof=1`), minimum, maximum and
range for the replicated lambdas 0 and 0.01 and for the lambda-0.01-minus-0
matched deltas. Do not create across-seed summaries for lambdas 0.03, 0.1 or
0.3 because each has only seed 42; report their point metrics, matched deltas,
bootstrap intervals and gaps directly. Do not pool bootstrap rows across seeds.
For every arm and seed, export Validation and Test values and `Test - Validation`
generalization gaps for BA, EOD3 and EOD6. The
exact required file names and columns are locked in the config as
`test_evaluation.required_output_files`; additional research-result fields are
not added without user approval.

Execution status, metric values and research interpretation are three separate
layers. Code may record only `completed`, `failed` or `infeasible` execution
status with operational reason codes and the neutral numeric outputs above. It
must not generate a research `pass/fail`, mitigation-effectiveness result,
strong-support classification, aggregate seed rule, recommendation or narrative
conclusion. The user owns all comparison choices and research interpretation.
In particular, code must not convert CI endpoints, sign patterns, Gate C
selection or successful execution into a scientific conclusion.

Use `stability_status: not_assessed` for RQ4b and report the requested seed
variability without inventing a stability threshold. Do not select a best seed,
exclude an unfavorable seed, extend the budget after Test or add any arm beyond
the nine explicitly selected here.

Every completed RQ4b-3 Test run therefore reports two machine-generated layers:

- `execution_status`: `completed`, `failed`, or `infeasible`;
- the locked metric tables and lineage evidence.

Research interpretation is intentionally absent from machine-generated output.

For RQ4a, the completed-run `status.json` emits `execution_status: completed`.
The `failed` and `infeasible` values are attempt-level outcomes emitted only in
the immutable failure receipt described above; they do not authorize a partial
completed-run layout or synthetic result rows.

At minimum, reason codes include `none`, `prerequisite_rq1_paired_bootstrap_missing`,
`technical_spike_pending`, `pytabkit_revision_unpinned`,
`adversarial_hook_unavailable`, `score_tensor_unresolved`,
`nonfinite_preflight`, `serialization_reload_failed`,
`lambda_zero_equivalence_failed`, `required_group_metric_invalid`,
`insufficient_valid_bootstrap_replicates`, `target_eod_reduction_not_demonstrated`,
`ba_budget_exceeded`, `unstable_diagnostics`, `runtime_failure`, and
`postprocessing_infeasible`. Multiple reason codes may apply.

## 9. Sequencing and stop gates

1. Approve this plan and both machine-readable configs: the RQ4a contract delta
   closure (T2) is `COMPLETE` and the C2 independent review is `PASS`.
2. Complete the separate RQ1 paired-bootstrap prerequisite: `COMPLETE` (T1).
3. Resolve the RQ4a open decisions (`stability_status` definition and the
   EOD-failure reason code): `COMPLETE`, locked in the C2 review.
4. Run the dedicated CPU RQ4 environment/Fairlearn smoke: `COMPLETE` (T3,
   run `rq4a_cpu_env_smoke_20261016_02`, C3 review `PASS`).
5. Implement and validate RQ4a without retraining/reinference: pure core
   (inputs, metrics, post-processing) `COMPLETE` (T4, C4 review `PASS`), and
   RQ4a runner `COMPLETE` (T5, C5 review `PASS`).
6. Execute RQ4a-3, then the required RQ4a-6 sensitivity, each under its own
   immutable run ID: `COMPLETE` (T6/C6 `PASS`; T7/C7 `PASS`).
7. Complete reporting-only Notebook 11 and close RQ4a/RQ3a: `COMPLETE`
   (T8/C8 `PASS`).
8. RQ4b internal-extension technical spike, including CUDA/PyTabKit: accepted
   after R1 as C-spike `PASS` in synthetic two-epoch scope; see STATUS.
9. The RQ4b-3 amended nine-unit development execution is `COMPLETE` and Gate B
   is `PASS`. Preserve the five seed-42 units, two seed-43 units and two
   seed-44 units under their immutable run identities without refit or
   reinference. STATUS owns the accepted execution evidence.
10. Gate C aggregation is `COMPLETE`. The nine-arm Test output contract was
    authorized and locked. Its cross-run implementation and synthetic
    validation are `COMPLETE`. The explicitly launched production Test run
    `rq4b3_test_20261024_01` is `COMPLETE` (one execution, receipt `PASS`). Its
    no-inference package validation is `PASS`, and reporting-only Notebook 12
    is `COMPLETE`. RQ4b-3 is closed; see STATUS for the evidence.
11. Leave RQ4b-6 deferred. Open it only as a new stream through a separate future
   authorization, with a separate diagnostic acceptance policy approved first.

RQ4a is closed after the accepted T2--T8 chain and C2--C8 reviews. Its two
immutable production runs and reporting-only Notebook 11 provide the accepted
runtime, mitigation-effect, stability, criterion, and uncertainty evidence.
The RQ4b technical spike has been accepted in its limited scope. The amended
nine-unit development roster is complete, Gate B is `PASS`, and Gate C
aggregation is complete; STATUS owns the direct evidence. The user-selected
nine-arm Test run, its package validation and Notebook 12 are complete, so
RQ4b-3 is closed. Its research interpretation is user-owned and is not recorded
by this plan. RQ4b-6 remains deferred.

## 10. Interpretation boundary

### RQ4b-3 applicant-sex supplementary evaluation (locked 2026-10-24)

This supplementary package evaluates the already closed nine-arm RQ4b-3 Test
output only. It is a separate, immutable namespace and does not alter the
production Test run, its receipts, or RQ3/RQ4a artifacts. The protected
attribute is `applicant_sex`; it is not `derived_sex` and no race columns are
combined or redefined. The locked RQ3 mapping is `1=Male`, `2=Female`,
`3=Information not provided`, `4=Not applicable`, and `6=Applicant selected
both male and female`.

The fixed sex-EOD roster is exactly Male/Female. All five mapped labels are
reported for coverage and group metrics. The package reads saved Test and
Validation predictions plus the existing 1,000 locked Test bootstrap indices;
it must not fit, refit, run CV, infer, select a threshold, or generate
bootstrap indices. Per-arm output, same-seed nonzero-lambda deltas, decision
changes, paired-bootstrap validity payloads, replicated-arm summaries, and
Validation--Test gaps are descriptive outputs. Execution status, metric values,
and research interpretation remain separate; the package records no research
conclusion or recommendation.

All RQ4 outputs are exploratory model-behavior evidence. They cannot establish
legal compliance, discrimination, causal effects, creditworthiness, or default
risk. A smaller observed fairness gap does not prove absence of unfairness, and
a utility/fairness trade-off does not determine policy acceptability.

