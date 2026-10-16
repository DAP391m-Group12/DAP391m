# Dataset B RQ4 Equalized-Odds mitigation plan

## 1. Status, scope, and terminology

This document locks governance, methodology, and machine-readable configuration
for a future Equalized-Odds mitigation study. It does not implement or execute
mitigation. The repository names are:

- **RQ4a**: Equalized-Odds post-processing; manuscript equivalent **RQ3a**.
- **RQ4b**: Equalized-Odds adversarial RealMLP; manuscript equivalent **RQ3b**.

Existing RQ3 remains the completed descriptive fairness audit. RQ4 must use a
new `rq4` artifact namespace and must never write mitigation outputs under the
accepted `artifacts/dataset_b/rq3/` namespace.

The four predeclared configurations are:

| Configuration | Role | Current readiness |
|---|---|---|
| RQ4a-3 | Primary post-processing analysis on G3 | Ready for later implementation after documentation/config approval |
| RQ4a-6 | Required post-processing sensitivity on G6 | Ready for later implementation after documentation/config approval; small-cell instability must remain visible |
| RQ4b-3 | Primary adversarial RealMLP analysis on G3 | Blocked pending the internal-extension technical spike |
| RQ4b-6 | Optional adversarial RealMLP sensitivity on G6 | Deferred and not committed; requires a separate later authorization after RQ4b-3 gates pass |

A separate RQ1 paired-bootstrap task must be completed before any RQ4
execution. It will compare predictive performance, thresholded utility, and
calibration pairwise across frozen models. That prerequisite is outside this
write scope and is not implemented here.

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
consume a newly generated Test-resampling stream. Child-stream identity must
be persisted.

## 6. RQ4b adversarial RealMLP protocol

For every RQ4b candidate, predictor loss uses all Train rows. Adversarial loss
is masked to the configured G3 or G6 roster. Race and sex never enter predictor
features. All rows, including rows outside the adversary mask, receive a
prediction from the new predictor and one global calibrated threshold.

The adversary is multiclass and conditioned on true `y_approved` during
training only. Its input is the predictor logit if the technical spike confirms
a suitable logit tensor; otherwise it is the positive-class probability. The
final tensor choice, dimensions, and extraction point are a required
pre-execution gate, not an implementation assumption.

The predictor-side adversarial objective is explicitly
`L_predictor_total = L_prediction - lambda * L_adversary`. A gradient-reversal
layer is permitted only when it is mathematically equivalent to this signed
objective. The minus sign is part of the locked contract: predictor training
must maximize adversary loss while minimizing prediction loss; an additive
`+ lambda * L_adversary` predictor objective is not equivalent and is
prohibited.

The lambda grid is `[0, 0.01, 0.03, 0.1, 0.3, 1.0]`. Development seeds are
`[42, 43, 44]`; final seeds are `[42, 43, 44, 45, 46]`.

For each development seed and lambda candidate:

1. train on Train;
2. calibrate one global threshold on the complete Validation partition using the frozen
   RQ1 grid `0.001..0.999` by `0.001` and tie rule (closest to 0.5, then lower);
3. evaluate matched BA and EOD3 on the same complete Validation partition;
4. compute matched delta BA against the same-seed lambda=0 control.

A candidate is eligible only when its mean matched delta BA is at least -0.01.
RQ4b-3 selects the eligible candidate with minimum mean EOD3. An exact tie
chooses the smaller lambda. Lambda=0 is a valid winner. EOD6 is reported as an
audit metric and cannot select lambda. RQ4b-6 is not part of the committed
execution scope; a later separately authorized protocol may open it after the
RQ4b-3 technical and stability gates pass.

Train lambda=0 controls for all five final seeds. For seeds 42-44, reuse the
identical selected-lambda artifacts from development rather than retraining
them. The committed RQ4b-3 plan requires at most approximately 22 unique full
fits. A later RQ4b-6 authorization would add approximately 17 primary fits and
may reuse the lambda=0 controls. Its optional weighted sensitivity would add
five more fits.

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

Every applicable result also records Denied F1/precision/recall; per-group TPR,
FPR, and selection rate; extrema groups; group coverage and denominators; sex
EOD as a safety audit; and decision-change counts/rates overall and by group.
RQ4a additionally records inherited ROC-AUC/Brier without interpreting them as
post-processing gains.

## 8. Uncertainty and reporting

RQ4b uses within-seed paired Test bootstrap comparisons against the matched
same-seed lambda=0 control, using the locked Test indices. It separately reports
across-seed mean, standard deviation, and range. These are distinct uncertainty
views and must not be pooled. No seed is selected from Test.

Every configuration must report three independent fields:

- `execution_status`: `completed`, `failed`, or `infeasible`;
- `stability_status`: `stable`, `unstable`, or `not_assessed`;
- `primary_criterion`: `pass`, `fail`, or `not_evaluable`.

At minimum, reason codes include `none`, `prerequisite_rq1_paired_bootstrap_missing`,
`technical_spike_pending`, `pytabkit_revision_unpinned`,
`adversarial_hook_unavailable`, `score_tensor_unresolved`,
`nonfinite_preflight`, `serialization_reload_failed`,
`lambda_zero_equivalence_failed`, `required_group_metric_invalid`,
`insufficient_valid_bootstrap_replicates`, `ba_budget_exceeded`,
`unstable_diagnostics`, and `runtime_failure`. Multiple reason codes may apply.

## 9. Sequencing and stop gates

1. Approve this plan and both machine-readable configs.
2. Complete the separate RQ1 paired-bootstrap prerequisite.
3. Implement and validate RQ4a without retraining/reinference.
4. Execute RQ4a-3, then the required RQ4a-6 sensitivity.
5. Run the RQ4b internal-extension technical spike.
6. Open RQ4b-3 only if all technical gates pass.
7. Leave RQ4b-6 deferred. Open it only through a separate future authorization
   after RQ4b-3 gates pass and diagnostics are stable.

The present task stops after step 1 governance/config authoring. It makes no
runtime, mitigation-effect, stability, or criterion-pass claim.

## 10. Interpretation boundary

All RQ4 outputs are exploratory model-behavior evidence. They cannot establish
legal compliance, discrimination, causal effects, creditworthiness, or default
risk. A smaller observed fairness gap does not prove absence of unfairness, and
a utility/fairness trade-off does not determine policy acceptability.

