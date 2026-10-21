# Dataset B split and evaluation contract

## 1. Purpose and ownership

This contract defines the Dataset B partition and evaluation boundaries used by
RQ1 and the gates that must pass before RQ2 or RQ3 can begin. It owns:

- the immutable record-level split;
- Train-only cross-validation and configuration selection;
- the full-Train refit boundary;
- Validation-only threshold selection;
- Test-only final evaluation;
- RQ1 metrics, bootstrap, and cross-model comparison boundaries;
- run-level lineage from preprocessing through Test evaluation;
- the conditions for opening RQ2 and RQ3.

It does not own feature decisions, preprocessing semantics, model parameters, or
model execution status. Feature governance is defined in
[Feature governance](feature_governance.md). Classical preprocessing is defined
in [Classical preprocessing](preprocessing_classical.md).

## 2. Immutable split contract

Dataset B uses `split_v1`, created once at `record_id` level and stratified by
`y_approved` with random state `42`. There is no LEI or other grouping
constraint.

| Partition | Fraction | Expected rows | Permitted role |
| --- | ---: | ---: | --- |
| Train | 70% | 205,310 | Train-only CV, configuration selection, then fresh full-Train fit |
| Validation | 15% | 43,995 | Threshold selection after the full-Train fit |
| Test | 15% | 43,996 | Final RQ1 evaluation after preprocessing, model, and threshold are locked |
| **Total** | **100%** | **293,301** | **Every eligible `record_id` appears exactly once** |

The split policy and ordered membership are defined by:

- `config/dataset_b/split_v1.yaml` for the split policy;
- `artifacts/dataset_b/split_v1.json` for ordered partition membership;
- `data/dataset_b/split_v1/train.csv`;
- `data/dataset_b/split_v1/validation.csv`;
- `data/dataset_b/split_v1/test.csv`.

Consumers must preserve the manifest's listed order. A changed membership,
order, seed, stratification rule, or grouping rule requires a new split version;
it must not overwrite or be presented as `split_v1`.

The final 15-feature benchmark retains this exact `split_v1` membership, order,
and row count. Every benchmark model family must be retrained against the same
Train, Validation, and Test records. The final input-quality gate must therefore
PASS without removing a `split_v1` record. If it rejects any current member,
the retraining run stops for data-quality resolution; a runner must not silently
drop the record, change the partition counts, or regenerate the split while
continuing to identify it as `split_v1`.

## 3. Train-only cross-validation

RQ4b has a model-specific epoch-selection exception to the generic selection
table below; see Section 3.3 and the
[RQ4 mitigation plan](../RQ4_MITIGATION_PLAN.md#6-rq4b-adversarial-realmlp-protocol).
The exception does not change the frozen benchmark protocols.

Cross-validation is restricted to the Train partition:

| Property | Contract value |
| --- | --- |
| Splitter | `StratifiedKFold` |
| Folds | `5` |
| Shuffle | `true` |
| Random state | `42` |
| Allowed population | `train_only` |
| Primary selection metric | Mean ROC-AUC |
| Stability evidence | Standard deviation of fold ROC-AUC |

For each fold, every learned component must be initialized fresh and fitted
only on fold-train rows. This includes medians, category state, scalers,
model parameters, and any backend-owned transformation. Fold-validation may be
transformed and predicted but may not update fitted state.

The Phase 1 classical protocol gives each baseline exactly one fixed
configuration and sets `hyperparameter_search: false`. Its five folds measure
performance and stability; they do not constitute a search. Any future tuning
must use a new approved protocol and remain entirely inside Train-only CV.

Validation and Test identifiers must not appear in a CV fit or fold-validation
set. Out-of-fold predictions must retain the ordered Train `record_id` and
cover each Train row exactly once.

### 3.1 Fixed class-imbalance handling

Random Forest uses sklearn `class_weight="balanced"`. The class weights are
derived independently from the labels in each CV fold-train population. For
the fresh full-Train fit, they are derived again from full Train. Validation
and Test labels are excluded from all weighting calculations.

The XGBoost weighting rule is executable policy:
`scale_pos_weight = n_denied / n_approved`, calculated from the population
being fitted. It must be calculated separately from each CV fold-train
population and again from full Train for the fresh full-Train fit. It is not a
fixed entry in `models.xgb.parameters`, and Validation and Test labels must not
contribute to the calculation.

Both rules are fixed fit-time policy, not hyperparameter tuning. The protocol
continues to require `sample_weight: false`, `resampling: false`, and
`hyperparameter_search: false`.

PlainMLP uses fit-population `pos_weight = n_denied / n_approved`, with
Approved as the positive class. The temporary selector for each outer fold
derives the value from that fold's inner-fit rows; the fresh final outer
estimator recalculates it from the complete outer fold-train population. The
full-Train temporary selector derives it from the full-Train inner-fit rows;
the fresh final full-Train estimator recalculates it from all 205,310 Train
rows. PlainMLP must not combine `pos_weight` with sample weights or resampling,
and Validation/Test labels must not contribute to any weight.

### 3.2 PlainMLP inner-stopping and refit boundary

PlainMLP uses a stratified 90% inner-fit / 10% inner-stop split with random
state 42 inside each permitted fit population. The temporary selector monitors
inner-stop BCE loss in `min` mode for at most 100 epochs, with patience 10 and
`min_delta = 0.00001`, and selects the best epoch. The selector, optimizer, and
their fitted preprocessing state are then discarded. A fresh final estimator
and fresh fit-local preprocessor fit the complete outer fold-train population
for exactly the selected epoch count before predicting outer fold-validation
once. Outer fold-validation, project Validation, and Test are forbidden from
the inner-stop set.

The same procedure applies to full Train: a temporary full-Train selector
chooses the epoch using only its Train-derived inner split, then a fresh final
estimator and fresh preprocessor fit all 205,310 Train rows for that epoch
count. The complete sequence is five temporary outer-fold selectors, five
final outer-fold estimators, one temporary full-Train selector, and one final
full-Train estimator: 12 fresh model initializations, of which only the six
final estimators are model evidence. Temporary selector state is never
published or reused by a final estimator.

### 3.3 RQ4b post-training CV epoch selection

For each configured lambda/model seed, use five Train-only stratified folds
with shuffle and split random state 42, shared across lambdas and model seeds.
Train every fold for exactly 256 epochs without patience stopping. Fit fresh
preprocessing and model/backend/optimizer state only on each fold-train.
At each epoch evaluate unweighted prediction CE (Approved=1) on that fold's
complete validation rows in eval/no-gradient mode, using a stable logit-based
equivalent of binary probability CE. Average the five fold CE values equally.
Choose the minimum unrounded mean CE; exact ties choose the earliest epoch.
Require all 5 x 256 values finite and present; missing folds/epochs must not
be ignored. Fairness is monitoring only; adversarial loss, EOD and project
Validation/Test cannot select epochs. CV estimates used in this selection are
development evidence and are not independent performance estimates afterward.

The selected epoch is specific to lambda/seed. Fresh full-Train refit uses it
without a stopping holdout or transferred fitted CV state. Selected epoch 256
sets `epoch_cap_hit`, without automatically increasing the budget. The user has
locked refit schedule horizon at 256 for LR, weight decay and dropout, while
executing only the selected number of epochs; do not rescale schedules to that
budget. Production support still requires implementation/validation. This exception is not
the historical fixed-50 RealMLP pilot policy or PlainMLP inner stopping.

## 4. Full-Train refit boundary

After one configuration has been selected for a model family:

1. create a new preprocessing pipeline and a new model instance;
2. fit all learned preprocessing state on the complete Train partition;
3. fit the model on that same complete Train partition;
4. freeze the fitted preprocessing, fitted model, configuration, feature
   contract, split version, runtime settings, and `run_id` before reading
   Validation for threshold selection.

Fold-fitted preprocessing or a fold-fitted model is a CV artifact and cannot be
reused as the full-Train fitted state. The full-Train fit must not receive
project Validation or Test as an internal channel. PlainMLP may use only the
Train-derived inner-stop procedure in Section 3.2; another internal channel
requires a later approved model-specific protocol compatible with this
partition boundary.

## 5. Validation threshold boundary

For RQ4b only, repeat the fresh refit boundary for each predeclared candidate.
The complete project Validation partition calibrates each candidate's global
threshold and selects lambda from the locked grid under the matched BA/EOD3
rule in the RQ4 plan. This is an explicit exception to the generic prohibition
on Validation configuration selection below, not permission for open-ended
tuning. Do not select predictor epochs, seeds or adversary architecture using
project Validation/Test. Report shared calibration/selection optimism; Test is
opened only after lambda, thresholds, seeds and production policies are locked.

RQ4b Test assessment uses adversarial-minus-same-seed-control deltas: the
per-seed point criterion requires delta EOD3 < 0 and full-Test delta BA >= -0.01.
Report 1,000 locked paired-bootstrap replicates per seed, percentile 95% CIs
and separate across-five-seed mean/SD (`ddof=1`)/min/max/range; never pool seeds'
bootstrap rows. Preserve the existing Y-stratified RQ1 Test indices, not new
race-by-Y indices. Every CI quantity requires 1,000/1,000 valid replicates;
otherwise it is not evaluable, without replacement sampling or an automatic
scientific-failure conclusion. At least 4/5 seeds must each satisfy upper CI
of delta EOD3 < 0 AND lower CI of full-Test delta BA > -0.01. Do not substitute
point criteria for those CI conditions. Use quantiles 0.025/0.975 with
`method="linear"`, without rounding before CI calculation or endpoint comparison. Stability
is `not_assessed`; scientific support and capability are distinct from execution.

The separate capability probe may split the frozen RQ1 RealMLP Validation G3
population 60/20/20, stratified by `(race,Y)`, seed 42. Fit probe and Y-only
comparator only on probe-training; select checkpoint on probe-validation and
report gain/CI on probe-holdout. This exception does not subdivide Validation
for predictor calibration/lambda selection or change project split membership.
Probe bootstrap is paired and stratified by `(race,Y)` over probe-holdout rows,
separate from the locked Test bootstrap. Real-probe gain does not select
production architecture or decide production PASS/FAIL; see the RQ4 plan.
The final probe optimizer/stopping, float64 stored-probability log-ratio and
synthetic-generation rules are user-locked in that plan and the RQ4b YAML;
implementation must not substitute defaults or silently clip boundary scores.

Validation is independent of CV and full-Train fitting. Each frozen model
produces `P(y_approved = 1)` for the ordered Validation records exactly once for
the threshold-selection stage.

Each model selects its own threshold by maximizing Balanced Accuracy on
Validation. The threshold-selection artifact must identify:

- the model and `run_id`;
- the feature, split, preprocessing, and fitted-model versions;
- the ordered Validation `record_id`, prediction values, and objective values;
- the selected threshold and any approved tie rule.

Validation must not refit preprocessing or the model, update a vocabulary,
change a configuration, or trigger open-ended tuning. Once selected, the
threshold is immutable for Test evaluation and the corresponding RQ1 bootstrap.

The authoritative shared threshold policy for final benchmark models uses the
999 candidates from `0.001` through `0.999` inclusive in steps of `0.001`.
Select the candidate that maximizes Balanced Accuracy. Resolve ties first by
minimum absolute distance to `0.5`, then by the lower threshold. Applying this
fixed evaluation policy to a model's single Validation prediction pass is not
hyperparameter search.

For PlainMLP, the persisted probability matrix is `float64`, finite, shaped
`(n_rows, 2)`, bounded in
`[0, 1]`, and each row sums to one under `rtol = 0`, `atol = 1.0e-7`. Both
columns returned by `predict_proba` are persisted directly: no rounding,
clipping, or persistence-layer reconstruction is allowed.

TabMmini instead persists its canonical finite `float32` vector
`p_approved = P(y_approved = 1)` with shape `(n_rows,)`. If a shared evaluator
requires two columns, its evaluation adapter derives
`[1.0 - p_approved, p_approved]`; column 0 is `P(Denied)` and column 1 is
`P(Approved)`. This derived matrix does not replace the canonical prediction.

## 6. Test evaluation boundary

Test is opened only after the preceding configuration, full-Train fitted state,
and Validation threshold have been frozen. The frozen pipeline produces Test
probabilities without calling `fit`, `partial_fit`, `fit_transform`, or an
equivalent backend update.

The Test prediction package must use the same `run_id` as its preprocessing,
fresh full-Train model, and Validation threshold package. Ordered Test
`record_id`, prediction values, and the locked threshold are checked directly
before metrics are accepted.

The required RQ1 outputs are:

- ROC-AUC;
- PR-AUC for the Denied class;
- Precision, Recall, and F1 for Denied;
- Recall for Approved;
- Balanced Accuracy at the locked threshold;
- Brier score;
- calibration curve, calibration intercept, and calibration slope;
- training time and inference time.

Probabilities are interpreted as `p_approved = P(y_approved = 1)`. When a
Denied-oriented metric is calculated, `y_denied = 1 - y_approved` and
`p_denied = 1 - p_approved` must be used consistently. Test labels may be used
to calculate final metrics, but they must not alter preprocessing, model,
configuration, or threshold.

## 7. Bootstrap and uncertainty boundary

RQ1 uses a 95% confidence interval with 1,000 bootstrap replicates. Bootstrap
sampling is performed over the fixed Test indices, not over newly split data.

The following rules are mandatory:

- all model comparisons use the same replicate indices;
- fitted preprocessing, fitted models, and locked thresholds remain fixed;
- no model is retrained inside a replicate;
- each replicate recomputes only the metric from the resampled Test records;
- the bootstrap record identifies its `run_id`, seed, the `split_v1` Test
  partition, index dimensions, replicate count, and applicable validity rules;
- point estimates and interval summaries remain linked to the prediction and
  threshold artifacts from which they were calculated;
- an independent recomputation compares replicate identifiers and each
  replicate's valid/invalid payload directly;
- ordered Test `record_id`, the locked threshold, and prediction values are
  compared directly against the values recorded for the same `run_id`;
- `calibration_slope` is compared after rounding both stored and recomputed
  values to six decimal places; the other metrics and all interval endpoints
  use `rtol = 0`, `atol = 1.0e-14`; point metrics use `rtol = 0`,
  `atol = 1.0e-12`. No other metric uses decimal rounding;
- the existing locked bootstrap indices must not be regenerated;
- SHA, digest, raw-byte identity, and the probability row-sum tolerance are
  not numerical-equivalence gates for metrics or confidence intervals.

The detailed confidence-interval method for between-model metric differences
and the rule for declaring statistical significance are still open in the Set B
requirements. Until approved and versioned, per-model confidence intervals do
not authorize a claim that one model significantly outperforms another.

## 8. Cross-model comparison boundary

An RQ1 comparison is valid only when the compared runs use:

- the same Dataset B cohort and `split_v1` Test membership;
- the same final ordered 15-feature raw information;
- family-appropriate preprocessing fitted within the same data boundaries;
- a threshold selected separately for each model on Validation;
- the same required metric definitions;
- the same paired Test bootstrap indices for difference estimates.

Encoded widths may differ across model families only when their approved
preprocessing route preserves the same raw business information. A comparison
must not mix results from the superseded 12-feature benchmark with results from
the final 15-feature benchmark, different Test memberships, or thresholds
selected on Test. Superseded runs remain historical evidence and must not be
pooled, ranked, or used as substitutes for a final benchmark run.

## 9. Required evaluation lineage

Every prediction and evaluation package must link, directly or through its run
manifest, all of the following:

| Lineage item | Required content |
| --- | --- |
| Run | One unique `run_id` used by preprocessing, fitted model, Validation predictions and threshold, Test predictions, and evaluation outputs |
| Dataset | Dataset/view version and target mapping |
| Features | Final feature contract and ordered 15 raw inputs |
| Split | `split_v1`, partition, and ordered `record_id` |
| Input normalization | Per-field raw `1111` counts, normalized `Exempt` counts, and PASS/FAIL equality result for all five exemption-coded flags |
| Preprocessing | Version, profile, fitted scope, ordered schema, category lists, and transformation settings |
| Model | Family, exact configuration fields, fresh-fit status, and runtime versions |
| Prediction | Ordered `record_id`, target when permitted, `p_approved` values, `run_id`, and partition |
| Threshold | Validation selection objective, selected value, tie rule, and ordered source predictions |
| Evaluation | Metric definitions, point estimates, replicate identifiers, validity payloads, numerical-comparison rules, and interval method |

For each `run_id`, preprocessing must record the raw `1111` and normalized
`Exempt` counts directly for `reverse_mortgage`,
`open-end_line_of_credit`, `business_or_commercial_purpose`,
`interest_only_payment`, and `balloon_payment`. The equality gate passes only
when each pair of field counts is equal, no semantic `1111` remains, and row
count, order, and `record_id` alignment are preserved.

Acceptance is based on these directly recorded schemas, settings, identifiers,
counts, ordered values, and numerical checks. The same `run_id` must provide an
unbroken link from preprocessing through the model, Validation predictions and
threshold, Test predictions, and final evaluation. A `run_id` is a
cross-reference, not a substitute for the direct checks required by this
contract.

Audit metadata stays outside predictor input and is joined to predictions only
through `record_id` for downstream analysis.

## 10. Conditions for opening RQ2

RQ2 may begin only when:

1. RQ1 artifacts for all six comparison models come from
   complete, independently verified, and frozen final 15-feature benchmark
   runs;
2. each of the six runs has complete `run_id` lineage across preprocessing,
   model, Validation threshold, Test predictions, and evaluation;
3. Dataset B and the final feature lock, including final explanation groups,
   are recorded;
4. fitted models, preprocessing, ordered predictions, and `record_id` lineage
   required for explanation are preserved;
5. the [RQ2 explanation contract](rq2_explanation.md) and
   `config/dataset_b/rq2_protocol.yaml` agree and have passed their stated
   protocol gates.

RQ2 reads frozen RQ1 artifacts. It must not retrain a model, recreate the split,
mix in a superseded 12-feature run, use Test rows for explanation, or infer
unapproved GSV or Shapley-Lorenz settings from an example.

## 11. Conditions for opening RQ3

RQ3 may begin only when:

1. frozen RQ1 predictions from final 15-feature benchmark runs for all six
   required model families are available on the same ordered Test records;
2. each model's predictions have complete `run_id` lineage from preprocessing
   through Test evaluation;
3. prediction rows can be joined one-to-one by `record_id` to the preserved
   audit metadata without placing protected attributes in predictor input;
4. the `applicant_sex` and `derived_race` group mappings, missing/unknown
   handling, reference-group directions, minimum denominators,
   undefined-result rules, calibration residual definition, and CI procedure
   are approved in the RQ3 fairness contract;
5. the audit remains descriptive and does not introduce group-specific model
   thresholds unless a new research decision explicitly authorizes them.

The approved race scope uses `derived_race` in a separate RQ3-only audit view,
joined from the immutable raw source using the one-based source-row number in
the numeric `record_id` suffix (`dataframe position = suffix - 1`). The grouping, denominator,
pairwise, and global-range rules are owned by the RQ3 fairness contract. This
approval does not place `derived_race` in the controlled model view or change
the Test partition, stored predictions, frozen Validation thresholds, or locked
bootstrap indices.

## 12. Prohibited boundary crossings

- Do not regenerate or reorder `split_v1` inside a notebook or model runner.
- Do not fit any preprocessing or model state on Validation or Test.
- Do not place outer fold-validation, project Validation, or Test in an inner
  stopping set, and do not reuse a temporary selector as a final estimator.
- Do not use Validation for repeated, unbounded configuration tuning.
- Do not choose or revise a threshold on Test.
- Do not run final Test inference more than once for a model `run_id`.
- Do not retrain inside a Test bootstrap replicate.
- Do not regenerate locked bootstrap indices or use SHA/digest identity as a
  numerical-comparison gate.
- Do not describe a per-model Test result as a paired cross-model comparison.
- Do not mix superseded 12-feature results with final 15-feature benchmark
  results in a comparison, ranking, bootstrap summary, or RQ2/RQ3 input.
- Do not break one model run across unrelated preprocessing, Validation,
  threshold, Test, or evaluation lineages; the same `run_id` must connect them.
- Do not open RQ2 or RQ3 from legacy or non-frozen benchmark evidence.

## 13. Source traceability

This contract was reconstructed from:

- [Documentation migration record](../archive/documentation_migration_2026-09.md),
  which identifies `Dataset_B_Pipeline.md` as a former source;
- `config/dataset_b/split_v1.yaml`;
- `config/dataset_b/model_protocol_v1.yaml`;
- `artifacts/dataset_b/split_v1.json`;
- `src/dataset_b/splits/split_manifest.py`;
- `src/dataset_b/splits/run_split.py`;
- `docs/Yêu cầu Kỹ thuật Set B cho RQ1 và RQ2 và RQ3 - HMDA NY 2024 - Revised.docx`.

The contract records protocol and evidence boundaries. It intentionally omits
model run status, fitted parameter values, selected thresholds, and model result
tables.
