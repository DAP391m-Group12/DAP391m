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

## 3. Train-only cross-validation

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

## 4. Full-Train refit boundary

After one configuration has been selected for a model family:

1. create a new preprocessing pipeline and a new model instance;
2. fit all learned preprocessing state on the complete Train partition;
3. fit the model on that same complete Train partition;
4. freeze the fitted preprocessing, fitted model, configuration, feature
   version, split version, and runtime identity before reading Validation for
   threshold selection.

Fold-fitted preprocessing or a fold-fitted model is a CV artifact and cannot be
reused as the full-Train fitted state. The full-Train fit must not receive an
internal Validation or Test channel unless a later approved model-specific
protocol explicitly changes the contract and remains compatible with this
partition boundary.

## 5. Validation threshold boundary

Validation is independent of CV and full-Train fitting. Each frozen model
produces `P(y_approved = 1)` for the ordered Validation records exactly once for
the threshold-selection stage.

Each model selects its own threshold by maximizing Balanced Accuracy on
Validation. The threshold-selection artifact must identify:

- the model and run version;
- the feature, split, preprocessing, and fitted-model versions;
- the ordered Validation prediction artifact and its objective values;
- the selected threshold and any approved tie rule.

Validation must not refit preprocessing or the model, update a vocabulary,
change a configuration, or trigger open-ended tuning. Once selected, the
threshold is immutable for Test evaluation and the corresponding RQ1 bootstrap.

## 6. Test evaluation boundary

Test is opened only after the preceding configuration, full-Train fitted state,
and Validation threshold have been frozen. The frozen pipeline produces Test
probabilities without calling `fit`, `partial_fit`, `fit_transform`, or an
equivalent backend update.

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
- the bootstrap record identifies its seed, the `split_v1` Test partition,
  index dimensions, replicate count, and applicable validity rules;
- point estimates and interval summaries remain linked to the prediction and
  threshold artifacts from which they were calculated.

The detailed confidence-interval method for between-model metric differences
and the rule for declaring statistical significance are still open in the Set B
requirements. Until approved and versioned, per-model confidence intervals do
not authorize a claim that one model significantly outperforms another.

## 8. Cross-model comparison boundary

An RQ1 comparison is valid only when the compared runs use:

- the same Dataset B cohort and `split_v1` Test membership;
- the same approved raw feature information;
- family-appropriate preprocessing fitted within the same data boundaries;
- a threshold selected separately for each model on Validation;
- the same required metric definitions;
- the same paired Test bootstrap indices for difference estimates.

Encoded widths may differ across model families only when their approved
preprocessing route preserves the same raw business information. A comparison
must not mix a temporary feature version with a final feature version, different
Test memberships, or thresholds selected on Test.

## 9. Required evaluation lineage

Every prediction and evaluation package must link, directly or through its run
manifest, all of the following:

| Lineage item | Required content |
| --- | --- |
| Dataset | Dataset/view version and target mapping |
| Features | Feature version and ordered raw inputs |
| Split | `split_v1`, partition, and ordered `record_id` |
| Preprocessing | Version, profile, fitted scope, and schema/order |
| Model | Family, exact configuration, fitted-state identifier, and runtime versions |
| Prediction | `record_id`, target when permitted, `p_approved`, model/run identifier, and partition |
| Threshold | Validation selection objective, selected value, and source prediction artifact |
| Evaluation | Metric definitions, point estimates, bootstrap identity, and interval method |

Audit metadata stays outside predictor input and is joined to predictions only
through `record_id` for downstream analysis.

## 10. Conditions for opening RQ2

RQ2 may begin only when:

1. RQ1 artifacts required for the selected comparison models are complete and
   independently verified;
2. Dataset B and the final feature lock, including final explanation groups,
   are versioned;
3. the representative neural model is selected under an approved rule;
4. fitted models, preprocessing, ordered predictions, and `record_id` lineage
   required for explanation are preserved;
5. the RQ2 protocol has approved its explanation observations, background,
   baseline/value-function semantics, dependence treatment, approximation
   budget, Top-k and tie rules, bootstrap validity rules, and CI method.

RQ2 reads frozen RQ1 artifacts. It must not retrain a model, recreate the split,
or infer unapproved GSV or Shapley-Lorenz settings from an example.

## 11. Conditions for opening RQ3

RQ3 may begin only when:

1. frozen RQ1 predictions for all six required model families are available on
   the same ordered Test records;
2. prediction rows can be joined one-to-one by `record_id` to the preserved
   audit metadata without placing protected attributes in predictor input;
3. `applicant_sex` raw-code handling, missing/unknown handling, reference-group
   direction, minimum denominators, undefined-result rules, calibration
   residual definition, and CI procedure are approved;
4. the audit remains descriptive and does not introduce group-specific model
   thresholds unless a new research decision explicitly authorizes them.

Preserving race fields as audit metadata does not automatically expand the
approved RQ3 scope beyond gender. Any race analysis requires its own grouping
and denominator decisions.

## 12. Prohibited boundary crossings

- Do not regenerate or reorder `split_v1` inside a notebook or model runner.
- Do not fit any preprocessing or model state on Validation or Test.
- Do not use Validation for repeated, unbounded configuration tuning.
- Do not choose or revise a threshold on Test.
- Do not retrain inside a Test bootstrap replicate.
- Do not describe a per-model Test result as a paired cross-model comparison.
- Do not open RQ2 or RQ3 from temporary feature-lock evidence alone.

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
