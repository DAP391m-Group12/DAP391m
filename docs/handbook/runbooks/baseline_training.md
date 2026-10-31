# Dataset B baseline training runbook

## 1. Purpose and ownership

This runbook is the operational path for the three fixed Phase 1 classical
baselines:

- Logistic Regression (`lr`);
- Random Forest (`rf`);
- XGBoost (`xgb`).

It explains how a coder prepares, runs, and checks Train-only cross-validation,
fresh full-Train fitting, Validation threshold selection, and the gated Test
stage. It does not define preprocessing semantics or evaluation policy. Read
[Classical preprocessing](../contracts/preprocessing_classical.md) and
[Split and evaluation](../contracts/split_and_evaluation.md) as the controlling
contracts.

This runbook does not report saved model results or current run status.

## 2. Executable entry points and public interfaces

The model runs are implemented in these notebooks:

| Model | Notebook | Protocol key | CV matrix prefix | Raw profile |
| --- | --- | --- | --- | --- |
| Logistic Regression | [01_linear_model_v1.ipynb](../../../notebooks/dataset_b/v1/01_linear_model_v1.ipynb) | `lr` | `lr` | `lr` |
| Random Forest | [02_random_forest_v1.ipynb](../../../notebooks/dataset_b/v1/02_random_forest_v1.ipynb) | `rf` | `tree` | `rf` |
| XGBoost | [03_xgboost_v1.ipynb](../../../notebooks/dataset_b/v1/03_xgboost_v1.ipynb) | `xgb` | `tree` | `xgb` |

For the final XGBoost workstream, `xgb` remains the machine-readable protocol
key and source-module prefix (`xgb_`), while `xgboost` is the canonical
reporting key and artifact directory name. Final run artifacts belong under
`artifacts/dataset_b/xgboost/{working,final}/<run_id>/`. This naming adapter
must be explicit; consumers must not infer that the two keys are interchangeable.

There is no repository-level public baseline-training runner. The notebooks are
the executable orchestration layer. Their supported Dataset B interfaces are:

```python
from src.dataset_b.data import DatasetBSplitLoader, DatasetBSplitter
from src.dataset_b.preprocessing import make_preprocessing_pipeline
```

- `DatasetBSplitLoader().load(partition)` loads one immutable partition and
  checks schema, ordered `record_id`, and target mapping.
- `make_preprocessing_pipeline("lr" | "rf" | "xgb")` returns a fresh,
  unfitted classical preprocessing pipeline.
- `DatasetBSplitter.from_yaml().cv_splitter()` returns the mandated five-fold
  Train-only splitter.

Do not invent a baseline CLI or import notebook-local helper functions as if
they were a public API.

## 3. Preflight

Run from the repository environment and locate the project root by the
presence of `config/dataset_b/model_protocol_v1.yaml`. Before fitting a model,
read:

- `config/dataset_b/model_protocol_v1.yaml`;
- `config/dataset_b/feature_set_v1.yaml`;
- `config/dataset_b/split_v1.yaml`.

Create one new `run_id` when the authorized final retraining execution begins.
Reuse that same identifier unchanged in the preprocessing export, CV/OOF
outputs, full-Train fitted pipeline, later Validation threshold, Test
predictions, model working directory, model final directory, and every
manifest. A downstream stage of the same execution must not create another
`run_id` merely because training or evaluation has advanced to a new stage.
A new `run_id` is required only for a genuinely new execution, including one
that changes the feature, split, preprocessing, model configuration, or fitted
state lineage. Suggested run-scoped locations are:

- `data/preprocessing/dataset_b_v1/cv_preprocessing_v2/<run_id>/`;
- `artifacts/dataset_b/<model_key>/working/<run_id>/`;
- `artifacts/dataset_b/<model_key>/final/<run_id>/`.

Regenerate the preprocessing inputs for this `run_id`, then read
`data/preprocessing/dataset_b_v1/cv_preprocessing_v2/<run_id>/manifest.json`
and run the following gate before fitting any estimator.

Before any pilot, CV fold, or full-Train fit, run the final data-quality gate
and stop unless all of the following hold:

1. protocol is `model_protocol_v1` for `dataset_b`;
2. the ordered final feature schema contains exactly the approved 15 raw
   inputs and matches the preprocessing manifest;
3. split version is `split_v1` everywhere;
4. preprocessing version is `dataset_b_preprocessing_v2`;
5. the manifest says `source_partition: train`,
   `fit_scope: fold_train_only`, and `use: cross_validation_only`;
6. the retained Train, Validation, and Test membership, order, and row counts
   match `split_v1`, with no row rejected from the current membership;
7. no retained final input is null, empty, whitespace-only, or textual `nan`;
8. every raw token satisfies its field domain and numeric validity rules;
9. for each of the five exemption-coded flags,
   `raw_1111_count == normalized_exempt_count`;
10. `semantic_1111_count == 0` for every model-facing input;
11. normalization preserves retained row order, `record_id`, target, and
    partition alignment;
12. the semantic schema has exactly 19 ordered columns and the classical OHE
    schema has exactly 62 ordered columns;
13. folds are numbered 1 through 5 and CV is
    `StratifiedKFold(5, shuffle=True, random_state=42)` on Train only;
14. `hyperparameter_search` is false and the trial budget is one fixed
    configuration per model;
15. model parameter, preprocessing, dtype, category, and runtime fields match
    their approved configuration directly.

Record the input and retained row counts, blank counts, invalid-token counts,
the two exemption counts, equality-gate result, ordered schemas, and all
configuration fields in the run manifest. Any failed check is `FAIL` and stops
the execution before fitted preprocessing or model state is created.

Use estimator classes named by the protocol. For Logistic Regression and
Random Forest, pass the corresponding `models.<key>.parameters` mapping
unchanged. For XGBoost, preserve that mapping as the fixed base parameters and
add only the fit-local `scale_pos_weight` required by Section 3.1. At the
current interface those classes are `sklearn.linear_model.LogisticRegression`,
`sklearn.ensemble.RandomForestClassifier`, and `xgboost.XGBClassifier`.
Do not add parameters from a notebook experiment or from another model family.

### 3.1 Fixed class-imbalance policy

Random Forest uses sklearn `class_weight="balanced"`. Sklearn derives those
weights separately from the labels in the population passed to each fit: the
current fold-train rows for each CV fold, and all Train rows for the fresh
full-Train fit. Validation and Test labels must not contribute to either
calculation.

For XGBoost, the approved executable policy is
`scale_pos_weight = n_denied / n_approved`, with both counts taken from the
population used by that fit. The runner must calculate and add the ratio to a
fresh estimator separately for each CV fold-train population and again from
full Train for the fresh full-Train fit. `scale_pos_weight` is therefore not a
fixed value in `models.xgb.parameters`. Validation and Test labels must not
contribute to either calculation.

These weighting rules are fixed fit-time rules, not hyperparameter tuning.
`sample_weight: false`, `resampling: false`, and
`hyperparameter_search: false` remain in force.

## 4. Train-only cross-validation

### 4.1 Required route: regenerated fold matrices

Regenerate the fold matrices for the new 15-feature `run_id` before measuring
the fixed baseline configuration. Do not resume, extend, or relabel a fitted
preprocessor or matrix export from the superseded 12-feature runs. For each
manifest fold:

1. load `<prefix>_train.npz`, `<prefix>_fold_validation.npz`, and `rows.npz`;
2. use prefix `lr` for Logistic Regression and `tree` for RF/XGBoost;
3. verify both matrices are CSR, finite, and exactly 62 columns;
4. verify matrix row counts match the labels, positions, and `record_id` arrays
   in `rows.npz`;
5. verify stored positions reproduce the ordered Train identifiers and labels;
6. create a new estimator from the exact protocol parameters;
7. fit on the fold-train matrix only;
8. calculate `predict_proba(... )[:, 1]` on fold-validation and verify the
   class order is `[0, 1]`;
9. store the probability against the fold-validation `record_id` and calculate
   fold ROC-AUC.

After all five folds, verify every Train position received exactly one
out-of-fold prediction. Report the mean and standard deviation of fold
ROC-AUC. These measurements check performance and stability; they are not a
hyperparameter search.

Do not attach another preprocessing pipeline to an already transformed `.npz`
matrix. The exported matrices are CV-only and cannot be used for Validation or
Test.

### 4.2 Raw-input route

Use raw input only when the operation requires a fitted pipeline, such as an
explicit raw-input CV check or the full-Train fit:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

estimator_class = {
    "lr": LogisticRegression,
    "rf": RandomForestClassifier,
    "xgb": XGBClassifier,
}[model_key]
model = Pipeline([
    ("preprocess", make_preprocessing_pipeline(model_key)),
    (
        "estimator",
        estimator_class(**protocol["models"][model_key]["parameters"]),
    ),
])
```

For raw-input CV, load only `DatasetBSplitLoader().load("train")`, obtain folds
from `DatasetBSplitter.from_yaml().cv_splitter()`, and clone or rebuild the
entire pipeline for every fold. Never carry median, scaling, category, or model
state from one fold to another.

Generate the run-scoped CV preprocessing inputs with the implemented exporter:

```powershell
python -m src.dataset_b.workflows.export_classical_cv --run-id <run_id>
```

The exporter must target the new `run_id` directory and refuse an existing
destination. Do not delete or overwrite a historical root merely to rerun it,
and do not use an export created under a different feature, split,
preprocessing, or `run_id` lineage. Validate the regenerated 15/19/62 schemas
and the data-quality evidence before fitting the first CV estimator. If the
current exporter cannot write the run-scoped final schema or cannot record the
required quality evidence, stop before CV; do not fall back to the legacy
export.

The former `src.dataset_b.preprocessing.export_cv` command remains a
compatibility alias to the canonical workflow.

## 5. Fresh full-Train fit and Validation predictions

After CV checks and configuration lock:

1. load Train and Validation with `DatasetBSplitLoader`;
2. confirm both expose the same ordered 15 raw model inputs and keep identifier,
   target source, target, and audit columns outside `X`;
3. rerun the data-quality gate and confirm the 19-column semantic and
   62-column encoded schemas against the run manifest;
4. create a new preprocessing pipeline and a new estimator; do not load or
   resume any fitted 12-feature preprocessing or model state;
5. fit the combined pipeline once on all Train rows;
6. freeze the feature, split, preprocessing, model-parameter, runtime fields,
   and `run_id`;
7. call `predict_proba(validation.X)[:, 1]` without any refit;
8. verify finite probabilities in `[0, 1]`, class order `[0, 1]`, unique ordered
   Validation identifiers, and one probability per identifier.

The persisted model must be the complete fitted pipeline, not the estimator
alone. Keep the ordered Validation prediction table with at least
`record_id`, `y_approved`, and `p_approved`. Record fit/prediction timing,
warnings, exact parameters, library versions, and source contract versions in
the run manifest together with the `run_id`, ordered schemas, category lists,
normalization counts, and equality-gate result.

Write model artifacts under that model run's `working/<run_id>` directory,
never inside the preprocessing artifact root. If an output path already
exists, stop rather than silently replacing or resuming a partial or
incompatible run.
After persistence, reload the fitted pipeline and predictions and confirm that
identifiers, labels, and probabilities reproduce the in-memory result within
the model-appropriate numeric tolerance.

## 6. Validation threshold gate

Validation is opened only after the full-Train pipeline and configuration are
frozen. Select one model-specific threshold using the protocol objective,
Balanced Accuracy, and preserve:

- the candidate set and objective values;
- the approved tie rule;
- the selected threshold;
- the ordered Validation `record_id`, labels, and prediction values;
- all feature, split, preprocessing, model, and runtime fields plus the same
  `run_id`.

Threshold selection must not refit preprocessing or the estimator, change a
feature/configuration, or start an open-ended search. A change to model or
preprocessing state invalidates the threshold and requires a new full-Train
run and new Validation predictions.

## 7. Test gate and evaluation handoff

Do not load Test until the full-Train pipeline, Validation threshold, and their
lineage are frozen. At the Test stage:

1. load Test once through `DatasetBSplitLoader`;
2. reload and verify the persisted full pipeline and locked threshold;
3. produce `p_approved` without `fit`, `fit_transform`, `partial_fit`, or any
   equivalent update;
4. preserve ordered `record_id`, `y_approved`, `p_approved`, and the decisions
   obtained from the locked threshold;
5. compare the ordered records, threshold, prediction values, shapes, dtype,
   finiteness, and range directly with the run manifest;
6. link the prediction package to the same `run_id`;
7. hand the frozen predictions to the evaluation procedure in the split and
   evaluation contract.

Test labels may be used for the specified final metrics only. They must not
change preprocessing, model parameters, threshold, or feature policy.

## 8. Required checks and stop conditions

For each model, retain enough evidence to check:

- exact estimator class and protocol parameters;
- feature, split, preprocessing, and protocol versions;
- fold membership, row counts, shapes, identifiers, and labels;
- finite matrices and probabilities;
- `[0, 1]` estimator class order and `p_approved` semantics;
- one out-of-fold prediction per Train record;
- fresh full-Train preprocessing and estimator state;
- ordered Validation and Test alignment;
- raw `1111` counts, normalized `Exempt` counts, and the equality result;
- semantic `1111` count of zero;
- warnings, timings, library versions, direct manifest/content checks, and
  reload verification.

Stop the run when any contract/version mismatch, row-alignment failure,
non-finite value, unexpected class order, unsupported overwrite, or estimator
failure is detected. Record convergence warnings; follow the model notebook's
explicit publication guard instead of treating a warning as success.

After CV, fresh full-Train fit, Validation threshold selection, Test
prediction, and the required evaluation checks all pass, validate the complete
working bundle directly against its manifest. Set its state to `COMPLETE`,
close all file handles, publish it atomically as `final/<run_id>`, validate the
final directory again, and freeze it against overwrite. A model file, a Test
prediction file, or a completion marker without the directly validated bundle
is not completion evidence.

## 9. Prohibited shortcuts

- Do not regenerate or reorder `split_v1` in a model notebook.
- Do not use Validation or Test in cross-validation.
- Do not fit preprocessing or model state on Validation or Test.
- Do not reconstruct a fitted preprocessor from manifest summaries.
- Do not reuse a fold-fitted pipeline for full-Train inference.
- Do not resume or reuse any fitted preprocessor, model, checkpoint, matrix, or
  prediction from a superseded 12-feature run.
- Do not initialize one model from fitted state belonging to another `run_id`.
- Do not add class or sample weighting beyond the fixed RF rule and the
  executable fit-local XGBoost rule in the protocol; do not add resampling, early
  stopping, or a search budget absent from the fixed protocol.
- Do not select or revise a threshold on Test.
- Do not treat this runbook itself as evidence that a model is complete; only
  a directly validated final run manifest may carry `COMPLETE`.

## 10. Source traceability

This runbook was reconstructed from:

- [Documentation migration record](../archive/documentation_migration_2026-09.md),
  which identifies `Dataset_B_v1_preprocessing.md` as a former source;
- `docs/handbook/contracts/preprocessing_classical.md`;
- `docs/handbook/contracts/split_and_evaluation.md`;
- `config/dataset_b/model_protocol_v1.yaml`;
- `config/dataset_b/feature_set_v1.yaml`;
- `config/dataset_b/split_v1.yaml`;
- `src/dataset_b/preprocessing/__init__.py`;
- `src/dataset_b/preprocessing/profiles.py`;
- `src/dataset_b/data/split_inputs.py`;
- `src/dataset_b/workflows/export_classical_cv.py`;
- compatibility path `src/dataset_b/preprocessing/export_cv.py`;
- `src/dataset_b/data/__init__.py`;
- `src/dataset_b/data/split_manifest.py`;
- the three baseline notebooks listed in Section 2.

The notebooks remain the execution evidence. This runbook describes how to
operate and verify them; it does not turn notebook code examples into new
repository APIs.
