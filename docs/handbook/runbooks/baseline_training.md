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
| Logistic Regression | [01_linear_model_v1.ipynb](../../../notebooks/dataset_b/01_linear_model_v1.ipynb) | `lr` | `lr` | `lr` |
| Random Forest | [02_random_forest_v1.ipynb](../../../notebooks/dataset_b/02_random_forest_v1.ipynb) | `rf` | `tree` | `rf` |
| XGBoost | [03_xgboost_v1.ipynb](../../../notebooks/dataset_b/03_xgboost_v1.ipynb) | `xgb` | `tree` | `xgb` |

There is no repository-level public baseline-training runner. The notebooks are
the executable orchestration layer. Their supported Dataset B interfaces are:

```python
from src.dataset_b.preprocessing import (
    DatasetBSplitLoader,
    make_preprocessing_pipeline,
)
from src.dataset_b.splits import DatasetBSplitter
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
- `config/dataset_b/split_v1.yaml`;
- `data/preprocessing/dataset_b_v1/cv_preprocessing_v2/manifest.json`.

Stop before training unless all of the following hold:

1. protocol is `model_protocol_v1` for `dataset_b`;
2. feature version matches the preprocessing manifest;
3. split version is `split_v1` everywhere;
4. preprocessing version is `dataset_b_preprocessing_v2`;
5. the manifest says `source_partition: train`,
   `fit_scope: fold_train_only`, and `use: cross_validation_only`;
6. encoded width is 66 and folds are numbered 1 through 5;
7. CV is `StratifiedKFold(5, shuffle=True, random_state=42)` on Train only;
8. `hyperparameter_search` is false and the trial budget is one fixed
   configuration per model;
9. class weighting, sample weighting, and resampling are disabled.

Use estimator classes named by the protocol and pass the corresponding
`models.<key>.parameters` mapping unchanged. At the current interface those
classes are `sklearn.linear_model.LogisticRegression`,
`sklearn.ensemble.RandomForestClassifier`, and `xgboost.XGBClassifier`.
Do not add parameters from a notebook experiment or from another model family.

## 4. Train-only cross-validation

### 4.1 Preferred route: exported fold matrices

Use the existing v2 matrices when measuring the fixed baseline configuration.
For each manifest fold:

1. load `<prefix>_train.npz`, `<prefix>_fold_validation.npz`, and `rows.npz`;
2. use prefix `lr` for Logistic Regression and `tree` for RF/XGBoost;
3. verify both matrices are CSR, finite, and exactly 66 columns;
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

If the v2 CV export must be recreated for a newly authorized artifact version,
the implemented exporter is:

```powershell
python -m src.dataset_b.preprocessing.export_cv
```

The current exporter refuses an existing destination. Do not delete or
overwrite the existing v2 root merely to rerun it, and do not use an export
created under a different feature, split, or preprocessing version.

## 5. Fresh full-Train fit and Validation predictions

After CV checks and configuration lock:

1. load Train and Validation with `DatasetBSplitLoader`;
2. confirm both expose the same ordered 12 raw model inputs and keep identifier,
   target source, target, and audit columns outside `X`;
3. create a new preprocessing pipeline and a new estimator;
4. fit the combined pipeline once on all Train rows;
5. freeze the feature, split, preprocessing, model-parameter, and runtime
   identities;
6. call `predict_proba(validation.X)[:, 1]` without any refit;
7. verify finite probabilities in `[0, 1]`, class order `[0, 1]`, unique ordered
   Validation identifiers, and one probability per identifier.

The persisted model must be the complete fitted pipeline, not the estimator
alone. Keep the ordered Validation prediction table with at least
`record_id`, `y_approved`, and `p_approved`. Record fit/prediction timing,
warnings, exact parameters, library versions, and source contract versions in
the run manifest.

Write model artifacts under that model run's artifact directory, never inside
the preprocessing artifact root. If an output path already exists, verify its
manifest and contents; do not silently replace a partial or incompatible run.
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
- the ordered Validation prediction file and its hash;
- all feature, split, preprocessing, model, and run identities.

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
5. hash the prediction artifact and link it to the run manifest;
6. hand the frozen predictions to the evaluation procedure in the split and
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
- warnings, timings, library versions, file hashes, and reload verification.

Stop the run when any contract/version mismatch, row-alignment failure,
non-finite value, unexpected class order, unsupported overwrite, or estimator
failure is detected. Record convergence warnings; follow the model notebook's
explicit publication guard instead of treating a warning as success.

## 9. Prohibited shortcuts

- Do not regenerate or reorder `split_v1` in a model notebook.
- Do not use Validation or Test in cross-validation.
- Do not fit preprocessing or model state on Validation or Test.
- Do not reconstruct a fitted preprocessor from manifest summaries.
- Do not reuse a fold-fitted pipeline for full-Train inference.
- Do not add class/sample weighting, resampling, early stopping, or a search
  budget absent from the fixed protocol.
- Do not select or revise a threshold on Test.
- Do not report a saved result or completion status from this runbook.

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
- `src/dataset_b/preprocessing/split_inputs.py`;
- `src/dataset_b/preprocessing/export_cv.py`;
- `src/dataset_b/splits/__init__.py`;
- `src/dataset_b/splits/split_manifest.py`;
- the three baseline notebooks listed in Section 2.

The notebooks remain the execution evidence. This runbook describes how to
operate and verify them; it does not turn notebook code examples into new
repository APIs.
