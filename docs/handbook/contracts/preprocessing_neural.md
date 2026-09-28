# Dataset B neural preprocessing contract

## 1. Purpose and ownership

This contract defines the preprocessing boundary shared by the Dataset B
neural routes under `dataset_b_neural_preprocessing_v1`. It owns:

- the shared neural semantic layer after the deterministic raw parser;
- the PlainMLP external-OHE route;
- the RealMLP and TabMmini native-input routes;
- container, dtype, category-support, and artifact requirements.

The ordered 12 raw predictors, parser semantics, and numeric two-channel
construction are canonical in
[Classical preprocessing](preprocessing_classical.md). This contract imports
that interface rather than redefining it. Split ownership and fit populations
are canonical in [Split and evaluation](split_and_evaluation.md).

Model training, execution history, run status, and evaluation results are
outside this contract.

## 2. Imported input boundary

The neural routes consume the same temporary 12-input training interface and
the same deterministic parser used by the classical routes. The parser and
two-channel layer must produce these 16 columns in this exact order:

1. `loan_type`
2. `loan_purpose`
3. `lien_status`
4. `reverse_mortgage`
5. `open-end_line_of_credit`
6. `business_or_commercial_purpose`
7. `income_value`
8. `income_status`
9. `debt_to_income_ratio`
10. `loan_amount_value`
11. `loan_amount_status`
12. `loan_to_value_ratio_value`
13. `loan_to_value_ratio_status`
14. `loan_term_value`
15. `loan_term_status`
16. `occupancy_type`

The four value columns are `income_value`, `loan_amount_value`,
`loan_to_value_ratio_value`, and `loan_term_value`. The other 12 columns are
categorical or reporting-status channels. `record_id`, target fields, audit
columns, and unresolved feature candidates are not model inputs.

The use of these inputs remains governed by the temporary feature lock in
[Feature governance](feature_governance.md); this contract does not convert it
into a final feature decision.

## 3. Shared neural semantic layer

Each neural route applies the following ordered operations:

1. validate and parse the 12 raw predictors using the imported deterministic
   parser contract;
2. create the four numeric value/status pairs;
3. learn each numeric placeholder median from the permitted fit population in
   original scale;
4. fill only the numeric value channels with those medians;
5. apply `log1p` only to the four numeric value channels;
6. preserve all categorical and reporting-status meanings;
7. pass the resulting 16-column semantic frame to the selected engine
   adapter.

No neural route applies `RobustScaler` or another project-level numeric
scaler. A backend may perform documented internal transformations only after
the native-input boundary; those transformations are not project
preprocessing artifacts.

Before and after `log1p`, the implementation must verify:

- every numeric value is finite;
- every numeric value is non-negative before transformation;
- only the four numeric value columns change;
- category/status values and the 16-column order remain unchanged;
- row count and `record_id` alignment remain unchanged;
- repeating a transform with the same fitted state is deterministic.

## 4. Fit-scope boundary

| Stage | Fitted population | Permitted application |
| --- | --- | --- |
| CV fold | Fold-train only | Matching fold-validation only |
| Final RQ1 fit | Full Train only | Validation, then Test after the split contract permits it |

Every CV fold receives a fresh preprocessing pipeline. Fold-local medians,
category-support observations, encoders, and any adapter-fitted state must not
be shared across folds. Validation and Test must not contribute fitted state.

Fold-fitted artifacts are `cross_validation_only`. They cannot be reused as
the full-Train preprocessor for Validation or Test. Full-Train preprocessing
must be fitted afresh and stored separately from fold-local artifacts.

## 5. Category semantics and support gate

The fixed category schema preserves valid `NA`, `Exempt`, `MISSING`, and
`1111` meanings exactly as defined by the imported parser/schema contract.
The project preprocessing layer must not:

- merge or delete rare but schema-valid categories automatically;
- consolidate categories using the target;
- resample rows to balance predictor categories;
- silently map an unknown code to an all-zero or missing representation.

For every schema category in every categorical/status column, the adapter must
evaluate support before creating a fold artifact:

| Fold-train support | Fold-validation support | Required outcome |
| --- | --- | --- |
| Present | Any | Pass and record counts |
| Absent | Absent | Preserve schema position and record a warning |
| Absent | Present | Fail before training and require an explicit decision |

This gate applies before both external OHE and native category conversion.

## 6. Engine routes

### 6.1 PlainMLP: external fixed-schema OHE

The PlainMLP route is:

`raw 12 -> parser -> two-channel 16 -> log1p -> fixed-schema OHE -> CSR float32`

The encoded representation has exactly 66 columns: four numeric value columns
followed by the fixed categorical/status indicators. `OBSERVED` is the omitted
reference level for each numeric status channel. The encoder must reject
unknown categories and must retain schema-defined columns even when a category
is absent from a fold.

The stored artifact is SciPy CSR with `float32` values. PlainMLP may convert
only the current batch to a dense `float32` tensor at the model boundary. The
full dataset must not be materialized as a dense matrix merely to feed the
model.

### 6.2 RealMLP: native 16-column frame

The RealMLP route is:

`raw 12 -> parser -> two-channel 16 -> log1p -> native DataFrame adapter`

There is no project-level external OHE. The adapter must emit a pandas
DataFrame with the exact 16-column order:

- the four numeric value columns are finite `float32`;
- the 12 categorical/status columns use pandas `category` dtype;
- each categorical dtype contains the complete fixed category list in the
  contract order, including schema-valid categories absent from fit data;
- missing categorical codes after conversion are forbidden.

Categorical column names are passed explicitly to the backend. Encoding or
other transformations performed inside the backend remain backend-owned and
must not be described as exported project OHE artifacts.

### 6.3 TabMmini: native route

TabMmini imports the same 16-column semantic frame and does not use the
PlainMLP external-OHE artifact. Numeric values must remain finite `float32`.
When its adapter materializes indexed categorical tensors, their dtype must be
integer/`long`.

The TabMmini adapter or its manifest must define the category-to-index maps,
unseen-category behavior, exact output container, and any backend-owned
transform. Those details must not be inferred from the RealMLP adapter.

## 7. Dtype and container contract

| Boundary | Required container | Required dtype |
| --- | --- | --- |
| Shared semantic frame | pandas DataFrame, 16 ordered columns | finite numeric values; categorical/status meanings preserved |
| PlainMLP stored input | SciPy CSR, 66 columns | `float32` |
| PlainMLP model batch | dense tensor for the current batch only | `float32` |
| RealMLP native input | pandas DataFrame, 16 ordered columns | numeric `float32`; categorical pandas `category` |
| TabMmini native input | adapter-defined native container | numeric `float32`; indexed categorical tensors integer/`long` |

An adapter must fail on a different column set/order, a non-finite numeric
value, an unsupported category, or a dtype/container that violates its route.

## 8. Preprocessing artifact contract

### 8.1 Common manifest requirements

Every persisted neural preprocessing artifact must record:

- dataset, preprocessing version, route/profile, and split version;
- source partition and exact fit scope;
- ordered raw and transformed schemas;
- numeric medians and `log1p` audit results where applicable;
- fixed category schema and per-fold category-support outcomes;
- output container, shape, dtype, and adapter boundary;
- row counts and `record_id` alignment evidence;
- relevant library versions;
- source/config hashes and hashes of persisted artifact files.

The preprocessing version is `dataset_b_neural_preprocessing_v1`. A manifest
must describe observed artifact properties, not merely copy expected config
values.

### 8.2 PlainMLP CV artifact

The fold-local export destination is:

`data/preprocessing/dataset_b_v1/cv_neural_preprocessing_v1/`

When an export is produced, it must contain one manifest and, for each of five
folds:

- `plain_mlp_train.npz`;
- `plain_mlp_fold_validation.npz`;
- `rows.npz` for row-alignment evidence.

The manifest contract requires five fold-train matrices and five matching
fold-validation matrices, all CSR `float32`, all 66 columns, with 10 matrices
in total. Each fold entry records the learned medians, log audit, category
support, shapes, and file hashes. This export is for cross-validation only;
artifact existence and acceptance state are recorded in [STATUS](../STATUS.md).

### 8.3 Native-route artifacts

RealMLP and TabMmini do not consume the PlainMLP matrix export. Their run or
preprocessing manifests must instead bind the native 16-column adapter to:

- the preprocessing and split versions;
- the exact schema and dtype audit;
- category schema, observed support, and adapter-fitted state;
- backend-owned transformation metadata needed for reproducibility;
- source/config and persisted-artifact hashes.

A model-run manifest that records only model outputs or training controls does
not by itself satisfy this preprocessing artifact contract.

## 9. Acceptance gates

A neural preprocessing artifact is acceptable only when all applicable checks
pass:

1. the 12 raw inputs and 16 semantic columns match the imported contracts;
2. fitted state comes only from fold-train or full Train, as appropriate;
3. all numeric values are non-negative before `log1p` and finite afterward;
4. category/status meanings and fixed schema are preserved;
5. absent-category support is audited with the outcomes in Section 5;
6. PlainMLP output is CSR `float32` with exactly 66 columns and batch-only
   densification;
7. RealMLP output is the native 16-column DataFrame with the required dtypes;
8. TabMmini records its own native mapping and container rather than borrowing
   undocumented RealMLP behavior;
9. row alignment, source/config hashes, artifact hashes, and actual library
   versions are recorded;
10. fold-local and full-Train artifacts are distinct and cannot be confused.

## 10. Evidence sources

This contract was reconciled against:

- [Documentation migration record](../archive/documentation_migration_2026-09.md),
  which identifies `Dataset_B_Engine_preprocesing.md` as a former source;
- `docs/handbook/contracts/preprocessing_classical.md`;
- `docs/handbook/contracts/split_and_evaluation.md`;
- `docs/handbook/contracts/feature_governance.md`;
- `config/dataset_b/plain_mlp_v1.yaml`;
- `config/dataset_b/realmlp_v1.yaml`;
- `src/dataset_b/preprocessing/neural_log_transformer.py`;
- `src/dataset_b/preprocessing/neural_profiles.py`;
- `src/dataset_b/preprocessing/export_neural_cv.py`;
- `src/dataset_b/preprocessing/realmlp_native.py`.

These sources support the preprocessing interface only. They are not evidence
for training completion, model quality, or Test performance.
