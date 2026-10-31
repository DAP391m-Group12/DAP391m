# Dataset B neural preprocessing contract

## 1. Purpose and ownership

This contract defines the preprocessing boundary shared by the Dataset B
neural routes under `dataset_b_neural_preprocessing_v1`. It owns:

- the ordered 15-column raw input interface imported from final feature
  governance;
- the shared 19-column neural semantic layer after deterministic parsing and
  normalization;
- the PlainMLP external-OHE route;
- the RealMLP and TabMmini native-input routes;
- container, dtype, category-support, run identity, and artifact requirements.

The ordered raw predictors, parser semantics, exemption normalization, and
numeric two-channel construction are canonical in
[Classical preprocessing](preprocessing_classical.md). This contract imports
that interface rather than redefining it. The input-quality boundary is
canonical in [Data contract](data_contract.md). Split ownership and fit
populations are canonical in
[Split and evaluation](split_and_evaluation.md).

Model metrics, execution history, current run status, and evaluation results
are outside this contract.

## 2. Imported input boundary

All neural routes consume the final ordered 15-input interface through the
same deterministic input-quality gate used by the classical routes. That gate
is the sole owner of blank handling, raw-domain validation, and raw `1111` to
semantic `Exempt` normalization. It emits the normalized 15-column frame that
enters the fitted two-channel layer. The two-channel layer must then produce
these 19 semantic columns in this exact order:

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
17. `preapproval`
18. `interest_only_payment`
19. `balloon_payment`

The four value columns are `income_value`, `loan_amount_value`,
`loan_to_value_ratio_value`, and `loan_term_value`. The other 15 columns are
categorical or reporting-status channels. `record_id`, target fields, audit
columns, and excluded candidates are not model inputs.

The input is governed by the final 15-field `FINAL_MODEL_FEATURES` decision in
[Feature governance](feature_governance.md). This preprocessing contract does
not reopen feature selection.

## 3. Shared neural semantic layer

Each neural route applies the following ordered operations:

1. pass the 15 raw predictors through the final input-quality gate, which
   validates raw domains and centrally normalizes raw `1111` to semantic
   `Exempt` for the five approved exemption-coded flags;
2. accept only the gate's PASS output as the normalized 15-column model input;
3. create the four numeric value/status pairs without parsing or normalizing
   the input a second time;
4. learn each numeric placeholder median from the permitted fit population in
   original scale;
5. fill only the numeric value channels with those medians;
6. apply `log1p` only to `income_value` and `loan_amount_value`; leave
   `loan_to_value_ratio_value` and `loan_term_value` unchanged after imputation;
7. for PlainMLP only, fit `RobustScaler` on all four numeric value columns
   using the permitted fit population and transform only those four columns;
   do not scale OHE indicators;
8. preserve all categorical and reporting-status meanings;
9. pass the resulting 19-column semantic frame to the selected engine adapter.

The implementation boundary is therefore fixed as:

`raw partition -> quality gate/parser/normalizer -> normalized 15 -> fold-local two-channel 19 -> model-specific tail`

The quality gate has no fitted state. Median imputation is fitted state and
must be learned separately on each fold-train population or afresh on full
Train. Neural pipeline factories and engine adapters must start from the
normalized 15-column frame and must not invoke the raw parser again.

PlainMLP requires the project-level `RobustScaler` step defined above. RealMLP
and TabMmini do not inherit that scaler by implication; a native backend may
perform documented internal transformations only after its native-input
boundary, and those transformations are not project preprocessing artifacts.

Before and after `log1p`, the implementation must verify:

- every numeric value is finite;
- `income_value` and `loan_amount_value` are non-negative before `log1p`;
- only `income_value` and `loan_amount_value` change during `log1p`;
- `loan_to_value_ratio_value` and `loan_term_value` remain unchanged during
  `log1p`;
- category/status values and the 19-column order remain unchanged;
- retained row count and `record_id` alignment remain unchanged;
- repeating a transform with the same fitted state is deterministic.

The central normalizer must additionally verify, for each of
`reverse_mortgage`, `open-end_line_of_credit`,
`business_or_commercial_purpose`, `interest_only_payment`, and
`balloon_payment`, that the raw `1111` count equals the semantic `Exempt`
count. No semantic/model-facing `1111` may remain. `income = 1111` remains the
observed numeric value `1111` and is not part of this field-specific mapping.

## 4. Fit-scope and fresh-state boundary

| Stage | Fitted population | Permitted application |
| --- | --- | --- |
| CV fold | Fold-train only | Matching fold-validation only |
| Final RQ1 fit | Full Train only | Validation, then Test after the split contract permits it |

Every CV fold receives a fresh preprocessing pipeline. Fold-local medians,
category-support observations, encoders, and adapter-fitted state must not be
shared across folds. Validation and Test must not contribute fitted state.

Fold-fitted artifacts are `cross_validation_only`. They cannot be reused as
the full-Train preprocessor for Validation or Test. Full-Train preprocessing
must be fitted afresh and stored separately from fold-local artifacts.

PlainMLP, RealMLP, and TabMmini must all be retrained from fresh model state
against the final semantic contract. A final retraining run must not reuse
model weights, optimizer state, checkpoints, fitted preprocessing state, or a
warm-start state from a legacy run.

## 5. Category semantics and support gate

The fixed model-facing category schema preserves valid `NA` and `Exempt`
meanings exactly as defined by the imported parser and normalizer contract.
There is no `MISSING` category, and the native categorical schema contains no
`1111`. PlainMLP, RealMLP, and TabMmini all receive semantic `Exempt` for the
five exemption-coded flags.

Raw-to-semantic mapping is owned only by the shared deterministic normalizer.
PlainMLP OHE, the RealMLP adapter, the TabMmini adapter, and their backends must
not implement an independent `1111` mapping or reinterpret `Exempt`.

The project preprocessing layer must not:

- merge or delete rare but schema-valid categories automatically;
- consolidate categories using the target;
- resample rows to balance predictor categories;
- silently map an unknown code to an all-zero or missing representation.

For every schema category in every categorical/status column, including
`Exempt`, the adapter must evaluate support before creating a fold artifact:

| Fold-train support | Fold-validation support | Required outcome |
| --- | --- | --- |
| Present | Any | Pass and record counts |
| Absent | Absent | Preserve schema position and record a warning |
| Absent | Present | Fail before training and require an explicit decision |

The gate records `Exempt` support separately for every field whose semantic
schema permits it. This gate applies before both external OHE and native
category conversion.

## 6. Engine routes

### 6.1 PlainMLP: external fixed-schema OHE

The PlainMLP route is:

`raw 15 -> quality gate/parser/normalizer -> normalized 15 -> two-channel 19 -> log1p on income and loan amount -> RobustScaler on four numeric value columns -> fixed-schema OHE -> CSR float64`

The encoded representation has exactly 62 columns: four numeric value columns
followed by 58 fixed categorical/status indicators. `OBSERVED` is the omitted
reference level for each numeric status channel. The encoder must reject
unknown categories and retain schema-defined columns even when a category is
absent from a fold. It encodes semantic `Exempt`; it has no `1111` or
`MISSING` output column.

The scaler is fitted only on the applicable inner-fit, complete outer
fold-train, full-Train inner-fit, or complete full-Train population, matching
the estimator stage. The stored artifact is SciPy CSR with `float64` values.
PlainMLP may convert only the current batch to a dense `float64` tensor at the
model boundary. The full dataset must not be materialized as a dense matrix
merely to feed the model.

### 6.2 RealMLP: native 19-column frame

The RealMLP route is:

`raw 15 -> quality gate/parser/normalizer -> normalized 15 -> two-channel 19 -> log1p on income and loan amount -> native DataFrame adapter`

There is no project-level external OHE. The adapter must emit a pandas
DataFrame with the exact 19-column order:

- the four numeric value columns are finite `float32`;
- the 15 categorical/status columns use pandas `category` dtype;
- each categorical dtype contains the complete fixed semantic category list
  in contract order, including schema-valid `Exempt` categories absent from
  fit data;
- categorical codes missing after conversion are forbidden.

Categorical column names are passed explicitly to the backend. Encoding or
other transformations performed inside the backend remain backend-owned and
must not be described as exported project OHE artifacts. The backend receives
the already-normalized semantic frame and must not remap `1111` or `Exempt`.

### 6.3 TabMmini: native 19-column route

TabMmini imports the same 19-column semantic frame and does not use the
PlainMLP external-OHE artifact. Numeric values must remain finite `float32`.
When its adapter materializes indexed categorical tensors, their dtype must be
integer/`long`.

The TabMmini adapter builds fixed category-to-index maps from the authoritative
semantic schema, not from observed pilot, fold, full-Train, Validation, or Test
values. The maps remain identical across the pilot, all five folds, and full
Train; preserve schema positions for absent categories. They include `Exempt`
wherever allowed and exclude `1111`, `MISSING`, and an unknown-category bucket.
An unknown category fails rather than being silently remapped.

Integer indexing is the native container boundary, not project-level OHE. The
project must not apply OHE, `StandardScaler`, or `RobustScaler` on this route.
The indexed categories are passed once to the TabM package, whose native
categorical encoding is backend-owned; an adapter must not pre-encode and then
invoke that encoding a second time. The backend must not repeat raw-to-semantic
normalization. The complete locked model-specific policy is in the
[TabMmini v1 training runbook](../runbooks/tabmmini_training.md).

## 7. Dtype and container contract

| Boundary | Required container | Required dtype |
| --- | --- | --- |
| Shared semantic frame | pandas DataFrame, 19 ordered columns | finite numeric values; categorical/status meanings preserved |
| PlainMLP stored input | SciPy CSR, 62 columns | `float64` |
| PlainMLP model batch | dense tensor for the current batch only | `float64` |
| RealMLP native input | pandas DataFrame, 19 ordered columns | numeric `float32`; categorical pandas `category` |
| TabMmini native input | adapter-defined native container derived from 19 ordered columns | numeric `float32`; indexed categorical tensors integer/`long` |

An adapter must fail on a different column set/order, a non-finite numeric
value, an unsupported category, or a dtype/container that violates its route.

## 8. Preprocessing artifact contract

### 8.1 Common manifest requirements

Every persisted neural preprocessing artifact must record directly reviewable
fields for:

- the unique `run_id` linking preprocessing, fitted model, predictions, and
  evaluation outputs from the same execution;
- dataset, preprocessing version, route/profile, and split version;
- source partition and exact fit scope;
- ordered 15-field raw schema and ordered 19-column semantic schema;
- numeric medians, transformation settings, and `log1p` audit results;
- raw `1111` and normalized `Exempt` counts for the five named flags;
- fixed category lists and per-fold category-support outcomes, including
  `Exempt` support;
- route-specific output columns or category-to-index maps;
- output container, shape, dtype, and adapter boundary;
- row counts and ordered `record_id` alignment evidence;
- relevant library versions;
- model and runtime configuration fields used by the run.

The preprocessing version remains `dataset_b_neural_preprocessing_v1`. A
manifest must record observed artifact properties and directly readable
schema/configuration values, not opaque content identities and not merely a
copy of expected config values.

### 8.2 PlainMLP CV artifact

The fold-local export destination is run-scoped:

`data/preprocessing/dataset_b_v1/cv_neural_preprocessing_v1/<run_id>/`

When an export is produced, it must contain one manifest and, for each of five
folds:

- `plain_mlp_train.npz`;
- `plain_mlp_fold_validation.npz`;
- `rows.npz` for row-alignment evidence.

The manifest contract requires five fold-train matrices and five matching
fold-validation matrices, all CSR `float64`, all 62 columns, with 10 matrices
in total. Each fold entry records learned medians, the log audit, category
support, shape, schema, and row-alignment evidence. This export is for
cross-validation only; artifact existence and acceptance state are recorded
in [STATUS](../STATUS.md).

### 8.3 Native-route artifacts

RealMLP and TabMmini do not consume the PlainMLP matrix export. Their run or
preprocessing manifests must instead bind the native 19-column adapter to:

- the `run_id`, preprocessing version, and split version;
- the exact ordered semantic schema and dtype audit;
- semantic category lists, observed support, and adapter-fitted state;
- direct category-to-index maps where the route materializes them;
- backend-owned transformation and model/runtime configuration fields needed
  to reproduce the route;
- row and `record_id` alignment evidence.

A model-run manifest that records only model outputs or training controls does
not by itself satisfy this preprocessing artifact contract.

### 8.4 Legacy compatibility boundary

Previously fitted neural preprocessors, matrices, native frames, model states,
and predictions were created against the superseded feature and category
contracts. They are incompatible with the final retraining interface and must
not be relabeled, extended in place, warm-started, or consumed by a final run.
They remain historical evidence only.

## 9. Acceptance gates

A neural preprocessing artifact is acceptable only when all applicable checks
pass:

1. the 15 raw inputs and 19 semantic columns match the imported contracts;
2. the blank-row gate has passed and no `MISSING` model category exists;
3. raw `1111` counts equal semantic `Exempt` counts for the five named flags,
   no semantic `1111` remains, and no adapter/backend repeats the mapping;
4. fitted state comes only from fold-train or full Train, as appropriate;
5. `income_value` and `loan_amount_value` are non-negative before `log1p`, all
   numeric values are finite afterward, and the other two numeric values remain
   unchanged during `log1p`;
6. category/status meanings and the fixed semantic schema are preserved;
7. category support, including `Exempt`, is audited with the outcomes in §5;
8. PlainMLP output is CSR `float64` with exactly 62 columns, the four numeric
   value columns are scaled by fit-local `RobustScaler`, OHE columns are not
   scaled, and densification is batch-only;
9. RealMLP output is the native 19-column DataFrame with the required dtypes;
10. TabMmini consumes the native 19-column semantic frame and records its own
    mapping and container rather than borrowing undocumented RealMLP behavior;
11. the manifest stores the `run_id`, directly readable schema/configuration
    fields, actual library versions, and row-alignment evidence;
12. PlainMLP, RealMLP, and TabMmini preprocessing and model states are fitted
    afresh for the final retraining;
13. fold-local and full-Train artifacts are distinct and cannot be confused;
14. legacy artifacts remain historical and are not treated as compatible final
    retraining inputs.

## 10. Evidence sources

This contract was reconciled against:

- [Data contract](data_contract.md);
- [Classical preprocessing](preprocessing_classical.md);
- [Feature governance](feature_governance.md);
- [Split and evaluation](split_and_evaluation.md);
- [Documentation migration record](../archive/documentation_migration_2026-09.md),
  which identifies `Dataset_B_Engine_preprocesing.md` as a former source;
- `config/dataset_b/plain_mlp_v1.yaml`;
- `config/dataset_b/realmlp_v1.yaml`;
- `config/dataset_b/tabmmini_v1.yaml`;
- `src/dataset_b/models/neural/common/log_transformer.py`;
- `src/dataset_b/models/neural/plain_mlp/preprocessing.py`;
- `src/dataset_b/workflows/export_neural_cv.py`;
- `src/dataset_b/models/neural/realmlp/preprocessing.py`;
- `src/dataset_b/models/neural/tabmmini/preprocessing.py`.

This document records the approved final neural preprocessing contract.
PlainMLP, RealMLP, and TabMmini implementation and execution status is
evidenced by their frozen final runs and the authoritative
[STATUS](../STATUS.md) register. Historical sources listed here do not by
themselves establish final retraining completion, model quality, or Test
performance.
