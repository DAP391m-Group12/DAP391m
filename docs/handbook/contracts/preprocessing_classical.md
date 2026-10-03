# Dataset B classical preprocessing contract

## 1. Purpose and ownership

This contract defines the classical preprocessing interface for Logistic
Regression, Random Forest, and XGBoost under
`dataset_b_preprocessing_v2`. It owns:

- the ordered 15-column raw input interface;
- deterministic parser semantics;
- four numeric two-channel representations;
- the fixed 62-column encoded schema;
- Logistic Regression selective `log1p` and scaling;
- Random Forest and XGBoost preprocessing;
- CV preprocessing artifact schema, format, and dtype.

Split, threshold, Test, bootstrap, and RQ2/RQ3 boundaries belong to
[Split and evaluation](split_and_evaluation.md). Feature inclusion decisions
belong to [Feature governance](feature_governance.md). Other model-family
preprocessing and model execution status are outside this contract.

## 2. Final input contract

The classical pipeline consumes the following 15 raw fields in this exact
order from the final `FINAL_MODEL_FEATURES` contract:

1. `loan_type`
2. `loan_purpose`
3. `lien_status`
4. `reverse_mortgage`
5. `open-end_line_of_credit`
6. `business_or_commercial_purpose`
7. `income`
8. `debt_to_income_ratio`
9. `loan_amount`
10. `loan_to_value_ratio`
11. `loan_term`
12. `occupancy_type`
13. `preapproval`
14. `interest_only_payment`
15. `balloon_payment`

The parser rejects a different column set or order. `record_id`,
`action_taken`, `y_approved`, the nine audit columns, and excluded candidates
are outside `X`. The historical 12-input `temporary_training_lock_v0_2`
interface remains lineage for its existing artifacts only and must not be used
for final retraining.

## 3. Fit and transform boundary

The preprocessing boundary is identical to the split contract:

| Stage | Fitted population | Permitted application |
| --- | --- | --- |
| CV fold | Fold-train only | Matching fold-validation only |
| Final RQ1 fit | Full Train only | Validation, then Test after threshold lock |

Every fold receives a fresh pipeline. Medians and Logistic Regression
`RobustScaler` statistics are learned only from the matching fold-train. The
approved `log1p` operation is deterministic and has no fitted state. After
configuration selection, another fresh pipeline is fitted on full Train.
Validation and Test must not contribute medians, category decisions, scaling
statistics, or any other fitted state.

The final input-quality PASS/FAIL gate in [Data contract](data_contract.md)
runs before fitted preprocessing. Only retained rows may enter a fold or the
full-Train fit. Its blank-row and exemption-normalization audit must preserve
the retained row order and `record_id` alignment.

The exported CV matrices are fold-specific, `cross_validation_only` artifacts.
They are not fitted preprocessing objects and cannot be used as the full-Train
preprocessor for Validation or Test.

## 4. Deterministic raw parser

`DatasetBRawParser` validates the exact 15-column interface without fitting
data-dependent state.

### 4.1 Missing and special tokens

- null, empty, whitespace-only, and case-insensitive textual `nan` values are
  rejected by the input-quality gate before parsing; they do not become a
  `MISSING` model category;
- literal `NA` and literal `Exempt` remain distinct states where the field
  schema permits them;
- raw code `1111` is valid only for `reverse_mortgage`,
  `open-end_line_of_credit`, `business_or_commercial_purpose`,
  `interest_only_payment`, and `balloon_payment`; after raw validation it is
  normalized to the semantic category `Exempt` before encoding;
- literal `Exempt` is not a valid raw substitute for `1111` in those five
  exemption-coded flags;
- `income = 1111` is parsed as a number, not an exemption category;
- an invalid or out-of-domain token raises an error instead of being coerced.

For each of the five flags, the raw `1111` count must equal the normalized
`Exempt` count. Normalization must preserve retained row count and order, and
no semantic/model-facing `1111` may remain.

### 4.2 Numeric rules

- negative `income` is mapped to numeric `0`;
- observed `income = 0` remains a valid observed value;
- `loan_amount`, numeric `loan_to_value_ratio`, and numeric `loan_term` must be
  positive;
- numeric `loan_term` must be integer-valued;
- observed LTV values above 1,000 percent are retained and counted in parser
  diagnostics;
- after the input-quality gate, preprocessing does not delete rows, clip, or
  winsorize values; the model-profile transform is applied later under §7.

### 4.3 Categorical domains

The fixed parser domains are:

| Field | Accepted raw tokens | Model-facing semantic tokens |
| --- | --- | --- |
| `loan_type` | `1`, `2`, `3`, `4` | Same |
| `loan_purpose` | `1`, `2`, `31`, `32`, `4`, `5` | Same |
| `lien_status` | `1`, `2` | Same |
| `reverse_mortgage`, `open-end_line_of_credit`, `business_or_commercial_purpose` | `1`, `2`, `1111` | `1`, `2`, `Exempt` |
| `occupancy_type` | `1`, `2`, `3` | Same |
| `preapproval` | `1`, `2` | Same |
| `interest_only_payment`, `balloon_payment` | `1`, `2`, `1111` | `1`, `2`, `Exempt` |
| `debt_to_income_ratio` | `<20%`, `20%-<30%`, `30%-<36%`, integer bands `36` through `49`, `50%-60%`, `>60%`, `NA`, `Exempt` | Same |

There is no `MISSING` category in the parser, semantic frame, or OHE schema.
Unknown categories fail validation during encoding; they are not represented as
an all-zero row.

## 5. Numeric two-channel representation

The four numeric fields are:

- `income`;
- `loan_amount`;
- `loan_to_value_ratio`;
- `loan_term`.

Each field becomes a value and a status channel:

| Raw field | Value channel | Status states |
| --- | --- | --- |
| `income` | `income_value` | `OBSERVED`, `NA` |
| `loan_amount` | `loan_amount_value` | `OBSERVED` |
| `loan_to_value_ratio` | `loan_to_value_ratio_value` | `OBSERVED`, `NA`, `Exempt` |
| `loan_term` | `loan_term_value` | `OBSERVED`, `NA`, `Exempt` |

For an observed numeric token, the value channel retains its parsed value. For
the field-specific nonnumeric states listed in the table above, the value
channel receives the median learned from observed numeric values in the fitted
fold-train or full Train. The status channel preserves the original reason for
the placeholder.

The median is a model-ready placeholder, not a claim that the raw value was
observed. `NA` and `Exempt` must not be merged. The semantic frame contains 19
columns: four numeric value channels, four numeric status channels, and the 11
raw categorical fields.

## 6. Fixed-schema one-hot encoding

The two-channel step produces 19 columns: four numeric value channels, four
numeric status channels, and the 11 raw categorical fields. The encoder then
creates a deterministic classical representation:

1. retain the four numeric value channels first;
2. encode every schema-defined categorical/status level with fixed positions;
3. remove the four `OBSERVED` status dummies as reference categories;
4. retain valid `NA`, `Exempt`, and not-applicable levels;
5. emit 58 encoded categorical/status columns after the four numeric values.

The result is exactly 62 columns in a stable order. The 11 raw categorical
fields contribute 53 full-OHE columns. The numeric status fields contribute
five non-reference columns after the four `OBSERVED` dummies are removed. A
category keeps its schema position even if it is absent from one fold-train
sample. `Exempt` is the only encoded exemption category for the five flags;
neither `1111` nor `MISSING` has a model-facing column.

The ordered column names, raw-field sources, and encoded categories are stored
in the preprocessing manifest as `output_columns`, `output_raw_sources`, and
`output_categories`.

## 7. Model-specific classical profiles

| Model family | Profile | Numeric handling | Encoded handling | Output |
| --- | --- | --- | --- | --- |
| Logistic Regression | `lr` | Fold-local median placeholder; `log1p` on `income_value` and `loan_amount_value` only; then `RobustScaler` on all four numeric value columns | OHE columns pass through unchanged | CSR, 62 columns |
| Random Forest | `rf` | Fold-local median placeholder; all four numeric values remain raw; no logarithmic transform or scaling | Fixed-schema OHE | CSR, 62 columns |
| XGBoost | `xgb` | Fold-local median placeholder; all four numeric values remain raw; no logarithmic transform or scaling | Same tree representation as RF | CSR, 62 columns |

RF and XGBoost share the exported `tree` matrices because their preprocessing
representation is identical. They do not share fitted model state.

For LR, the selective `log1p` step runs after parsing, status extraction, and
median placeholder insertion, and before `RobustScaler`. Negative income has
already been mapped to zero under §4.2. `loan_to_value_ratio_value` and
`loan_term_value` must not receive `log1p`. Scaling is never applied to OHE
columns.

This approved final update remains under `dataset_b_preprocessing_v2`; it does
not open another preprocessing version. A unique `run_id` distinguishes the
new retraining execution and links its directly stored schema and settings.

## 8. Classical dtype and matrix contract

`DatasetBOneHotEncoder` uses `float64`. Every final classical export must
satisfy this output contract:

- all LR and tree matrices are SciPy CSR;
- all matrix values use `float64`;
- all stored nonzero values are finite;
- every matrix has 62 columns;
- target arrays use `int8`;
- row-position arrays use `int64`;
- stored `record_id` arrays are Unicode strings.

`float32` requirements from other preprocessing profiles do not apply to these
classical artifacts. Historical 66-column classical artifacts must not be
relabeled as final 62-column artifacts or consumed by the final retraining run.

## 9. CV artifact schema

The final CV root is run-scoped:

`data/preprocessing/dataset_b_v1/cv_preprocessing_v2/<run_id>/`

Its manifest records:

| Field | Contract value or role |
| --- | --- |
| `run_id` | Unique identifier shared by preprocessing and downstream outputs from this execution |
| `preprocessing_version` | `dataset_b_preprocessing_v2` |
| `split_version` | `split_v1` |
| `feature_contract` | Ordered 15-field `FINAL_MODEL_FEATURES` |
| `source_partition` | `train` |
| `fit_scope` | `fold_train_only` |
| `use` | `cross_validation_only` |
| `semantic_feature_count` | `19` |
| `encoded_feature_count` | `62` |
| `status_reference_categories` | `OBSERVED` for all four numeric statuses |
| `categorical_normalization` | Raw `1111` to semantic `Exempt` for the five named flags only |
| `blank_row_audit` | Input, retained, and rejected counts plus rejected IDs and per-column blank counts |
| `normalization_audit` | Per-flag raw `1111` and normalized `Exempt` counts; semantic `1111` count must be zero |
| `tree_profile_shared_by` | `rf`, `xgb` |
| `numeric_transforms` | LR: `log1p` on income/loan amount and `RobustScaler` on four numeric values; RF/XGBoost: raw numeric values and no scaling |
| `schema_settings` | Ordered output columns, raw sources, categories, dtype, sparse format, and profile settings stored directly |
| `library_versions` | NumPy, pandas, scikit-learn, SciPy, and XGBoost versions used for export |

Each `fold_01` through `fold_05` contains:

- `lr_train.npz` with shape `164248 × 62`;
- `lr_fold_validation.npz` with shape `41062 × 62`;
- `tree_train.npz` with shape `164248 × 62`;
- `tree_fold_validation.npz` with shape `41062 × 62`;
- `rows.npz` containing train/fold-validation positions, ordered
  `record_id`, and `y_approved` arrays.

The five fold-validation row sets together must cover all 205,310 Train rows
exactly once. Matrix rows, row-position arrays, identifiers, and labels must
remain aligned.

The row counts above assume the approved Train membership passes the blank-row
gate without rejection. If rows are rejected, the manifest must record the
audited counts and the fold shapes must reconcile to the retained population;
the split and fold-assignment contract must be revalidated before fitting.
Historical files at the unscoped CV root remain lineage only and must not be
overwritten or used as final 62-column inputs.

## 10. Artifact use and preservation

The CV export stores transformed matrices, labels, row positions, and
identifiers. It does not store a fitted preprocessing pipeline. Median and
scaler summaries in the manifest are audit evidence and are insufficient to
reconstruct a fitted object for holdout inference.

Downstream consumers must:

- load `lr` matrices only for Logistic Regression;
- load `tree` matrices for Random Forest or XGBoost;
- avoid applying preprocessing a second time to a stored matrix;
- preserve the manifest's encoded order and row alignment;
- keep predictions linked to the corresponding `record_id`;
- use the same `run_id` for the matching preprocessing, model, predictions,
  and evaluation outputs;
- write model outputs outside the preprocessing artifact root;
- preserve historical files rather than overwriting or relabeling them.

For raw-input CV or the full-Train stage, create a fresh classical pipeline with
the required profile and fit it within the boundary defined in Section 3.

## 11. Acceptance checks

A classical preprocessing artifact passes only when:

1. the retained raw input has exactly the 15 fields in the contract order;
2. target, identifier, and audit fields are absent from `X`;
3. the blank-row gate runs first and no retained row contains a blank final
   input or a `MISSING` model category;
4. raw `1111` and normalized `Exempt` counts match for all five flags, no
   semantic `1111` remains, and retained row count and order are unchanged;
5. parser rules preserve valid `NA` and `Exempt` states and reject invalid
   tokens;
6. medians and scaling state come only from the fitted Train scope;
7. the semantic frame has exactly 19 ordered columns and every profile has the
   same ordered 62-column encoded schema;
8. LR applies `log1p` only to income and loan amount, then scales all four
   numeric value columns with fold-local or full-Train `RobustScaler` state;
9. RF/XGBoost use all four numeric values without logarithmic transformation
   or scaling;
10. matrices are finite CSR `float64` with exact row and column counts;
11. row positions, identifiers, labels, and matrices remain aligned;
12. the manifest stores the `run_id`, ordered schema, categories, profile
    settings, audits, split/preprocessing identifiers, and library versions;
13. no historical classical artifact is overwritten or reinterpreted;
14. no Validation or Test data contributed fitted state.

## 12. Prohibited changes

- Do not add an input outside the final ordered 15-feature contract.
- Do not move target or audit metadata into `X`.
- Do not create a `MISSING` parser, status, or OHE category.
- Do not retain `1111` as a model-facing category or merge `Exempt` with code
  `2`, `NA`, a blank, or another state.
- Do not silently coerce an invalid token to missing.
- Do not drop a row after it passes the input-quality gate or clip/winsorize
  numeric values.
- Do not apply `log1p` to LTV or loan term.
- Do not apply `log1p` or scaling to RF/XGBoost numeric values.
- Do not scale OHE columns for Logistic Regression.
- Do not fit on Validation or Test.
- Do not reuse fold-fitted state for full-Train or holdout inference.
- Keep the approved preprocessing identifier unchanged for this update.
- Do not reuse historical 66-column artifacts for final retraining.
- Do not use this contract to report saved LR, RF, or XGBoost results.
- Do not reuse this classical external-OHE artifact outside the three classical
  model families named by this contract.

## 13. Source traceability

This contract was reconstructed from:

- [Data contract](data_contract.md);
- [Feature governance](feature_governance.md);
- [Documentation migration record](../archive/documentation_migration_2026-09.md),
  which identifies `Dataset_B_v1_preprocessing.md` and the classical sections
  of `Dataset_B_Engine_preprocesing.md` as former sources;
- `config/dataset_b/feature_set_v1.yaml`;
- `config/dataset_b/split_v1.yaml`;
- `config/dataset_b/model_protocol_v1.yaml`;
- `src/dataset_b/preprocessing/raw_parser.py`;
- `src/dataset_b/preprocessing/median_imputer.py`;
- `src/dataset_b/preprocessing/numeric_two_channel.py`;
- `src/dataset_b/preprocessing/categorical_encoder.py`;
- `src/dataset_b/preprocessing/profiles.py`;
- `src/dataset_b/preprocessing/split_inputs.py`;
- `src/dataset_b/preprocessing/export_cv.py`;
- `data/preprocessing/dataset_b_v1/cv_preprocessing_v2/manifest.json`;
- the stored fold matrices and row artifacts under the same CV root.

This document records the approved final classical preprocessing contract. The
listed implementation and stored artifacts describe the historical baseline
until a separately authorized code task updates and validates them. It
intentionally omits notebook instructions, model run results, Test status, and
preprocessing routes owned by other model families.
