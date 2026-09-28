# Dataset B classical preprocessing contract

## 1. Purpose and ownership

This contract defines the classical preprocessing interface for Logistic
Regression, Random Forest, and XGBoost under
`dataset_b_preprocessing_v2`. It owns:

- the ordered 12-column raw input interface;
- deterministic parser semantics;
- four numeric two-channel representations;
- the fixed 66-column encoded schema;
- Logistic Regression scaling;
- Random Forest and XGBoost preprocessing;
- CV preprocessing artifact schema, format, and dtype.

Split, threshold, Test, bootstrap, and RQ2/RQ3 boundaries belong to
[Split and evaluation](split_and_evaluation.md). Feature inclusion decisions
belong to [Feature governance](feature_governance.md). Other model-family
preprocessing and model execution status are outside this contract.

## 2. Versioned input contract

The classical pipeline consumes the following 12 raw fields in this exact
order from `temporary_training_lock_v0_2`:

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

The parser rejects a different column set or order. `record_id`,
`action_taken`, `y_approved`, the nine audit columns, and any unresolved
candidate are outside `X`. The use of these 12 inputs is a temporary RQ1
training contract; it is not a final feature lock.

## 3. Fit and transform boundary

The preprocessing boundary is identical to the split contract:

| Stage | Fitted population | Permitted application |
| --- | --- | --- |
| CV fold | Fold-train only | Matching fold-validation only |
| Final RQ1 fit | Full Train only | Validation, then Test after threshold lock |

Every fold receives a fresh pipeline. Medians and Logistic Regression scaling
statistics are learned only from the matching fold-train. After configuration
selection, another fresh pipeline is fitted on full Train. Validation and Test
must not contribute medians, category decisions, scaling statistics, or any
other fitted state.

The exported CV matrices are fold-specific, `cross_validation_only` artifacts.
They are not fitted preprocessing objects and cannot be used as the full-Train
preprocessor for Validation or Test.

## 4. Deterministic raw parser

`DatasetBRawParser` validates the exact 12-column interface without fitting
data-dependent state.

### 4.1 Missing and special tokens

- null, empty, whitespace-only, and textual `nan` values become `MISSING`;
- literal `NA`, literal `Exempt`, and actual `MISSING` remain distinct states
  where the field schema permits them;
- code `1111` remains a valid category for `reverse_mortgage`,
  `open-end_line_of_credit`, and `business_or_commercial_purpose`;
- `income = 1111` is parsed as a number, not an exemption category;
- an invalid or out-of-domain token raises an error instead of becoming
  `MISSING`.

### 4.2 Numeric rules

- negative `income` is mapped to numeric `0`;
- observed `income = 0` remains a valid observed value;
- `loan_amount`, numeric `loan_to_value_ratio`, and numeric `loan_term` must be
  positive;
- numeric `loan_term` must be integer-valued;
- observed LTV values above 1,000 percent are retained and counted in parser
  diagnostics;
- preprocessing does not delete rows, clip or winsorize values, or apply a log
  transform.

### 4.3 Categorical domains

The fixed parser domains are:

| Field | Permitted non-missing tokens |
| --- | --- |
| `loan_type` | `1`, `2`, `3`, `4` |
| `loan_purpose` | `1`, `2`, `31`, `32`, `4`, `5` |
| `lien_status` | `1`, `2` |
| Three exemption-coded flags | `1`, `2`, `1111` |
| `occupancy_type` | `1`, `2`, `3` |
| `debt_to_income_ratio` | `<20%`, `20%-<30%`, `30%-<36%`, integer bands `36` through `49`, `50%-60%`, `>60%`, `NA`, `Exempt` |

`MISSING` is added by the parser to every categorical schema. Unknown categories
fail validation during encoding; they are not represented as an all-zero row.

## 5. Numeric two-channel representation

The four numeric fields are:

- `income`;
- `loan_amount`;
- `loan_to_value_ratio`;
- `loan_term`.

Each field becomes a value and a status channel:

| Raw field | Value channel | Status states |
| --- | --- | --- |
| `income` | `income_value` | `OBSERVED`, `NA`, `MISSING` |
| `loan_amount` | `loan_amount_value` | `OBSERVED`, `MISSING` |
| `loan_to_value_ratio` | `loan_to_value_ratio_value` | `OBSERVED`, `NA`, `Exempt`, `MISSING` |
| `loan_term` | `loan_term_value` | `OBSERVED`, `NA`, `Exempt`, `MISSING` |

For an observed numeric token, the value channel retains its parsed value. For
the field-specific nonnumeric states listed in the table above, the value
channel receives the median learned from observed numeric values in the fitted
fold-train or full Train. The status channel preserves the original reason for
the placeholder.

The median is a model-ready placeholder, not a claim that the raw value was
observed. `NA`, `Exempt`, and `MISSING` must not be merged.

## 6. Fixed-schema one-hot encoding

The two-channel step produces 16 columns: four numeric value channels, four
numeric status channels, and the eight raw categorical fields. The encoder then
creates a deterministic classical representation:

1. retain the four numeric value channels first;
2. encode every schema-defined categorical/status level with fixed positions;
3. remove the four `OBSERVED` status dummies as reference categories;
4. retain valid `NA`, `Exempt`, `MISSING`, `1111`, and not-applicable levels;
5. emit 62 encoded categorical/status columns after the four numeric values.

The result is exactly 66 columns in a stable order. The eight raw categorical
fields use full one-hot encoding. A category keeps its schema position even if
it is absent from one fold-train sample.

The ordered column names, raw-field sources, and encoded categories are stored
in the preprocessing manifest as `output_columns`, `output_raw_sources`, and
`output_categories`.

## 7. Model-specific classical profiles

| Model family | Profile | Numeric handling | Encoded handling | Output |
| --- | --- | --- | --- | --- |
| Logistic Regression | `lr` | Fold-local median placeholder, then `RobustScaler` on the first four numeric value columns only | OHE columns pass through unchanged | CSR, 66 columns |
| Random Forest | `rf` | Fold-local median placeholder; no scaling | Fixed-schema OHE | CSR, 66 columns |
| XGBoost | `xgb` | Fold-local median placeholder; no scaling | Same tree representation as RF | CSR, 66 columns |

RF and XGBoost share the exported `tree` matrices because their preprocessing
representation is identical. They do not share fitted model state. Scaling is
never applied to OHE columns.

The classical route does not apply a logarithmic transform. A changed numeric
representation requires a new approved experiment and a new
preprocessing version; it must not rewrite `dataset_b_preprocessing_v2`.

## 8. Classical dtype and matrix contract

`DatasetBOneHotEncoder` defaults to `float64` for classical backward
compatibility. Every classical v2 export must satisfy this output contract:

- all LR and tree matrices are SciPy CSR;
- all matrix values use `float64`;
- all stored nonzero values are finite;
- every matrix has 66 columns;
- target arrays use `int8`;
- row-position arrays use `int64`;
- stored `record_id` arrays are Unicode strings.

`float32` requirements from other preprocessing profiles do not apply to these
classical v2 artifacts. Existing classical artifacts must not be relabeled as
`float32` or as artifacts from another profile.

## 9. CV artifact schema

The current CV root is:

`data/preprocessing/dataset_b_v1/cv_preprocessing_v2/`

Its manifest records:

| Field | Contract value or role |
| --- | --- |
| `preprocessing_version` | `dataset_b_preprocessing_v2` |
| `split_version` | `split_v1` |
| `feature_version` | `temporary_training_lock_v0_2` |
| `source_partition` | `train` |
| `fit_scope` | `fold_train_only` |
| `use` | `cross_validation_only` |
| `encoded_feature_count` | `66` |
| `status_reference_categories` | `OBSERVED` for all four numeric statuses |
| `tree_profile_shared_by` | `rf`, `xgb` |
| `numeric_scaling` | LR uses `RobustScaler`; RF/XGBoost use none |
| `sample_weights_exported` | `false` |
| `library_versions` | NumPy, pandas, scikit-learn, SciPy, and XGBoost versions used for export |

Each `fold_01` through `fold_05` contains:

- `lr_train.npz` with shape `164248 × 66`;
- `lr_fold_validation.npz` with shape `41062 × 66`;
- `tree_train.npz` with shape `164248 × 66`;
- `tree_fold_validation.npz` with shape `41062 × 66`;
- `rows.npz` containing train/fold-validation positions, ordered
  `record_id`, and `y_approved` arrays.

The five fold-validation row sets together must cover all 205,310 Train rows
exactly once. Matrix rows, row-position arrays, identifiers, and labels must
remain aligned.

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
- write model outputs outside the preprocessing artifact root;
- preserve existing v2 files rather than overwriting them.

For raw-input CV or the full-Train stage, create a fresh classical pipeline with
the required profile and fit it within the boundary defined in Section 3.

## 11. Acceptance checks

A classical preprocessing artifact passes only when:

1. the raw input has exactly the 12 fields in the contract order;
2. target, identifier, and audit fields are absent from `X`;
3. parser rules preserve valid special states and reject invalid tokens;
4. medians and scaling state come only from the fitted Train scope;
5. all profiles produce the same ordered 66-column semantic schema;
6. LR scales only the first four numeric values;
7. RF/XGBoost do not scale;
8. matrices are finite CSR `float64` with exact row and column counts;
9. row positions, identifiers, labels, and matrices remain aligned;
10. the manifest links feature, split, preprocessing, and library versions;
11. no existing classical artifact is overwritten or reinterpreted;
12. no Validation or Test data contributed fitted state.

## 12. Prohibited changes

- Do not add a thirteenth raw input through preprocessing configuration.
- Do not move target or audit metadata into `X`.
- Do not merge `NA`, `Exempt`, `MISSING`, or valid `1111` states.
- Do not silently coerce an invalid token to missing.
- Do not drop rows, clip outliers, or add a log transform in classical v2.
- Do not scale OHE columns for Logistic Regression.
- Do not fit on Validation or Test.
- Do not reuse fold-fitted state for full-Train or holdout inference.
- Do not use this contract to report saved LR, RF, or XGBoost results.
- Do not reuse this classical external-OHE artifact outside the three classical
  model families named by this contract.

## 13. Source traceability

This contract was reconstructed from:

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

The contract records the implemented classical representation and artifact
interface. It intentionally omits notebook instructions, model run results,
Test status, and preprocessing routes owned by other model families.
