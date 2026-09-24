# Dataset B — Engine preprocessing contract for six models

## 1. Purpose and status

This document records the preprocessing design for the six Dataset B models:

- classical pipeline: Logistic Regression (LR), Random Forest (RF), and XGBoost;
- neural pipeline: plain MLP, RealMLP, and TabMmini.

It is a technical decision register and implementation-readiness contract. As of 23/09/2026, PMLP-00 has implemented and verified the plain-MLP environment, preprocessing profile, five Train-only fold artifacts, and estimator wrapper. No plain-MLP training, Validation prediction, threshold selection, or Test evaluation has been run.

The current classical implementation remains governed by `dataset_b_preprocessing_v2`. Neural preprocessing uses the separate version `dataset_b_neural_preprocessing_v1` and must never overwrite, mutate, or reinterpret the classical artifacts. The plain-MLP route is implemented independently of the still-open RealMLP and TabMmini adapter decisions.

## 2. Shared data boundaries

All six models must preserve the same upstream Dataset B contract.

### 2.1 Model input and audit boundary

`X` contains exactly these 12 raw predictors, in this order:

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

`record_id`, `action_taken`, `y_approved`, `lei`, `county_code`, `applicant_race-1` through `applicant_race-5`, `applicant_sex`, and `applicant_age` are not predictors. In particular, `action_taken` is the source of the target and is forbidden from `X`.

### 2.2 Split and fitting boundary

- Use the existing `split_v1`; do not split records again.
- Cross-validation is five-fold stratified CV on Train only, with shuffle enabled and random state 42.
- Every learned preprocessing state is fitted independently on fold-train and applied to that fold's validation rows.
- After model selection, create a new pipeline and fit it on full Train only.
- Validation is used for threshold selection; Test is reserved for final evaluation after preprocessing, model, and threshold are locked.
- Validation or Test must not contribute medians, category decisions, scaling statistics, or any other fitted state.

### 2.3 Raw parser semantics

The existing parser policy is shared by both pipelines:

- null, empty, whitespace-only, and textual `nan` values become `MISSING`;
- literal `NA`, literal `Exempt`, and actual `MISSING` remain distinct states;
- code `1111` remains a valid category in the three exemption-coded categorical fields;
- negative `income` is mapped to `0`, while observed `income = 0` remains valid;
- extreme observed loan-to-value values are retained;
- invalid or out-of-domain tokens fail validation rather than being silently changed to `MISSING`;
- no row is deleted and no value is clipped or winsorized by preprocessing.

The special-state distinction is semantic, not cosmetic. `NA`, `Exempt`, and `MISSING` must not be merged merely because all three lack a usable numeric value.

## 3. Pipeline A — LR, RF, and XGBoost

### 3.1 Current implemented route

The current classical contract is `dataset_b_preprocessing_v2`:

```text
12 raw predictors
    -> deterministic raw parser
    -> four numeric two-channel fields
    -> fixed-schema one-hot encoding
    -> LR: RobustScaler on four numeric value columns only
       RF/XGBoost: no scaling
    -> CSR matrix with 66 columns
```

The four numeric fields are `income`, `loan_amount`, `loan_to_value_ratio`, and `loan_term`. Each becomes:

- `<feature>_value`: an observed numeric value or a fold-train median placeholder;
- `<feature>_status`: the applicable state among `OBSERVED`, `NA`, `Exempt`, and `MISSING`.

`OBSERVED` is removed as the reference dummy for numeric status. The other permitted status dummies are retained. The remaining eight raw categorical fields use full fixed-schema OHE. The current output contract is four numeric value columns followed by 62 encoded columns, for 66 columns total.

### 3.2 Model-specific handling

| Model | Numeric value handling | Categorical handling | Matrix contract |
| --- | --- | --- | --- |
| LR | Fold-local median placeholder, then `RobustScaler` on the four numeric value columns | Fixed-schema OHE; numeric status preserved; `OBSERVED` is reference | Existing LR CSR artifact, 66 columns |
| RF | Fold-local median placeholder; no scaling | Same fixed-schema OHE contract | Existing shared tree CSR artifact, 66 columns |
| XGBoost | Fold-local median placeholder; no scaling | Same fixed-schema OHE contract | Existing shared tree CSR artifact, 66 columns |

The classical pipeline is frozen for reproducibility. The neural decision to use `log1p` must not be back-ported into LR, RF, or XGBoost without a separate approved experiment and a new version.

## 4. Pipeline B — neural semantic layer

The neural family shares one semantic preprocessing layer before any engine-specific adapter:

```text
12 raw predictors
    -> existing deterministic raw parser
    -> four numeric two-channel fields
    -> fold-train median placeholder in value channels
    -> log1p on all four numeric value channels
    -> preserve status and categorical meaning
    -> engine-specific adapter
```

### 4.1 Numeric transformation

For each of the four numeric fields:

1. Determine the status from the parsed raw value.
2. Fit the median using observed numeric values from the current fold-train only.
3. Put the observed value or fold-train median placeholder into the value channel.
4. Apply `log1p` to the value channel.
5. Leave the status channel unchanged.

No `RobustScaler` is used in the neural pipeline. The log transform applies to all four numeric value channels, not to OHE columns or categorical codes.

The median must be computed in the original numeric scale and `log1p` applied afterward. This makes observed and placeholder values follow the same deterministic transformation while the status channel continues to identify why a placeholder was used.

### 4.2 Mandatory log audit

Task 00 must fail unless every fold confirms all of the following:

- no numeric value is below zero immediately before `log1p`;
- no NaN or infinity exists after `log1p`;
- every median came only from the matching fold-train observed values;
- numeric status values are unchanged by imputation and log transformation;
- row order and `record_id` alignment are unchanged;
- repeated transformation of identical input with identical fitted state is deterministic;
- Validation and Test did not participate in fitting.

## 5. Categorical representation and imbalance policy

### 5.1 Meaning preservation

The following rules apply to all neural routes:

- preserve `NA`, `Exempt`, `MISSING`, and valid code `1111` as distinct categories;
- do not delete a category because its frequency is low or because another category exceeds 70%;
- do not merge rare categories automatically;
- do not resample rows merely to balance predictor categories;
- do not use target information to consolidate categories;
- keep a fixed category schema and deterministic column or index ordering.

Predictor imbalance is not target class imbalance. A dominant categorical level is therefore handled through model capacity control, regularization, and validation evidence—not by silently changing the data semantics.

### 5.2 Absent-category gate

For every category in every CV fold:

| Fold condition | Required action |
| --- | --- |
| Support is greater than zero in fold-train | Pass and record the support |
| Absent from both fold-train and fold-validation | Keep its schema position and emit a warning |
| Absent from fold-train but present in fold-validation | Fail the gate and require an explicit decision |

The last case must not be repaired through an automatic merge. For external OHE it can expose a column whose model weight was never learned; for a native categorical adapter it can expose an unseen category/index. Both require an explicit, engine-aware policy.

## 6. Neural engine adapters

The semantic layer is shared, but physical representation may differ by model. “Same data meaning” does not require “same tensor layout.”

All three neural routes must still be exposed through the project's common scikit-learn-compatible wrapper contract. The wrapper must support independent cloning and fitted state per fold, `fit`, `predict_proba`, `get_params`, and `set_params`; publish `classes_` in `[0, 1]` order; and make `predict_proba(X)[:, 1]` represent `P(y_approved = 1)`. The adapter may own or delegate tensor conversion, batching, device, and engine-internal preprocessing, but it must record those choices and must not create a new split, read holdouts during fitting, or reuse fitted transformers, models, or optimizers across folds.

### 6.1 plain MLP — locked route

plain MLP uses the external fixed-schema OHE route:

```text
four log1p numeric value columns
    + fixed-schema OHE columns, including retained special states
    -> CSR float32 preprocessing artifact
    -> dense float32 conversion at the model boundary, preferably by batch
```

Rules:

- do not scale or standardize the OHE columns;
- store the preprocessing matrix as CSR `float32`;
- convert to dense `float32` only at the model/data-loader boundary;
- prefer batch conversion when supported, rather than materializing an unnecessary full dense copy;
- record exact input width, column order, dtype, and sparse-to-dense boundary in the run artifact.

If the neural category schema is identical to the existing classical schema, the expected width is 66. Task 00 must derive and verify that width rather than assuming it solely from the classical artifact.

### 6.2 RealMLP — route still open

RealMLP can receive either a common external numeric/OHE matrix or typed continuous and categorical inputs handled by its own adapter/preprocessing behavior. The project has not yet locked which route will be used.

Before implementation, choose exactly one:

- **External common-OHE route:** use the same external representation as plain MLP and explicitly disable or bypass overlapping internal preprocessing. The comparison then concerns models on a common OHE representation; it does not measure RealMLP's native categorical capability.
- **Native adapter route:** pass four transformed continuous values plus schema-controlled categorical inputs, while configuring the engine so that project semantics for `NA`, `Exempt`, `MISSING`, and `1111` remain intact. Record any internal transformation performed by the engine.

Do not apply external OHE and then unknowingly apply a second categorical preprocessing stage inside RealMLP.

### 6.3 TabMmini — route still open

TabM-style models normally separate numerical and categorical tensors. The project must choose whether TabMmini is constrained to the common external OHE representation or uses a native categorical adapter.

Before implementation, choose exactly one:

- **External common-OHE route:** feed the common `float32` OHE representation only if the selected TabMmini implementation explicitly supports it and internal categorical expansion is disabled or not invoked. Interpret results as a common-OHE comparison.
- **Native adapter route:** provide numerical input as `float32` and categorical indices as integer/`long`; derive deterministic token-to-index maps and cardinalities from the versioned fixed schema, and compute category-support diagnostics from fold-train only.

The native adapter route is the technical recommendation because it preserves the engine's intended input structure, but it remains a recommendation—not an approved implementation decision.

### 6.4 Open route decision

The route for RealMLP and TabMmini remains **OPEN** until the project explicitly chooses between:

1. a common external-OHE experiment for all three neural models; or
2. a shared semantic layer with native engine adapters for RealMLP and TabMmini.

The choice must be recorded before a RealMLP or TabMmini task generates artifacts. It does not block the plain-MLP-only PMLP task chain.

## 7. Dtype and sparse/dense boundary

| Stage | Required dtype/format |
| --- | --- |
| External neural OHE artifact | CSR `float32` |
| plain MLP input | Dense `float32`, converted at model boundary or per batch |
| RealMLP common-OHE route | `float32`; exact accepted container verified against selected backend |
| RealMLP native route | Dtypes required by the selected adapter, recorded explicitly |
| TabMmini native numerical input | `float32` |
| TabMmini native categorical input | Integer/`long` indices |
| Stored probabilities and metric calculations | May use `float64` for numerical stability; this is separate from model input dtype |

No backend may silently choose its own conversion rule. The adapter owns conversion, and the run manifest must record the actual format and dtype received by the model.

### 7.1. Locked plain-MLP environment

PMLP-00 uses the separate `dap391m-neural` environment with Python `3.12.13`, the stable CPU build of PyTorch release `2.14.0`, and no `torchvision` or `torchaudio`. The CPU wheel may report a local build suffix such as `2.14.0+cpu`; runtime evidence must therefore record the full `torch.__version__`, confirm release base `2.14.0`, confirm `torch.version.cuda is None`, and confirm `torch.cuda.is_available() is False`.

The environment installs the repository's base requirements first and then the CPU-only overlay in `requirements-neural.txt`. PMLP-00A performs import/config smoke checks only; it does not run preprocessing or train a model.

## 8. Regularization ownership

The shared neural preprocessing contract owns parser checks, fold boundaries, fitted preprocessing state, matrix/tensor schema, dtype, category support diagnostics, and deterministic outputs. PMLP-00A additionally records a runnable plain-MLP starting configuration, but does not promote it to a project default, final configuration, or CV-selected configuration.

The following belong to each model's configuration and CV evidence:

- weight decay or L2;
- dropout;
- layer width and depth;
- learning rate;
- batch size;
- epoch budget;
- early-stopping rule and patience.

When a future config needs runnable starter values, commonly used values may be written only with all of these metadata flags:

```yaml
status: provisional_starting_configuration
is_project_default: false
is_final_configuration: false
selected_by_train_cv: false
```

The provisional `plain_mlp_v1` starting point uses hidden layers `[128, 64]`, ReLU, one output logit, `BCEWithLogitsLoss`, AdamW, learning rate `1e-3`, weight decay `1e-4`, dropout `0.1`, batch size `256`, `100` epochs, no early stopping, no class/sample weighting, random state `42`, and CPU deterministic algorithms. These values are **not** project defaults, are **not** final, are **not** CV-selected, and must not be transferred automatically to RealMLP or TabMmini.

Categorical imbalance does not authorize adding class weights, sample weights, or resampling. Those mechanisms concern target handling and require a separate model-protocol decision.

## 9. Version and artifact contract

### 9.1 Classical artifacts

- Preserve `dataset_b_preprocessing_v2` and its current LR/tree artifacts unchanged.
- Do not reuse a fitted classical preprocessor as a neural preprocessor.
- Do not relabel existing artifacts as log-transformed or `float32`.

### 9.2 Neural artifacts

The plain-MLP artifact contract is locked as follows:

- preprocessing version: `dataset_b_neural_preprocessing_v1`;
- CV root: `data/preprocessing/dataset_b_v1/cv_neural_preprocessing_v1/`;
- manifest: `manifest.json`;
- model artifact root: `artifacts/dataset_b/plain_mlp_v1/`;
- exact schema/order, CSR `float32`, batch-owned dense `float32` conversion, fold-level category support, fit scope, parser/split/feature/library versions, `record_id` alignment, and hashes are recorded in the manifest.

Neural CV artifacts and full-Train artifacts must be separate. A fold-fitted preprocessor must not be used for Validation/Test, and a full-Train-fitted preprocessor must not be described as a CV artifact.

## 10. Six-model preprocessing matrix

| Model | Pipeline | Numeric value transform | Categorical route | Input boundary | Decision status |
| --- | --- | --- | --- | --- | --- |
| LR | Classical v2 | Median placeholder + RobustScaler | Fixed-schema OHE | CSR, current implementation dtype | Implemented contract; preserve |
| RF | Classical v2 | Median placeholder, no scaling | Fixed-schema OHE | CSR, current implementation dtype | Implemented contract; preserve |
| XGBoost | Classical v2 | Median placeholder, no scaling | Fixed-schema OHE | CSR, current implementation dtype | Implemented contract; preserve |
| plain MLP | Separate neural version | Median placeholder then `log1p`; no RobustScaler | Fixed-schema OHE | CSR `float32` artifact -> dense `float32` at boundary | PMLP-00 implemented and verified; training pending |
| RealMLP | Separate neural version | Same neural semantic transform | External OHE or native adapter | Must match selected route | **OPEN** |
| TabMmini | Separate neural version | Same neural semantic transform | External OHE or native categorical indices | `float32` numeric; categorical `long` if native | **OPEN** |

“Current implementation dtype” in the classical rows refers to the existing v2 implementation, which emits CSR `float64`. The `float32` decision applies to newly created neural artifacts only.

## 11. Task 00 acceptance gates

PMLP-00 is independent of the two open engine routes and may be implemented for plain MLP only. RealMLP and TabMmini remain outside its write scope. PMLP-00 must provide evidence for all applicable shared and external-OHE gates below.

### 11.1 Shared gates

- [x] Reads only the existing fixed Train partition for CV preprocessing.
- [x] Uses exactly the 12 predictors and keeps audit/target columns outside `X`.
- [x] Preserves parser semantics and the distinction among `NA`, `Exempt`, `MISSING`, and `1111`.
- [x] Fits every median and category state on fold-train only.
- [x] Passes all log-transform audit checks.
- [x] Records per-fold category counts and absent-category outcomes.
- [x] Preserves row order and exact `record_id` alignment.
- [x] Produces deterministic schema, ordering, dtype, and values.
- [x] Does not overwrite any classical artifact.

### 11.2 External-OHE gates

- [x] Keeps fixed category positions even when a category is absent from a fold.
- [x] Verifies output width from the neural schema; does not assume 66 without checking.
- [x] Emits CSR `float32`.
- [x] Confirms OHE columns were not scaled.
- [x] Records the exact sparse-to-dense boundary for plain MLP.

### 11.3 Native-adapter gates

- [ ] Records deterministic token-to-index maps and cardinalities.
- [ ] Defines the unseen-category behavior explicitly.
- [ ] Confirms categorical indices and numerical tensors have the required dtypes.
- [ ] Audits and records engine-internal preprocessing to prevent duplicate transforms.
- [ ] Demonstrates that special-state semantics survived the adapter unchanged.

## 12. Locked decisions, open decisions, and non-claims

### Locked and implemented for plain MLP

- Create a separate neural preprocessing version; never overwrite classical v2.
- Reuse the parser and two-channel semantics.
- Apply fold-train median placeholders and then `log1p` to all four numeric value channels.
- Do not use `RobustScaler` in the neural route.
- Preserve categorical semantics and fixed schema; do not merge or rebalance predictor categories automatically.
- Use external fixed-schema OHE for plain MLP.
- Use CSR `float32` artifacts and controlled dense `float32` conversion for plain MLP.
- Apply the absent-category and log-audit gates defined above.
- Keep regularization in model configuration and label starter values as provisional, not defaults.

### Still open outside the plain-MLP PMLP chain

- RealMLP: common external OHE versus native categorical adapter.
- TabMmini: common external OHE versus native categorical adapter.
- Engine-specific final architecture, optimization, regularization, and early-stopping values.

### Verified implementation evidence and non-claims

PMLP-00 evidence now establishes that:

- the separate `dap391m-neural` environment imports Python `3.12.13` and PyTorch `2.14.0+cpu`;
- five Train-only neural folds, 10 CSR `float32` matrices, five row files, and a manifest exist under the locked CV artifact root;
- all 15 payload hashes, 205,310 OOF row positions, record IDs, labels, 66-column width, finite values, and zero absent-train/present-validation gate failures were independently verified;
- the sklearn-compatible wrapper passed clone, probability, deterministic CPU, batch-densification, fresh-fit, and checkpoint round-trip tests.

This evidence does **not** claim that the provisional configuration is a project default, final, or CV-selected configuration; that RealMLP or TabMmini has an approved input route; or that any plain-MLP model has been trained on the real Dataset B folds, evaluated, or compared with the classical models.

## 13. Sources of truth

Project sources inspected for this contract:

- `config/dataset_b/feature_set_v1.yaml`
- `config/dataset_b/split_v1.yaml`
- `config/dataset_b/model_protocol_v1.yaml`
- `docs/Dataset_B_Pipeline.md`
- `docs/Dataset_B_v1_preprocessing.md`
- `src/dataset_b/preparation/dataset_view.py`
- `src/dataset_b/preprocessing/raw_parser.py`
- `src/dataset_b/preprocessing/median_imputer.py`
- `src/dataset_b/preprocessing/numeric_two_channel.py`
- `src/dataset_b/preprocessing/categorical_encoder.py`
- `src/dataset_b/preprocessing/profiles.py`
- `src/dataset_b/preprocessing/export_cv.py`

For RealMLP and TabMmini implementation, the exact installed backend version and its official interface must be verified again before locking either adapter route. Those future checks do not alter the implemented plain-MLP contract.
