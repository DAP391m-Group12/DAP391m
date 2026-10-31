# Dataset B — Data contract

## 1. Purpose and ownership

This document is the canonical handbook contract for the stable Dataset B data
boundary used in the HMDA New York 2024 credit-approval study. It owns:

- the eligible cohort and target mapping;
- interpretation limits for the target;
- the complete role partition of the 99 raw HMDA columns;
- the technical `record_id` boundary;
- mandatory schema, lineage, data-quality, timing, and leakage gates.

This document does **not** own feature-selection results, an active or final
model feature set, preprocessing transformations, split/evaluation procedures,
or operational progress. Those concerns belong to their respective
feature-governance, preprocessing, evaluation, runbook, and status documents.

The 35 columns listed as predictor candidates form a candidate universe. Their
presence in this contract does not mean that all 35 enter a model, have passed
feature selection, or belong to a final feature set.

## 2. Study scope and interpretation boundary

Dataset B covers HMDA loan/application records reported for New York in 2024
and models the probability of an observed HMDA credit-approval outcome.

The target must not be interpreted as:

- probability of borrower default;
- a complete measure of creditworthiness;
- the cause of a denial;
- causal evidence of discrimination.

`reverse_mortgage`, `open-end_line_of_credit`, and
`business_or_commercial_purpose` must not be used to filter the cohort. They are
transaction characteristics inside the predictor-candidate universe.

## 3. Cohort and target contract

The raw target source is `action_taken`.

| Raw `action_taken` | Cohort membership | `y_approved` |
| ---: | --- | ---: |
| `1` | Included | `1` |
| `2` | Included | `1` |
| `3` | Included | `0` |
| Any other value | Excluded | Not created |

The required sequence is:

1. determine cohort membership from raw `action_taken`;
2. derive `y_approved` from the mapping above;
3. retain raw `action_taken` for traceability;
4. keep both `action_taken` and `y_approved` outside predictor matrix `X`.

The positive values `{1, 2}`, negative value `{3}`, and eligible values
`{1, 2, 3}` must be non-empty, mutually consistent, and exhaustive for the
eligible cohort.

## 4. Record identity and controlled view

`record_id` is a project-generated technical identifier, not one of the 99 raw
HMDA columns. The current builder derives it from the row ordinal in the source
CSV; it is not an upstream HMDA identifier. It must:

- be non-empty and unique within Dataset B;
- preserve a deterministic link to the source-row ordinal for the same ordered
  source file;
- accompany split membership, predictions, and evaluation artifacts;
- remain outside predictor matrix `X`;
- not be included in the `35 + 9 + 1 + 54 = 99` reconciliation.

The implemented view builder currently derives IDs as
`dataset_b-<source-row-number padded to nine digits>`, where the first data row
after the CSV header is source-row ordinal 1. The identifier therefore depends
on the source file's row order and is not a stable key if that order changes.
Changing the identifier rule or source order requires regenerating and
revalidating dependent splits and artifacts rather than silently rewriting
IDs.

The controlled Dataset B view preserves, in order:

```text
record_id
action_taken
y_approved
model-input columns selected by feature governance
the nine audit-only columns
```

This view order does not change the complete raw-column role inventory defined
below.

## 5. Complete 99-column reconciliation

The frozen raw CSV header contains 99 unique columns. Their roles are mutually
exclusive and collectively exhaustive:

| Raw-column role | Count | Data-contract meaning |
| --- | ---: | --- |
| Predictor candidates | 35 | Eligible for feature-governance review; not automatically model inputs |
| Audit/RQ3 metadata | 9 | Preserved outside the main-model `X` matrix |
| Target source | 1 | Used to determine cohort and derive `y_approved`; forbidden from `X` |
| Excluded from Dataset B | 54 | Not carried into the controlled Dataset B view |
| **Total** | **99** | **Complete raw-header reconciliation** |

The required arithmetic is:

```text
35 + 9 + 1 + 54 = 99
```

No raw column may belong to more than one role, remain unassigned, or be added
to the controlled view without an explicit contract change.

## 6. Target source — 1 column

- `action_taken`

`action_taken` must remain available for lineage checks but is forbidden from
all model inputs.

## 7. Audit and RQ3 metadata — 9 columns

- `lei`
- `county_code`
- `applicant_race-1`
- `applicant_race-2`
- `applicant_race-3`
- `applicant_race-4`
- `applicant_race-5`
- `applicant_sex`
- `applicant_age`

These fields support organization, geography, protected-group, fairness, and
RQ3 audits. They remain outside the main-model `X` matrix. Moving any of them
into a predictor role requires a separately approved study design and a versioned
contract change.

The five applicant-race slots must be preserved as raw, traceable fields. Their
presence in audit metadata does not turn them into five independent business
predictors. The approved RQ3 fairness contract uses `derived_race`, rather than
these five raw slots, to calculate race-group metrics. The raw slots remain
available only for traceability under the current RQ3 protocol.

## 8. Predictor-candidate universe — 35 columns

### 8.1 Business-priority candidates — 15 columns

Loan structure and purpose:

- `loan_type`
- `loan_purpose`
- `lien_status`
- `reverse_mortgage`
- `open-end_line_of_credit`
- `business_or_commercial_purpose`

Financial capacity and loan/collateral scale:

- `income`
- `debt_to_income_ratio`
- `loan_amount`
- `property_value`
- `loan_to_value_ratio`

Loan term:

- `loan_term`

Property characteristics:

- `occupancy_type`
- `construction_method`
- `total_units`

“Business-priority” describes the order in which evidence should be examined;
it is not a keep decision.

### 8.2 Evidence-required candidates — 18 columns

Submission channel and institutional relationship:

- `preapproval`
- `submission_of_application`
- `initially_payable_to_institution`
- `conforming_loan_limit`

Special repayment structure:

- `negative_amortization`
- `interest_only_payment`
- `balloon_payment`
- `other_nonamortizing_features`

Manufactured-home and multifamily fields:

- `manufactured_home_secured_property_type`
- `manufactured_home_land_property_interest`
- `multifamily_affordable_units`

Credit-score model type:

- `applicant_credit_score_type`
- `co-applicant_credit_score_type`

These two fields identify a score model/type, not an applicant's numeric credit
score.

Automated Underwriting System slots:

- `aus-1`
- `aus-2`
- `aus-3`
- `aus-4`
- `aus-5`

The AUS slots represent one multi-valued concept. Slot position must not be
treated as an ordinal business meaning.

### 8.3 Alternative representations — 2 columns

- `derived_loan_product_type`
- `derived_dwelling_category`

These are candidates for representation comparison, not additions that may be
silently combined with all related raw fields. Choosing between raw and derived
representations belongs to feature governance.

## 9. Columns excluded from Dataset B — 54 columns

Exclusion from Dataset B means the column is not carried into the controlled
Dataset B view. It does not authorize deletion or mutation of the raw source.

The separately approved RQ3 fairness contract defines one controlled exception
for `derived_race`: it may be read from the immutable raw source into a
separate RQ3-only audit view through the one-based source-row number encoded in
the numeric suffix of `record_id`; dataframe position is `suffix - 1`. This
exception does not add `derived_race` to the
controlled model view or predictor matrix and may not change the cohort, split,
labels, stored predictions, or frozen Validation thresholds. The 99-column role
arithmetic in this contract remains unchanged.

### 9.1 Outcome-dependent or direct-leakage fields — 15 columns

- `purchaser_type`
- `interest_rate`
- `rate_spread`
- `hoepa_status`
- `total_loan_costs`
- `total_points_and_fees`
- `origination_charges`
- `discount_points`
- `lender_credits`
- `prepayment_penalty_term`
- `intro_rate_period`
- `denial_reason-1`
- `denial_reason-2`
- `denial_reason-3`
- `denial_reason-4`

These fields are formed after, depend strongly on, or directly expose the
decision and its downstream process. They must not be used to predict approval.

### 9.2 Other fields outside the retained roles — 39 columns

Year and non-retained geography:

- `activity_year`
- `derived_msa-md`
- `state_code`
- `census_tract`

Derived protected attributes:

- `derived_ethnicity`
- `derived_race`
- `derived_sex`

Ethnicity and observation flags:

- `applicant_ethnicity-1`
- `applicant_ethnicity-2`
- `applicant_ethnicity-3`
- `applicant_ethnicity-4`
- `applicant_ethnicity-5`
- `co-applicant_ethnicity-1`
- `co-applicant_ethnicity-2`
- `co-applicant_ethnicity-3`
- `co-applicant_ethnicity-4`
- `co-applicant_ethnicity-5`
- `applicant_ethnicity_observed`
- `co-applicant_ethnicity_observed`

Co-applicant race and race-observation flags:

- `co-applicant_race-1`
- `co-applicant_race-2`
- `co-applicant_race-3`
- `co-applicant_race-4`
- `co-applicant_race-5`
- `applicant_race_observed`
- `co-applicant_race_observed`

Co-applicant sex and sex-observation flags:

- `co-applicant_sex`
- `applicant_sex_observed`
- `co-applicant_sex_observed`

Age fields outside the retained metadata role:

- `co-applicant_age`
- `applicant_age_above_62`
- `co-applicant_age_above_62`

Census-tract context:

- `tract_population`
- `tract_minority_population_percent`
- `ffiec_msa_md_median_family_income`
- `tract_to_msa_income_percentage`
- `tract_owner_occupied_units`
- `tract_one_to_four_family_homes`
- `tract_median_age_of_housing_units`

`activity_year` and `state_code` are constant within the NY 2024 study scope.
`county_code` is already retained in the audit-only role; more granular or
contextual geography remains excluded under this contract.

## 10. Mandatory stable gates

### 10.1 Schema and role reconciliation

- The raw input must expose exactly the expected 99 unique HMDA headers.
- The four role inventories must remain disjoint and total 99 columns.
- Required target, selected model-input, and audit columns must exist in the
  source header before a controlled view is built.
- The raw source must not be modified in place.
- `loan_to_value_ratio` is the raw column name used by this dataset.
  `combined_loan_to_value_ratio` must not be treated as a second feature without
  an explicit schema mapping and contract change.

### 10.2 Target and cohort integrity

- Only `action_taken` values `1`, `2`, and `3` enter the cohort.
- Values `1` and `2` map to `y_approved = 1`; value `3` maps to
  `y_approved = 0`.
- `reverse_mortgage`, `open-end_line_of_credit`, and
  `business_or_commercial_purpose` must not filter rows.
- Raw `action_taken`, derived `y_approved`, and `record_id` must remain outside
  `X`.

### 10.3 Predictor/audit/target separation

- Model-input and audit-only columns must not overlap.
- Target source, derived target, identifiers, and audit-only fields must not
  enter the main-model predictor matrix.
- A candidate becomes a model input only through feature governance and the
  active machine-readable feature contract.
- A model-input decision does not change the underlying 99-column role
  reconciliation.

### 10.4 Lineage and timing

- Every retained candidate must have a documented source column and
  prediction-time availability assessment.
- Information formed after or because of the approval decision is prohibited
  from predictor use.
- Raw tokens must remain traceable through `record_id` and the `run_id` of each
  downstream execution.
- Holdout outcomes must not be used to decide candidate roles, repair timing
  violations, or reopen excluded leakage fields.

### 10.5 Data-quality semantics

For every predictor candidate, evidence must distinguish:

- observed values;
- null, blank, and whitespace-only values;
- literal `NA`;
- literal `Exempt`;
- valid not-applicable or exemption codes such as `1111` where defined by the
  source schema;
- invalid or out-of-domain tokens.

These states must not be merged merely because they all lack an immediately
usable numeric value. This requirement defines data meaning, not a particular
imputation or encoding implementation.

Quality review must cover valid domains, units, coverage, rare categories,
class-conditional missingness, organization concentration, and robust outlier
diagnostics where applicable. High missingness alone neither keeps nor excludes
a candidate.

#### 10.5.1 Final model-input quality boundary

[Feature governance](feature_governance.md) publishes the feature-selection
decision. This data contract imports, but does not reopen, its ordered 15-field
`FINAL_MODEL_FEATURES` interface for the final input-quality gate:

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

The gate must receive exactly these fields in this order. Target source,
derived target, `record_id`, protected attributes, and audit-only fields remain
outside model `X`.

#### 10.5.2 Blank-row gate

For the 15 final model inputs, a blank is any null value, empty string,
whitespace-only string, or case-insensitive textual `nan` after surrounding
whitespace is removed. Before model preprocessing begins:

- reject the affected row from the model-input population when any one of the
  15 fields is blank;
- record the input row count, retained row count, rejected `record_id`, total
  rejected count, and per-column blank counts;
- do not impute a blank or convert it to a `MISSING` category;
- preserve the frozen raw source unchanged.

Row rejection is a data-quality action, not a new target, feature-selection, or
modeling rule. If a controlled dataset has zero blank final inputs, this gate
must preserve all rows and their order.

#### 10.5.3 Raw exemption-code normalization

Raw HMDA token `1111` is a valid reporting-exemption code only in the five
final categorical flags listed below. After raw-domain validation and before
any encoder or native model adapter, deterministically normalize it to the
model-facing category `Exempt`:

| Raw fields | Accepted raw tokens | Model-facing semantic tokens |
| --- | --- | --- |
| `reverse_mortgage`, `open-end_line_of_credit`, `business_or_commercial_purpose` | `1`, `2`, `1111` | `1`, `2`, `Exempt` |
| `interest_only_payment` | `1`, `2`, `1111` | `1`, `2`, `Exempt` |
| `balloon_payment` | `1`, `2`, `1111` | `1`, `2`, `Exempt` |

This mapping is field-specific, not a global replacement rule:

- numeric `income = 1111` remains the observed numeric value `1111`;
- literal `NA` remains `NA` where its field schema permits it;
- literal `Exempt` remains `Exempt` in fields whose raw schema uses that
  literal, including `loan_to_value_ratio` and `loan_term`;
- literal `Exempt` is not a valid raw-source substitute for `1111` in the five
  exemption-coded flags, although `Exempt` is their required semantic output;
- `1111` in another field is handled by that field's own schema and is invalid
  unless the schema explicitly permits it.

The raw audit must preserve `raw_1111_count` by field. The normalized audit
must preserve `normalized_exempt_count` by field. For each of the five flags,
these counts must be equal, the normalization must not change row count or
order, and no semantic/model-facing `1111` value may remain. `Exempt` must not
be merged with code `2`, `NA`, a blank, or any missing state.

#### 10.5.4 Final input-quality PASS/FAIL gate

Downstream preprocessing or training may begin only when all applicable checks
pass:

1. the ordered 15-field schema and predictor/audit/target separation are exact;
2. `record_id` is non-empty and unique within the controlled input, and its row
   alignment is preserved through validation and normalization;
3. the blank-row audit is complete and no retained row contains a blank final
   input;
4. every raw category is valid for its field and every invalid or unknown token
   causes failure rather than coercion;
5. numeric values parse under their field rules and are finite;
6. target values and `y_approved` mapping satisfy §10.2;
7. raw `1111` and normalized `Exempt` counts agree for all five exemption-coded
   flags, with no model-facing `1111` remaining;
8. normalization preserves retained row count, order, target alignment, and
   `record_id` alignment.

The gate returns `PASS` only when every applicable check succeeds. Any failed
check returns `FAIL`, stops the run before fitted preprocessing or model state
is created, and reports the failed rule with affected columns and row counts.

### 10.6 Conditional applicability

- Manufactured-home fields must be assessed within applicable property and
  construction records.
- `multifamily_affordable_units` must be assessed within an applicable
  multifamily scope.
- AUS slots must preserve multi-valued-group semantics and distinguish valid
  not-applicable, exempt, and missing states.
- A structurally inapplicable field must not be converted into an arbitrary
  numeric value at the data-contract layer.

## 11. Ownership handoff

This contract hands the 35-column candidate universe and mandatory gates to
`docs/handbook/contracts/feature_governance.md`. That document owns evidence and
keep/exclude/alternative decisions.

Preprocessing documents may define parsing, encoding, imputation, scaling, and
model-family input containers, but they must preserve the meanings and role
separations defined here.

`docs/handbook/STATUS.md` may report execution progress but must not redefine
the data contract.

## 12. Source traceability

Project sources used to reconstruct this contract:

- [Documentation migration record](../archive/documentation_migration_2026-09.md)
  — identifies `Dataset_B_Data_Selection.md` as a former source and records
  how its role inventory and gate narrative were incorporated here;
- [Feature governance](feature_governance.md) §§9 and 11 — publishes the final
  15-field model-input list and 12-player explanation mapping imported by the
  final input-quality gate;
- `data/raw/state_NY_2024.csv` — frozen 99-column raw header used for exact role
  reconciliation;
- `config/dataset_b/feature_set_v1.yaml` — machine-readable protected-metadata,
  target, and role-separation evidence; the final 15-field model-input list and
  12-player explanation mapping are owned by the feature-governance contract;
- `src/dataset_b/data/dataset_view.py` — cohort filtering, target mapping,
  record-ID generation, and controlled-view construction;
- `src/dataset_b/data/split_inputs.py` — enforced model/audit/target
  separation at the downstream input boundary;
- `tests/test_dataset_b_view.py` — executable evidence for cohort filtering,
  target mapping, raw-token preservation, required columns, and role overlap
  rejection.

Related project reports, retained unchanged outside the handbook:

- `docs/Project_planning_recover.docx`;
- `docs/Yêu cầu Kỹ thuật Set A cho RQ1 và RQ2 - HMDA NY 2024 - Revised.docx`;
- `docs/Yêu cầu Kỹ thuật Set B cho RQ1 và RQ2 và RQ3 - HMDA NY 2024 - Revised.docx`.

Official HMDA references inherited from the legacy data-selection register:

- [2024 HMDA national loan-level dataset](https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/2024)
- [2024 Reportable HMDA Data: Regulatory and Reporting Overview Reference Chart](https://files.consumerfinance.gov/f/documents/cfpb_reportable-hmda-data_regulatory-and-reporting-overview-reference-chart_2024-10.pdf)
- [2024 HMDA Filing Instructions Guide](https://files.ffiec.cfpb.gov/documentation/2024-hmda-fig.pdf)
- [HMDA Data Browser — 2024](https://ffiec.cfpb.gov/data-browser/maps/2024)
